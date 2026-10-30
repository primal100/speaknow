import asyncio
import base64
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest


class FakeConnection:
    """Finite provider stream with the async interface used by the adapter."""

    def __init__(self, events=()):
        self.events = events
        self.session = NS(update=AsyncMock())
        self.send = AsyncMock()
        self.input_audio_buffer = NS(append=AsyncMock(), commit=AsyncMock())
        self.response = NS(create=AsyncMock())

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def __aiter__(self):
        for event in self.events:
            if isinstance(event, BaseException):
                raise event
            yield event


async def receive(service, events=()):
    connection = FakeConnection(events)
    service.client.realtime.connect.return_value = connection
    queue = asyncio.Queue()
    await service.handle_realtime_connection(queue)
    return connection, [queue.get_nowait() for _ in range(queue.qsize())]


@pytest.mark.parametrize("name,class_name", [
    ("openai", "OpenAIGPTRealtime"), ("google", "GeminiLiveService"), ("xai", "XAIGrokVoice"),
])
def test_factory_selects_provider_without_opening_connection(monkeypatch, config, name, class_name):
    from speaknow_ai_realtime_text_to_speech import ai_services
    from speaknow_ai_realtime_text_to_speech.ai_services import openai_gpt_realtime, google_gemini_flash_audio
    monkeypatch.setattr(openai_gpt_realtime, "AsyncOpenAI", Mock())
    monkeypatch.setattr(google_gemini_flash_audio.genai, "Client", Mock())
    config["ai_service"] = name
    service = ai_services.get_ai_service(config)
    assert type(service).__name__ == class_name
    assert service.user_config == config


def test_factory_rejects_unknown_provider(config):
    from speaknow_ai_realtime_text_to_speech.ai_services import get_ai_service
    config["ai_service"] = "unknown"
    with pytest.raises(KeyError, match="unknown"):
        get_ai_service(config)


@pytest.mark.parametrize("mode,expected", [
    ("manual", None),
    ("server", {"type": "server_vad", "idle_timeout_ms": 5000}),
    ("semantic", {"type": "semantic_vad", "eagerness": "high"}),
])
@pytest.mark.parametrize("transcription", [False, True])
async def test_session_configuration_matches_user_settings(openai_service, mode, expected, transcription):
    service = openai_service
    service.user_config.update(mode=mode, transcription_enabled=transcription)
    connection, events = await receive(service)
    sent = connection.session.update.call_args.kwargs["session"]
    assert sent["audio"]["input"]["turn_detection"] == expected
    assert sent["audio"]["input"]["transcription"] == (
        {"language": "en", "model": service.user_config["transcription_model"]} if transcription else None
    )
    assert sent["instructions"] == service.user_config["prompt"]
    assert sent["output_modalities"] == service.user_config["output_modalities"]
    assert events == []
    assert service.connection is None
    assert not service.connected.is_set()


async def test_session_and_speech_events_are_forwarded(openai_service):
    session = NS(id="session-1", model="test-model")
    _, events = await receive(openai_service, [
        NS(type="session.created", session=session),
        NS(type="session.updated", session=session),
        NS(type="input_audio_buffer.speech_started", item_id="item-1"),
        NS(type="input_audio_buffer.speech_stopped", item_id="item-1"),
    ])
    assert events == [
        {"type": "session_id", "session_id": "session-1"},
        {"type": "session_updated"},
        {"type": "session_id", "session_id": "test-model session-1"},
        {"type": "speech_started", "item_id": "item-1"},
        {"type": "speech_stopped", "item_id": "item-1"},
    ]


@pytest.mark.parametrize("event_type,output_type", [
    ("response.output_text.delta", "response_received"),
    ("response.output_audio_transcript.delta", "response_received"),
    ("conversation.item.input_audio_transcription.delta", "transcription_received"),
])
async def test_text_deltas_accumulate_independently_per_item(openai_service, event_type, output_type):
    _, events = await receive(openai_service, [
        NS(type=event_type, item_id="a", delta="Hello"),
        NS(type=event_type, item_id="b", delta="Other"),
        NS(type=event_type, item_id="a", delta=" world"),
    ])
    assert [e["text"] for e in events] == ["Hello", "Other", "Hello world"]
    assert all(e["type"] == output_type for e in events)


async def test_audio_events_decode_bytes_and_mark_first_chunk(openai_service):
    encoded = base64.b64encode(b"\x01\x00\xff\x7f").decode()
    _, events = await receive(openai_service, [
        NS(type="response.output_audio.delta", item_id=item, delta=encoded)
        for item in ("a", "a", "b")
    ])
    assert [e["is_first_in_response"] for e in events] == [True, False, True]
    assert all(e["data"] == b"\x01\x00\xff\x7f" for e in events)


@pytest.mark.parametrize("already_sent,responding,cancel", [
    (False, False, False), (False, True, True), (True, True, False),
])
async def test_microphone_audio_encoding_and_initial_cancel(openai_service, already_sent, responding, cancel):
    connection = FakeConnection()
    openai_service.connection = connection
    openai_service.connected.set()
    if responding:
        openai_service.response_in_progress.set()
    data = np.array([0, -32768, 32767], dtype=np.int16)
    assert await openai_service.send_audio(data, already_sent) is True
    encoded = connection.input_audio_buffer.append.call_args.kwargs["audio"]
    assert base64.b64decode(encoded) == data.tobytes()
    if cancel:
        connection.send.assert_awaited_once_with({"type": "response.cancel"})
    else:
        connection.send.assert_not_awaited()


@pytest.mark.parametrize("turn_detection", [None, {"type": "server_vad"}])
async def test_manual_response_commits_before_creating_response(openai_service, turn_detection):
    connection = FakeConnection()
    order = []
    connection.input_audio_buffer.commit.side_effect = lambda: order.append("commit")
    connection.response.create.side_effect = lambda: order.append("create")
    openai_service.connection = connection
    openai_service.connected.set()
    openai_service.session = NS(audio=NS(input=NS(turn_detection=turn_detection)))
    await openai_service.request_response_if_manual_mode()
    assert order == (["commit", "create"] if turn_detection is None else [])


@pytest.mark.parametrize("failure", [RuntimeError("stream broke"), asyncio.CancelledError()])
async def test_stream_failure_propagates_and_clears_connection(openai_service, failure):
    with pytest.raises(type(failure)):
        await receive(openai_service, [failure])
    assert openai_service.connection is None
    assert not openai_service.connected.is_set()


async def test_connection_startup_failure_preserves_original_error(openai_service):
    openai_service.client.realtime.connect.side_effect = RuntimeError("Connection refused")
    with pytest.raises(RuntimeError, match="Connection refused"):
        await openai_service.handle_realtime_connection(asyncio.Queue())
    assert openai_service.connection is None
    assert not openai_service.connected.is_set()


async def test_completed_response_logs_transcript_and_usage(openai_service):
    usage = NS(total_tokens=10)
    openai_service.write_realtime_tokens_wrapper = AsyncMock()
    await receive(openai_service, [
        NS(type="response.created", response=NS(id="r1")),
        NS(type="response.done", response=NS(
            id="r1", status="completed", status_details=None, usage=usage,
            output=[NS(content=[NS(transcript="Spoken answer")])],
        )),
    ])
    assert not openai_service.response_in_progress.is_set()
    openai_service.write_realtime_tokens_wrapper.assert_awaited_once_with(
        openai_service.user_config["model"], "Spoken answer", usage,
    )


async def test_completed_transcription_without_deltas_logs_final_text(openai_service):
    usage = object()
    openai_service.write_realtime_transcribe_tokens_wrapper = AsyncMock()
    await receive(openai_service, [NS(
        type="conversation.item.input_audio_transcription.completed",
        item_id="a", transcript="Question", usage=usage,
    )])
    openai_service.write_realtime_transcribe_tokens_wrapper.assert_awaited_once_with(
        openai_service.user_config["transcription_model"], "Question", usage,
    )


async def test_gemini_does_not_send_without_session(gemini_service):
    assert await gemini_service.send_audio(np.array([1], dtype=np.int16), False) is False


@pytest.mark.parametrize("data,expected", [
    (np.array([-32768, 0, 32767], dtype=np.int16), [-32768, 0, 32767]),
    (np.array([-1.0, 0.0, 1.0], dtype=np.float32), [-32767, 0, 32767]),
])
async def test_gemini_sends_pcm16_with_sample_rate(gemini_service, data, expected):
    gemini_service.session = NS(send_realtime_input=AsyncMock())
    assert await gemini_service.send_audio(data, False) is True
    blob = gemini_service.session.send_realtime_input.call_args.kwargs["audio"]
    assert np.frombuffer(blob.data, dtype=np.int16).tolist() == expected
    assert blob.mime_type == "audio/pcm;rate=24000"


async def test_gemini_streams_transcripts_audio_and_usage_then_cleans_up(gemini_service):
    def content(**values):
        return NS(**{
            "input_transcription": None, "output_transcription": None,
            "model_turn": None, "generation_complete": False,
            "turn_complete": False, "interrupted": False, **values,
        })

    usage = object()
    messages = [
        NS(server_content=content(input_transcription=NS(text="Question")), usage_metadata=None),
        NS(server_content=content(output_transcription=NS(text="An")), usage_metadata=None),
        NS(server_content=content(output_transcription=NS(text=" answer"),
           model_turn=NS(parts=[NS(inline_data=NS(data=b"pcm"))])), usage_metadata=None),
        NS(server_content=content(generation_complete=True), usage_metadata=None),
        NS(server_content=None, usage_metadata=usage),
    ]

    class GeminiSession(FakeConnection):
        async def receive(self):
            for message in messages:
                yield message
            # Model the user cancelling an otherwise infinite live connection.
            raise asyncio.CancelledError

    gemini_service.client.aio.live.connect.return_value = GeminiSession()
    gemini_service.write_realtime_tokens_wrapper = AsyncMock()
    queue = asyncio.Queue()
    with pytest.raises(asyncio.CancelledError):
        await gemini_service.handle_realtime_connection(queue)
    events = [queue.get_nowait() for _ in range(queue.qsize())]
    assert events[0] == {"type": "session_updated"}
    assert [e["text"] for e in events if e["type"] == "transcription_received"] == ["Question"]
    assert [e["text"] for e in events if e["type"] == "response_received"] == ["An", "An answer"]
    assert [e["data"] for e in events if e["type"] == "audio_response"] == [b"pcm"]
    gemini_service.write_realtime_tokens_wrapper.assert_awaited_once_with(
        gemini_service.user_config["model"], "An answer", usage,
    )
    assert gemini_service.session is None


def test_grok_uses_time_pricing_instead_of_token_pricing(monkeypatch, config):
    from speaknow_ai_realtime_text_to_speech.ai_services import openai_gpt_realtime
    from speaknow_ai_realtime_text_to_speech.ai_services.xai_grok_voice import XAIGrokVoice
    monkeypatch.setattr(openai_gpt_realtime, "AsyncOpenAI", Mock())
    service = XAIGrokVoice(config)
    assert service.log_token_usage is False
    assert service.log_per_interval_pricing is True
    assert service.transcription_pricing_cls is None
