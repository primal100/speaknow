import asyncio
from contextlib import asynccontextmanager
import json
from unittest.mock import AsyncMock

import numpy as np
import pytest


async def test_default_live_service_streams_audio_and_transcripts(monkeypatch, config):
    from speaknow_ai_realtime_text_to_speech.ai_services import get_ai_service, openai_gpt_live

    service = get_ai_service(config)
    events = asyncio.Queue()
    socket = AsyncMock()
    audio = np.array([1], dtype="<i2")

    async def messages():
        yield json.dumps({"type": "session.started", "session": {"id": "live_test"}})
        assert await service.send_audio(audio, False)
        for event_type, delta in (
            ("session.input_transcript.delta", "Hello"),
            ("session.output_transcript.delta", "Hi"),
            ("session.output_transcript.delta", " there"),
            ("session.output_audio.delta", "AQA="),
        ):
            yield json.dumps({"type": event_type, "delta": delta, "start_ms": 0, "end_ms": 100})

    socket.__aiter__.side_effect = messages

    @asynccontextmanager
    async def connect(url, *, additional_headers):
        assert url == "wss://api.openai.com/v1/live/sessions"
        assert additional_headers == {"Authorization": "Bearer test-key-not-a-real-credential"}
        yield socket

    monkeypatch.setattr(openai_gpt_live, "connect", connect)
    with pytest.raises(ConnectionError, match="session closed"):
        await service.handle_realtime_connection(events)

    start, append = [json.loads(call.args[0]) for call in socket.send.await_args_list]
    assert start["type"] == "session.start"
    assert start["session"]["model"] == "gpt-live-1.1"
    assert append == {"type": "session.input_audio.append", "audio": "AQA="}
    assert [events.get_nowait() for _ in range(events.qsize())] == [
        {"type": "session_id", "session_id": "live_test"},
        {"type": "session_updated"},
        {"type": "transcription_received", "text": "Hello"},
        {"type": "response_received", "text": "Hi"},
        {"type": "response_received", "text": "Hi there"},
        {"type": "audio_response", "data": audio.tobytes()},
    ]
    assert service.connection is None


def test_saved_realtime_service_keeps_its_model_defaults():
    from speaknow_ai_realtime_text_to_speech.config import ConfigManager

    manager = ConfigManager()
    manager.save({"ai_service": "openai"})
    config = manager.load()
    assert config["model"] == "gpt-realtime-2.1-mini"
    assert config["transcription_model"] == "gpt-4o-mini-transcribe"
