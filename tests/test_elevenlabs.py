import asyncio
import base64
import json
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import httpx
import numpy as np
import pytest
from websockets.exceptions import ConnectionClosedOK


@pytest.fixture
def elevenlabs(config):
    from speaknow_ai_realtime_text_to_speech.ai_services import (
        ElevenLabsRealTime,
        get_ai_service,
    )

    config.update(ElevenLabsRealTime.set_default_config_options_on_change())
    config["ai_service"] = "elevenlabs"
    return get_ai_service(config)


@pytest.fixture
def transport(monkeypatch):
    """Run the actual OpenAI SDK with fake HTTP and WebSocket transports."""
    import websockets.asyncio.client

    from speaknow_ai_realtime_text_to_speech.ai_services import elevenlabs_realtime

    requests = []
    replies = [httpx.Response(200, json={"value": "ephemeral-test-secret"})]

    def respond(request):
        requests.append(request)
        return replies.pop(0)

    real_client = httpx.AsyncClient
    clients = []

    def http_client(**kwargs):
        client = real_client(transport=httpx.MockTransport(respond), **kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr(elevenlabs_realtime.httpx, "AsyncClient", http_client)
    websocket = NS(
        recv=AsyncMock(side_effect=ConnectionClosedOK(None, None)),
        send=AsyncMock(), close=AsyncMock(),
    )
    connect = AsyncMock(return_value=websocket)
    monkeypatch.setattr(websockets.asyncio.client, "connect", connect)
    return NS(requests=requests, replies=replies, clients=clients, ws=websocket, connect=connect)


async def test_authenticated_connection_uses_ephemeral_secret_and_session(elevenlabs, transport):
    from speaknow_ai_realtime_text_to_speech.ai_services import AI_SERVICES_SELECTION

    assert ("ElevenLabs Realtime", "elevenlabs") in AI_SERVICES_SELECTION
    await elevenlabs.handle_realtime_connection(asyncio.Queue())
    request = transport.requests[0]
    assert str(request.url) == "https://api.elevenlabs.io/v1/convai/realtime/client_secrets"
    assert request.headers["xi-api-key"] == "test-key-not-a-real-credential"
    body = json.loads(request.content)
    session = body["session"]
    assert session["model"] == "eleven_realtime_v1_mini"
    assert session["instructions"] == elevenlabs.user_config["prompt"]
    assert session["audio"]["input"]["format"] == {"type": "audio/pcm", "rate": 24000}
    assert "transcription" not in session["audio"]["input"]
    assert session["audio"]["input"]["turn_detection"]["type"] == "semantic_vad"
    assert session["audio"]["output"]["voice"] == "brian"
    args, kwargs = transport.connect.call_args
    assert args[0] == "wss://api.elevenlabs.io/v1/convai/realtime?model=eleven_realtime_v1_mini"
    assert kwargs["additional_headers"]["Authorization"] == "Bearer ephemeral-test-secret"
    assert "test-key-not-a-real-credential" not in str(kwargs)
    assert json.loads(transport.ws.send.call_args.args[0])["session"] == session
    assert all(client.is_closed for client in transport.clients)
    transport.ws.close.assert_awaited_once()
    assert elevenlabs.connection is None
    assert not elevenlabs.connected.is_set()
    await elevenlabs.cleanup_resources()
    assert elevenlabs.client.is_closed()


async def test_audio_and_transcripts_flow_through_sdk(elevenlabs, transport):
    pcm = np.array([0, -32768, 32767], dtype=np.int16)
    encoded = base64.b64encode(pcm.tobytes()).decode()
    events = [
        {"type": "session.updated", "session": {"id": "s1", "model": "eleven_realtime_v1_mini"}},
        {"type": "response.output_audio.delta", "item_id": "a", "delta": encoded},
        {"type": "response.output_audio_transcript.delta", "item_id": "a", "delta": "Hello"},
        {"type": "conversation.item.input_audio_transcription.completed", "item_id": "u", "transcript": "Hi"},
    ]
    transport.ws.recv.side_effect = [json.dumps(e).encode() for e in events] + [ConnectionClosedOK(None, None)]
    queue = asyncio.Queue()
    await elevenlabs.handle_realtime_connection(queue)
    received = [queue.get_nowait() for _ in range(queue.qsize())]
    assert any(e["type"] == "session_updated" for e in received)
    assert next(e["data"] for e in received if e["type"] == "audio_response") == pcm.tobytes()
    assert [e["text"] for e in received if "text" in e] == ["Hello", "Hi"]
    transport.replies.append(httpx.Response(200, json={"value": "new-secret"}))
    async with elevenlabs._connect() as connection:
        elevenlabs.connection = connection
        elevenlabs.connected.set()
        assert await elevenlabs.send_audio(pcm, False)
    assert json.loads(transport.ws.send.call_args.args[0]) == {
        "type": "input_audio_buffer.append", "audio": encoded,
    }
    assert transport.connect.call_args.kwargs["additional_headers"]["Authorization"] == "Bearer new-secret"
    await elevenlabs.cleanup_resources()


@pytest.mark.parametrize("response,error", [
    (httpx.Response(401, json={"detail": "Invalid API key"}), httpx.HTTPStatusError),
    (httpx.Response(200, json={}), ValueError),
    (httpx.Response(200, json={"value": ""}), ValueError),
])
async def test_auth_failure_does_not_open_websocket(elevenlabs, transport, response, error):
    transport.replies[:] = [response]
    with pytest.raises(error):
        await elevenlabs.handle_realtime_connection(asyncio.Queue())
    transport.connect.assert_not_awaited()
    assert all(client.is_closed for client in transport.clients)
    assert elevenlabs.connection is None
    assert not elevenlabs.connected.is_set()
    await elevenlabs.cleanup_resources()


@pytest.mark.parametrize("failure", [RuntimeError("stream failed"), asyncio.CancelledError()])
async def test_stream_failure_closes_socket_and_clears_state(elevenlabs, transport, failure):
    transport.ws.recv.side_effect = failure
    with pytest.raises(type(failure)):
        await elevenlabs.handle_realtime_connection(asyncio.Queue())
    transport.ws.close.assert_awaited_once()
    assert elevenlabs.connection is None
    assert not elevenlabs.connected.is_set()
    await elevenlabs.cleanup_resources()


async def test_manual_controls_do_not_send_unsupported_events(elevenlabs, transport):
    await elevenlabs.request_response()
    await elevenlabs.request_response_if_manual_mode()
    transport.connect.assert_not_awaited()
    transport.ws.send.assert_not_awaited()
    await elevenlabs.cleanup_resources()
