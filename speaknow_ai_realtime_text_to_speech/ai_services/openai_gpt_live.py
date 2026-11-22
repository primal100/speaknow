import asyncio
import base64
from datetime import datetime
import json
import os
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from gpt_token_tracker.pricing_timed import PricingTimeInterval
from numpy import ndarray
from websockets.asyncio.client import ClientConnection, connect

from .base import BaseAIService


class OpenAIGPTLive(BaseAIService):
    """OpenAI GPT-Live over the primary audio WebSocket."""

    prefix = "openai_live"
    realtime_pricing_cls = PricingTimeInterval
    transcription_pricing_cls = None
    log_token_usage = False
    log_per_interval_pricing = True

    @classmethod
    def set_default_config_options_on_change(cls) -> dict[str, Any]:
        return {
            "model": "gpt-live-1.1",
            "base_url": os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1",
            "api_key_env": "OPENAI_API_KEY",
            "mode": "server",
            "audio": True,
            "transcription_model": "",
            "DISABLED": ("mode", "audio", "transcription_model", "language", "save_result"),
        }

    def __init__(self, user_config: dict[str, Any]):
        super().__init__(user_config)
        self.api_key = os.environ[user_config["api_key_env"]]
        self.connection: ClientConnection | None = None

    async def send_audio(self, data: ndarray, sent_audio: bool) -> bool:
        if self.connection is None:
            raise ConnectionError("GPT-Live session is disconnected")
        await self.connection.send(json.dumps({
            "type": "session.input_audio.append",
            "audio": base64.b64encode(data.tobytes()).decode("ascii"),
        }))
        return True

    async def handle_realtime_connection(self, event_queue: asyncio.Queue[dict[str, Any]]) -> None:
        config = self.user_config
        base_url = urlsplit(config["base_url"] or "https://api.openai.com/v1")
        scheme = {"https": "wss", "http": "ws"}.get(base_url.scheme, base_url.scheme)
        url = urlunsplit((scheme, base_url.netloc, base_url.path.rstrip("/") + "/live/sessions", "", ""))
        model = config["model"]
        transcripts = {"session.input_transcript.delta": "", "session.output_transcript.delta": ""}
        transcript_ends: dict[str, int] = {}
        started_at = None
        try:
            async with connect(url, additional_headers={"Authorization": f"Bearer {self.api_key}"}) as socket:
                await socket.send(json.dumps({
                    "type": "session.start",
                    "session": {
                        "model": model,
                        "instructions": config["prompt"],
                        "audio": {"format": {"type": "audio/pcm", "rate": 24000}},
                        "delegation": {"type": "client"},
                    },
                }))
                async for message in socket:
                    event = json.loads(message)
                    event_type = event["type"]
                    if event_type == "session.started":
                        started_at = datetime.now()
                        self.connection = socket
                        event_queue.put_nowait({"type": "session_id", "session_id": event["session"]["id"]})
                        event_queue.put_nowait({"type": "session_updated"})
                    elif event_type == "session.output_audio.delta":
                        event_queue.put_nowait({"type": "audio_response", "data": base64.b64decode(event["delta"])})
                    elif event_type in transcripts:
                        is_input = event_type == "session.input_transcript.delta"
                        if is_input and not config["transcription_enabled"]:
                            continue
                        # Live sends timed fragments without turn-complete events.
                        if event["start_ms"] - transcript_ends.get(event_type, 0) > 1500:
                            transcripts[event_type] = ""
                        transcripts[event_type] += event["delta"]
                        transcript_ends[event_type] = event["end_ms"]
                        event_queue.put_nowait({
                            "type": "transcription_received" if is_input else "response_received",
                            "text": transcripts[event_type],
                        })
                    elif event_type == "error":
                        raise RuntimeError(event["error"]["message"])
                    elif event_type == "session.closed":
                        break
            raise ConnectionError("GPT-Live session closed")
        finally:
            self.connection = None
            if started_at is not None:
                await self.write_connection_time(model, started_at)

    async def request_response(self) -> None:
        # GPT-Live automatically decides when to speak.
        pass

    async def request_response_if_manual_mode(self) -> None:
        pass
