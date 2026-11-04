import os
from contextlib import asynccontextmanager
from typing import Any

import httpx

from .openai_gpt_realtime import OpenAIGPTRealtime


class ElevenLabsRealTime(OpenAIGPTRealtime):
    """ElevenLabs Realtime via its OpenAI-compatible WebSocket protocol."""

    prefix = "elevenlabs"
    transcription_pricing_cls = None
    log_token_usage = False

    @classmethod
    def set_default_config_options_on_change(cls) -> dict[str, Any]:
        return {
            "model": "eleven_realtime_v1_mini",
            "base_url": "https://api.elevenlabs.io/v1/convai",
            "api_key_env": "ELEVENLABS_API_KEY",
            "mode": "semantic",
            "audio": True,
            "transcription_enabled": True,
            "transcription_model": "",
            "save_token_data": False,
            "DISABLED": (
                "mode", "audio", "transcription_enabled", "transcription_model",
                "language", "save_token_data", "save_result",
            ),
        }

    def _session_config(self) -> dict[str, Any]:
        # Use the documented ElevenLabs session schema; transcription is managed
        # by ElevenLabs rather than a separately selected OpenAI model.
        return {
            "type": "realtime",
            "model": self.user_config["model"],
            "instructions": self.user_config["prompt"],
            "output_modalities": ["audio"],
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": 24000},
                    "turn_detection": {"type": "semantic_vad", "eagerness": "high"},
                },
                "output": {"format": {"type": "audio/pcm"}, "voice": "brian"},
            },
        }

    @asynccontextmanager
    async def _connect(self):
        # Mint a fresh credential for each connection, using the same session
        # configuration as session.update. Never send the account key over WS.
        url = str(self.client.base_url).rstrip("/") + "/realtime/client_secrets"
        async with httpx.AsyncClient(timeout=30) as http_client:
            response = await http_client.post(
                url,
                headers={"xi-api-key": os.environ[self.user_config["api_key_env"]]},
                json={
                    "session": self._session_config(),
                    "expires_after": {"anchor": "created_at", "seconds": 600},
                },
            )
            response.raise_for_status()
            secret = response.json().get("value")
            if not isinstance(secret, str) or not secret:
                raise ValueError("ElevenLabs did not return a realtime client secret")

        async with self.client.realtime.connect(
            model=self.user_config["model"],
            extra_headers={"Authorization": f"Bearer {secret}"},
        ) as connection:
            yield connection

    async def cleanup_resources(self):
        try:
            await super().cleanup_resources()
        finally:
            await self.client.close()

    async def request_response(self) -> None:
        # ElevenLabs determines turn boundaries using semantic VAD.
        pass

    async def request_response_if_manual_mode(self) -> None:
        pass
