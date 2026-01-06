import asyncio

import google.genai.live

from speaknow_ai_realtime_text_to_speech.audio_util import SAMPLE_RATE
import logging
import numpy as np
from typing import Any

from google import genai
from google.genai import types
from gpt_token_tracker.pricing import PricingRealtime, PricingAudioTranscription
from .base import BaseAIService

log = logging.getLogger("realtime_app")


GOOGLE_MODALITY_MAP = {
    "text": types.Modality.TEXT,
    "audio": types.Modality.AUDIO,
}

events_log = logging.getLogger("events")


class GeminiLiveService(BaseAIService):
    prefix = "gemini"
    client: genai.Client
    session: google.genai.live.AsyncSession | None
    # Note: Pricing is based on 2026 rates for Gemini 2.5 Flash Native Audio
    realtime_costs = {
        "text_in": 0.50,  # per 1M tokens
        "audio_in": 3.00,  # per 1M tokens
        "text_out": 2.00,  # per 1M tokens
        "audio_out": 12.00  # per 1M tokens
    }
    # Gemini uses specific usage metadata keys
    realtime_pricing_cls = PricingRealtime
    transcription_pricing_cls = None

    @classmethod
    def set_default_config_options_on_change(cls) -> dict[str, Any]:
        return {
            'model': "gemini-2.5-flash-native-audio-preview-12-2025",
            'DISABLED': ('mode', 'transcription_model', 'audio', 'language')
        }

    def __init__(self, user_config: dict[str, Any]):
        super().__init__(user_config)
        self.client = genai.Client()
        self.session = None
        self.connected = asyncio.Event()
        self.response_in_progress = asyncio.Event()

    def write_realtime_tokens(self, model: str, result: str, usage: google.genai.types.UsageMetadata) -> None:
        log.info(usage)
        """self.token_logger_realtime.record(model, result, usage)
        self.csv_token_logger_realtime.record(model, result, usage)"""

    def write_realtime_transcribe_tokens(self, model: str, result: str, usage: Any) -> None:
        pass

    async def send_audio(self, data: np.ndarray, sent_audio: bool) -> bool:
        if not self.session:
            return False

        # 1. Ensure data is in int16 format (PCM 16-bit)
        # If data is float32 (range -1.0 to 1.0), scale it first:
        if data.dtype != np.int16:
            # Scale floats to int16 range and cast
            pcm_data = (data * 32767).astype(np.int16).tobytes()
        else:
            # Already int16, just get the raw bytes
            pcm_data = data.tobytes()

        # Gemini expects raw PCM or base64 blobs
        await self.session.send(
            input=types.LiveClientRealtimeInput(
                media_chunks=[types.Blob(data=pcm_data, mime_type=f"audio/pcm;rate={SAMPLE_RATE}")]
            )
        )
        return True

    async def handle_realtime_connection(self, event_queue: asyncio.Queue[dict[str, Any]]) -> None:
        current_transcription: str = ""
        current_output_transcription: str = ""
        model_id = self.user_config.get('model', 'gemini-2.5-flash-native-audio-preview-12-2025"')
        # Should be self.user_config.get('output_modalities') but anything other than just 'audio' is rejeceted at least as of gemini-2.5-flash-native-audio-preview-12-2025"
        output_modalities = ["audio"]
        # explicit_vad_signal gets rejected here as of gemini-2.5-flash-native-audio-preview-12-2025 "explicit_vad_signal is not supported in Gemini API"
        config = types.LiveConnectConfig(
            response_modalities=[GOOGLE_MODALITY_MAP[m] for m in output_modalities if m in GOOGLE_MODALITY_MAP],
            system_instruction=self.user_config.get('prompt'),
            input_audio_transcription = types.AudioTranscriptionConfig() if self.user_config['transcription_enabled'] else None,
            output_audio_transcription=types.AudioTranscriptionConfig(),
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=self.user_config.get('voice', 'Puck')    #TODO: Add config option for voice
                    )
                )
            )
        )

        try:
            async with self.client.aio.live.connect(model=model_id, config=config) as session:
                self.session = session
                self.connected.set()
                log.info("Gemini Live Session Started")
                event_queue.put_nowait({"type": "session_updated"})
                while True:
                    async for message in session.receive():
                        events_log.debug(message)
                        # 1. Handle Server Content (Audio/Text)
                        if server_content := getattr(message, "server_content", None):
                            events_log.info(message.server_content)
                            if input_transcription := getattr(server_content, "input_transcription", None):
                                current_transcription += input_transcription.text
                                event_queue.put_nowait({"type": "transcription_received", "item_id": None,
                                                        "text": current_transcription})
                            if output_transcription := getattr(server_content, "output_transcription", None):
                                current_output_transcription += output_transcription.text
                                event_queue.put_nowait({"type": "response_received", "item_id": None,
                                                        "text": current_output_transcription})
                            if model_turn := message.server_content.model_turn:
                                for part in model_turn.parts:
                                    if part.inline_data:
                                        # Native Audio Chunk
                                        event_queue.put_nowait({
                                            "type": "audio_response",
                                            "data": part.inline_data.data,
                                            "is_first_in_response": True
                                        })
                                    elif part.text:
                                        # Log thinking response
                                        log.info(part.text)

                            if message.server_content.generation_complete:
                                log.debug("Gemini generation complete")
                                current_transcription = ""
                                current_output_transcription = ""

                            if message.server_content.turn_complete:
                                log.debug("Gemini turn complete")
                                if current_transcription:
                                    log.info("[TRANSCRIPTION] REALTIME: %s", current_transcription)
                                current_transcription = ""
                                current_output_transcription = ""
                                self.response_in_progress.clear()

                            if message.server_content.interrupted:
                                log.info("Gemini response interrupted by user")
                                current_transcription = ""

                        # 2. Handle Usage Metadata (Tokens)
                        if message.usage_metadata:
                            usage = message.usage_metadata
                            await asyncio.to_thread(
                                self.write_realtime_tokens,
                                model_id,
                                "Gemini Audio Turn",
                                usage
                            )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.error(f"Gemini Connection Error: {e}")
        finally:
            self.connected.clear()
            self.session = None

    async def request_response(self) -> None: ...
        # Gemini triggers response automatically, not sure if it makes sense to do this using API


    async def request_response_if_manual_mode(self) -> None:
        await self.request_response()
