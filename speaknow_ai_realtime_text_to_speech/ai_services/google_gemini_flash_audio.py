import asyncio
from speaknow_ai_realtime_text_to_speech.audio_util import SAMPLE_RATE
from google.genai import types
import logging
from typing import Any

from google import genai
from google.genai import types
from .base import BaseAIService, LogWriter

log = logging.getLogger("realtime_app")


GOOGLE_MODALITY_MAP = {
    "text": types.Modality.TEXT,
    "audio": types.Modality.AUDIO,
}

events_log = logging.getLogger("events")


class GeminiLiveService(BaseAIService):
    prefix = "gemini"
    # Note: Pricing is based on 2026 rates for Gemini 2.5 Flash Native Audio
    realtime_costs = {
        "text_in": 0.10,  # per 1M tokens
        "audio_in": 0.40,  # per 1M tokens
        "text_out": 0.40,  # per 1M tokens
        "audio_out": 0.80  # per 1M tokens
    }
    # Gemini uses specific usage metadata keys
    realtime_pricing_cls = Any  # In practice, map to your specific logger

    @classmethod
    def set_default_config_options_on_change(cls) -> dict[str, Any]:
        return {
            'model': "gemini-2.5-flash-native-audio",
            'transcription_model': "gemini-2.5-flash-native-audio",
            'DISABLED': ('transcription_model',)
        }

    def __init__(self, user_config: dict[str, Any]):
        super().__init__(user_config)
        self.client = genai.Client()
        self.session: Any = None
        self.connected = asyncio.Event()
        self.response_in_progress = asyncio.Event()

    def write_realtime_tokens(self, model: str, result: str, usage: Any) -> None:
        # Gemini usage usually comes in 'usage_metadata'
        self.token_logger_realtime.record(model, result, usage)
        self.csv_token_logger_realtime.record(model, result, usage)

    async def send_audio(self, data: bytes, sent_audio: bool) -> bool:
        if not self.session:
            return False

        # Gemini expects raw PCM or base64 blobs
        await self.session.send(
            input=types.LiveClientRealtimeInput(
                media_chunks=[types.Blob(data=data, mime_type=f"audio/pcm;rate={SAMPLE_RATE}")]
            )
        )
        return True

    async def handle_realtime_connection(self, event_queue: asyncio.Queue[dict[str, Any]]) -> None:
        model_id = self.user_config.get('model', 'gemini-2.5-flash-native-audio')
        output_modalities = self.user_config.get('output_modalities')
        config = types.LiveConnectConfig(
            response_modalities=[GOOGLE_MODALITY_MAP[m] for m in output_modalities if m in GOOGLE_MODALITY_MAP],
            system_instruction=self.user_config.get('prompt'),
            explicit_vad_signal = self.user_config['turn_detection'] == "manual",
            input_audio_transcription = types.AudioTranscriptionConfig() if self.user_config['transcription_enabled'] else None,
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
                self.session = session # Todo: Check correct type here
                self.connected.set()
                log.info("Gemini Live Session Started")

                async for message in session.receive():
                    events_log.debug(message)
                    events_log.info("Event Type: %s. Item Id: %s", message.type, getattr(message, "item_id", ""))
                    # 1. Handle Server Content (Audio/Text)
                    if message.server_content:
                        model_turn = message.server_content.model_turn
                        if model_turn:
                            for part in model_turn.parts:
                                if part.inline_data:
                                    # Native Audio Chunk
                                    event_queue.put_nowait({
                                        "type": "audio_response",
                                        "data": part.inline_data.data,
                                    })
                                elif part.text:
                                    # Live Transcript
                                    event_queue.put_nowait({
                                        "type": "response_received",
                                        "text": part.text
                                    })

                        if message.server_content.turn_complete:
                            log.debug("Gemini turn complete")
                            self.response_in_progress.clear()

                        if message.server_content.interrupted:
                            log.info("Gemini response interrupted by user")
                            event_queue.put_nowait({"type": "speech_started"})

                    # 2. Handle Usage Metadata (Tokens)
                    if message.usage_metadata:
                        usage = message.usage_metadata
                        await asyncio.to_thread(
                            self.write_realtime_tokens,
                            model_id,
                            "Gemini Audio Turn",
                            usage
                        )

        except Exception as e:
            log.error(f"Gemini Connection Error: {e}")
        finally:
            self.connected.clear()
            self.session = None

    async def request_response(self) -> None:
        # Gemini triggers response automatically or via end_of_turn
        if self.session:
            await self.session.send(input="End of turn", end_of_turn=True)
            await self.session.send_re

    async def request_response_if_manual_mode(self) -> None:
        # Implementation depends on 'automatic_activity_detection' config
        if self.user_config.get('mode') == 'manual':
            await self.request_response()
