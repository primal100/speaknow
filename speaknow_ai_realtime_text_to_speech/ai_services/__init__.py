from .base import BaseAIService
from .openai_gpt_live import OpenAIGPTLive
from .openai_gpt_realtime import OpenAIGPTRealtime
from .google_gemini_flash_audio import GeminiLiveService
from .xai_grok_voice import XAIGrokVoice
from .elevenlabs_realtime import ElevenLabsRealTime
from typing import Any


ai_services = {
    "openai_live": OpenAIGPTLive,
    "openai": OpenAIGPTRealtime,
    "google": GeminiLiveService,
    "xai": XAIGrokVoice,
    "elevenlabs": ElevenLabsRealTime,
}


AI_SERVICES_SELECTION = [
    ("OpenAI GPT-Live", "openai_live"), ("OpenAI GPT Realtime", "openai"),
    ("Google Gemini", "google"),
    ("xAI Grok Voice", "xai"), ("ElevenLabs Realtime", "elevenlabs"),
]
DEFAULT_AI_SERVICE = "openai_live"
DEFAULT_AI_SERVICE_MODEL = OpenAIGPTLive.set_default_config_options_on_change()["model"]
DEFAULT_AI_SERVICE_TRANSCRIPTION_MODEL = OpenAIGPTLive.set_default_config_options_on_change()["transcription_model"]


def get_ai_service(user_config: dict[str, Any]) -> BaseAIService:
    ai_service_cls = ai_services[user_config['ai_service']]
    ai_service = ai_service_cls(user_config)
    return ai_service
