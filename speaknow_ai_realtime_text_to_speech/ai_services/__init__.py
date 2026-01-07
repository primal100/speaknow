from .base import BaseAIService
from .openai_gpt_realtime import OpenAIGPTRealtime
from .google_gemini_flash_audio import GeminiLiveService
from .xai_grok_voice import XAIGrokVoice
from typing import Any


ai_services = {
    "openai": OpenAIGPTRealtime,
    "google": GeminiLiveService,
    "xai": XAIGrokVoice
}


AI_SERVICES_SELECTION = [("OpenAI GPT", "openai"), ("Google Gemini", "google"), ("xAI Grok Voice", "xai")]
DEFAULT_AI_SERVICE = "openai"
DEFAULT_AI_SERVICE_MODEL = "gpt-realtime-mini"
DEFAULT_AI_SERVICE_TRANSCRIPTION_MODEL = "gpt-4o-mini-transcribe"


def get_ai_service(user_config: dict[str, Any]) -> BaseAIService:
    ai_service_cls = ai_services[user_config['ai_service']]
    ai_service = ai_service_cls(user_config)
    return ai_service
