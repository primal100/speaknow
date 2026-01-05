from .base import BaseAIService
from .openai_gpt_realtime import OpenAIGPTRealtime
from .google_gemini_flash_audio import GeminiLiveService
from typing import Any


ai_services = {
    "openai": OpenAIGPTRealtime,
    "google": GeminiLiveService
}


AI_SERVICES_SELECTION = [("OpenAI GPT", "openai"), ("Google Gemini", "google")]


def get_ai_service(user_config: dict[str, Any]) -> BaseAIService:
    ai_service_cls = ai_services[user_config['ai_service']]
    ai_service = ai_service_cls(user_config)
    return ai_service
