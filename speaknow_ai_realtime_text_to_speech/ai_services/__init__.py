from .base import BaseAIService
from .openai_gpt_realtime import OpenAIGPTRealtime
from typing import Any


ai_services = {
    "openai": OpenAIGPTRealtime,
}


AI_SERVICES_DEFAULT_MODEL = {
    "openai": {
        'realtime': "gpt-realtime-mini",
        'transcription': "gpt-4o-mini-transcribe"
    }
}

AI_SERVICES_SELECTION = [("OpenAI GPT", "openai")]


def get_ai_service(user_config: dict[str, Any]) -> BaseAIService:
    ai_service_cls = ai_services[user_config['ai_service']]
    ai_service = ai_service_cls(user_config)
    return ai_service
