import yaml
from typing import Any
from .ai_services import AI_SERVICES_DEFAULT_MODEL
from .directories import get_config_file


DEFAULT_AI_SERVICE = "openai"

class ConfigManager:
    def __init__(self):
        self.config_file = get_config_file()

        # Default values if file doesn't exist
        self.defaults = {
            "ai_service": DEFAULT_AI_SERVICE,
            "model": AI_SERVICES_DEFAULT_MODEL[DEFAULT_AI_SERVICE]["realtime"],
            "mode": "manual",
            "prompt": "Reply promptly. If a question is asked, answer it with just the answer.",
            "play_audio": True,
            "output_modalities": ["audio"],
            "transcription_enabled": True,
            "transcription_model": AI_SERVICES_DEFAULT_MODEL[DEFAULT_AI_SERVICE]["transcription"],
            "language": "en",
            "immediate_initialisation": False,
            "save_silence_multiplier": 0,
            "save_speech_multiplier": 0
        }

    def load(self) -> dict[str, Any]:
        if not self.config_file.exists():
            self.save(self.defaults)
            return self.defaults
        with open(self.config_file, "r") as f:
            return {**self.defaults, **yaml.safe_load(f)}

    def save(self, data: dict):
        with open(self.config_file, "w") as f:
            yaml.dump(data, f)
