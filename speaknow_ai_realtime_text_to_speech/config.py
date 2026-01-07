import yaml
from typing import Any
from .ai_services import DEFAULT_AI_SERVICE, DEFAULT_AI_SERVICE_MODEL, DEFAULT_AI_SERVICE_TRANSCRIPTION_MODEL
from .directories import get_config_file


class ConfigManager:
    def __init__(self):
        self.config_file = get_config_file()

        # Default values if file doesn't exist
        self.defaults = {

            "ai_service": DEFAULT_AI_SERVICE,
            "model": DEFAULT_AI_SERVICE_MODEL,
            "mode": "manual",
            "prompt": "Reply promptly. If a question is asked, answer it with just the answer.",
            "play_audio": True,
            "output_modalities": ["audio"],
            "transcription_enabled": True,
            "transcription_model": DEFAULT_AI_SERVICE_TRANSCRIPTION_MODEL,
            "language": "en",
            "immediate_initialisation": False,
            "save_token_data": True,
            "save_result": False
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
