import yaml
from pathlib import Path


class ConfigManager:
    def __init__(self):
        self.config_dir = Path.home() / ".ai_realtime_gui"
        self.config_file = self.config_dir / "config.yaml"
        self.config_dir.mkdir(exist_ok=True)

        # Default values if file doesn't exist
        self.defaults = {
            "model": "gpt-realtime-mini",
            "mode": "manual",
            "prompt": "You are a quiz contestant who answers concisely and clearly and as quickly as possible with just the answer, no need for a full sentence. If the question is a statement or a true/false question, answer true or false first and then give extra context",
            "play_audio": True,
            "output_modalities": ["text", "audio"],
            "transcription_enabled": True,
            "transcription_model": "gpt-4o-mini-transcribe",
            "language": "en",
            "immediate_initialisation": False,
            "save_silence_multiplier": 0,
            "save_speech_multiplier": 0
        }

    def load(self) -> dict:
        if not self.config_file.exists():
            self.save(self.defaults)
            return self.defaults
        with open(self.config_file, "r") as f:
            return {**self.defaults, **yaml.safe_load(f)}

    def save(self, data: dict):
        with open(self.config_file, "w") as f:
            yaml.dump(data, f)
