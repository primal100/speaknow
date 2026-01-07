import os
from pathlib import Path
from importlib.resources import files


APP_NAME = 'Speaknow'


HOME = Path(os.environ.get('appdata', Path.home() / ".config")) / APP_NAME


def get_default_log_config_file() -> Path:
    return Path(str(files("speaknow_ai_realtime_text_to_speech") / "conf" / "logging.conf"))


def get_config_file() -> Path:
    HOME.mkdir(parents=True, exist_ok=True)
    return HOME / 'config.yaml'


def get_log_config_file() -> Path:
    HOME.mkdir(parents=True, exist_ok=True)
    return HOME / 'logging.conf'


def get_log_dir() -> Path:
    log_dir = HOME / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def get_token_dir() -> Path:
    token_dir = HOME / 'tokens'
    token_dir.mkdir(parents=True, exist_ok=True)
    return token_dir

