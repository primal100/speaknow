import os
from pathlib import Path


APP_NAME = 'Speaknow'


HOME = Path(os.environ.get('appdata', Path.home() / ".config")) / APP_NAME


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


def get_recordings_dir() -> Path:
    recordings_dir = HOME / 'recordings'
    recordings_dir.mkdir(parents=True, exist_ok=True)
    return recordings_dir
