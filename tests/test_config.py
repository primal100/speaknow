import pytest
import yaml


def test_first_load_creates_default_configuration():
    from speaknow_ai_realtime_text_to_speech.config import ConfigManager
    manager = ConfigManager()
    assert not manager.config_file.exists()
    result = manager.load()
    assert yaml.safe_load(manager.config_file.read_text()) == result == manager.defaults


def test_partial_config_preserves_overrides_and_adds_defaults():
    from speaknow_ai_realtime_text_to_speech.config import ConfigManager
    manager = ConfigManager()
    manager.config_file.write_text("prompt: Custom instructions\nplay_audio: false\ncustom: retained\n")
    loaded = manager.load()
    assert loaded["prompt"] == "Custom instructions"
    assert loaded["play_audio"] is False
    assert loaded["custom"] == "retained"
    assert loaded["model"] == manager.defaults["model"]


def test_config_round_trip_preserves_types_and_unicode(config):
    from speaknow_ai_realtime_text_to_speech.config import ConfigManager
    manager = ConfigManager()
    config.update(prompt="Bonjour — 你好\nSecond line", output_modalities=["text"], play_audio=False)
    manager.save(config)
    assert manager.load() == config


def test_default_base_url_can_be_overridden(monkeypatch):
    from speaknow_ai_realtime_text_to_speech.config import ConfigManager
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.invalid/v1")
    assert ConfigManager().defaults["base_url"] == "https://example.invalid/v1"


@pytest.mark.parametrize("contents", ["", "# No settings yet\n"])
def test_empty_configuration_uses_defaults(contents):
    from speaknow_ai_realtime_text_to_speech.config import ConfigManager
    manager = ConfigManager()
    manager.config_file.write_text(contents)
    assert manager.load() == manager.defaults


@pytest.mark.parametrize("getter,relative,is_directory", [
    ("get_config_file", "config.yaml", False),
    ("get_log_config_file", "logging.conf", False),
    ("get_log_dir", "logs", True),
    ("get_token_dir", "tokens", True),
])
def test_application_paths_are_created_idempotently(getter, relative, is_directory):
    from speaknow_ai_realtime_text_to_speech import directories
    path = getattr(directories, getter)()
    assert path == directories.HOME / relative
    assert (path if is_directory else path.parent).is_dir()
    assert getattr(directories, getter)() == path


def test_bundled_logging_config_exists():
    from speaknow_ai_realtime_text_to_speech.directories import get_default_log_config_file
    assert get_default_log_config_file().is_file()


def test_log_config_replaces_all_placeholders_and_is_idempotent(tmp_path):
    from speaknow_ai_realtime_text_to_speech.utils import update_log_config
    path = tmp_path / "logging.conf"
    path.write_text("a=%BASE_LOG_DIR%/app.log\nb=%BASE_LOG_DIR%/events.log\n")
    log_dir = tmp_path / "logs with spaces"
    update_log_config(path, log_dir)
    expected = f"a={log_dir.as_posix()}/app.log\nb={log_dir.as_posix()}/events.log\n"
    assert path.read_text() == expected
    update_log_config(path, log_dir)
    assert path.read_text() == expected


def test_missing_log_config_is_reported_without_creating_file(tmp_path, capsys):
    from speaknow_ai_realtime_text_to_speech.utils import update_log_config
    path = tmp_path / "missing.conf"
    update_log_config(path, tmp_path)
    assert not path.exists()
    assert "not found" in capsys.readouterr().out
