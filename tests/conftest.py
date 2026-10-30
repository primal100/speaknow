"""Offline fixtures: no real credentials, user files, or audio devices."""

import importlib
import socket
import sys
from types import ModuleType
from unittest.mock import Mock

import pytest


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch, tmp_path, audio_backend):
    from speaknow_ai_realtime_text_to_speech import directories

    monkeypatch.setattr(directories, "HOME", tmp_path / "Speaknow")
    for name in ("OPENAI_API_KEY", "GEMINI_API_KEY", "XAI_API_KEY"):
        monkeypatch.setenv(name, "test-key-not-a-real-credential")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("GEMINI_NEXT_GEN_API_BASE_URL", raising=False)

    def local_connections_only(original):
        def connect(sock, address):
            # Windows asyncio may use a loopback socket pair internally.
            if sock.family == getattr(socket, "AF_UNIX", None) or (
                isinstance(address, tuple) and address[0] in ("127.0.0.1", "::1")
            ):
                return original(sock, address)
            raise AssertionError("Tests must not open external network connections")
        return connect

    monkeypatch.setattr(socket.socket, "connect", local_connections_only(socket.socket.connect))
    monkeypatch.setattr(socket.socket, "connect_ex", local_connections_only(socket.socket.connect_ex))

    monkeypatch.setattr(audio_backend, "InputStream", Mock(side_effect=AssertionError("Unexpected microphone access")))
    monkeypatch.setattr(audio_backend, "OutputStream", Mock(side_effect=AssertionError("Unexpected speaker access")))

    # Patch HOME before importing services: base.py creates its token directory
    # at import time. Reset its cached path for each subsequent test, too.
    from speaknow_ai_realtime_text_to_speech.ai_services import base
    monkeypatch.setattr(base, "TOKENS_DIR", directories.get_token_dir())


@pytest.fixture(scope="session")
def audio_backend():
    # Importing real sounddevice initializes PortAudio. Replace it before any
    # application import so even machines without audio drivers can run tests.
    backend = ModuleType("sounddevice")
    backend.InputStream = Mock()
    backend.OutputStream = Mock()
    backend.query_devices = Mock(return_value=[])
    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(sys.modules, "sounddevice", backend)
        yield backend


@pytest.fixture
def config():
    from speaknow_ai_realtime_text_to_speech.config import ConfigManager
    return {**ConfigManager().defaults, "save_token_data": False}


@pytest.fixture
def openai_service(monkeypatch, config):
    from speaknow_ai_realtime_text_to_speech.ai_services import openai_gpt_realtime
    monkeypatch.setattr(openai_gpt_realtime, "AsyncOpenAI", Mock())
    return openai_gpt_realtime.OpenAIGPTRealtime(config)


@pytest.fixture
def gemini_service(monkeypatch, config):
    from speaknow_ai_realtime_text_to_speech.ai_services import google_gemini_flash_audio
    monkeypatch.setattr(google_gemini_flash_audio.genai, "Client", Mock())
    config.update(ai_service="google", api_key_env="GEMINI_API_KEY")
    return google_gemini_flash_audio.GeminiLiveService(config)


@pytest.fixture
def app_module(monkeypatch):
    import logging.config
    # Import-time logging configuration must not replace pytest's handlers.
    monkeypatch.setattr(logging.config, "fileConfig", Mock())
    return importlib.import_module("speaknow")
