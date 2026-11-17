from unittest.mock import Mock

import numpy as np
import pytest


@pytest.fixture
def player(monkeypatch):
    from speaknow_ai_realtime_text_to_speech import audio_util
    stream = Mock()
    monkeypatch.setattr(audio_util.sd, "OutputStream", Mock(return_value=stream))
    player = audio_util.AudioPlayerAsync()
    yield player
    player.terminate()


def render(player, frames):
    output = np.full((frames, 1), -999, dtype=np.int16)
    player.callback(output, frames, None, None)
    return output[:, 0].tolist()


def test_empty_playback_fills_entire_buffer_with_silence(player):
    assert render(player, 4) == [0, 0, 0, 0]


def test_playback_preserves_chunk_order_and_pads_underflow(player):
    player.add_data(np.array([1, -2], dtype=np.int16).tobytes())
    player.add_data(np.array([3], dtype=np.int16).tobytes())
    assert render(player, 5) == [1, -2, 3, 0, 0]
    assert render(player, 2) == [0, 0]


def test_partial_chunk_is_preserved_for_next_callback(player):
    player.add_data(np.array([1, 2, 3, 4, 5], dtype=np.int16).tobytes())
    assert render(player, 2) == [1, 2]
    assert render(player, 4) == [3, 4, 5, 0]


def test_stream_starts_once_until_stopped(player):
    player.add_data(b"\x01\x00")
    player.add_data(b"\x02\x00")
    player.stream.start.assert_called_once()
    player.stop()
    assert player.playing is False
    assert render(player, 2) == [0, 0]
    player.stream.stop.assert_called_once()
    player.add_data(b"\x03\x00")
    assert player.stream.start.call_count == 2
    assert render(player, 1) == [3]


def test_terminate_closes_output_stream(player):
    player.terminate()
    player.stream.close.assert_called_once()


def test_device_name_uses_open_stream_device(monkeypatch, audio_backend):
    from speaknow_ai_realtime_text_to_speech.audio_util import stream_device_name

    query = Mock(return_value={"name": "USB microphone [2]"})
    monkeypatch.setattr(audio_backend, "query_devices", query)
    assert stream_device_name(Mock(device=7)) == "USB microphone [2]"
    query.assert_called_once_with(7)


def test_missing_device_metadata_does_not_interrupt_recording(monkeypatch, audio_backend):
    from speaknow_ai_realtime_text_to_speech.audio_util import stream_device_name

    monkeypatch.setattr(audio_backend, "query_devices", Mock(side_effect=RuntimeError("Device removed")))
    assert stream_device_name(Mock(device=7)) == "Unknown device"
    assert stream_device_name(None) == "Not ready"
