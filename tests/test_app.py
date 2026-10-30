import asyncio
from contextlib import suppress
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock, call

import pytest


@pytest.fixture
def app_state(config):
    """Small UI boundary double; the queue and application methods stay real."""
    return NS(
        user_config=config, event_queue=asyncio.Queue(),
        session_updated=asyncio.Event(), connection_cancelled=asyncio.Event(),
        should_send_audio=asyncio.Event(), audio_player=Mock(), query_one=Mock(),
        config_manager=Mock(), notify=Mock(), restart_workers=AsyncMock(),
        ai_service=NS(request_response_if_manual_mode=AsyncMock()),
    )


async def process(app_module, state, events):
    task = asyncio.create_task(app_module.RealtimeApp.process_event_queue(state))
    try:
        for event in events:
            state.event_queue.put_nowait(event)
        await asyncio.wait_for(state.event_queue.join(), timeout=1)
        # Propagate handler errors rather than leaving them as background warnings.
        if task.done():
            await task
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


@pytest.mark.parametrize("play_audio", [False, True])
async def test_audio_playback_obeys_mute_setting(app_module, app_state, play_audio):
    app_state.user_config["play_audio"] = play_audio
    await process(app_module, app_state, [{"type": "audio_response", "data": b"pcm"}])
    if play_audio:
        app_state.audio_player.add_data.assert_called_once_with(b"pcm")
    else:
        app_state.audio_player.add_data.assert_not_called()


async def test_session_events_update_display_and_signal_ready(app_module, app_state):
    display = NS(session_id=None)
    app_state.query_one.return_value = display
    await process(app_module, app_state, [
        {"type": "session_id", "session_id": "s1"}, {"type": "session_updated"},
    ])
    assert display.session_id == "s1"
    assert app_state.session_updated.is_set()


@pytest.mark.parametrize("event_type,pane", [
    ("transcription_received", "#middle-pane"), ("response_received", "#lower-middle-pane"),
])
@pytest.mark.parametrize("item_id", [None, "item-1"])
async def test_transcripts_replace_pane_content(app_module, app_state, event_type, pane, item_id):
    widget = Mock()
    app_state.query_one.return_value = widget
    await process(app_module, app_state, [{"type": event_type, "item_id": item_id, "text": "Hello"}])
    assert app_state.query_one.call_args.args[0] == pane
    assert widget.method_calls == [call.clear()] + (
        [call.write(item_id)] if item_id else []
    ) + [call.write("Hello")]


async def test_bad_event_still_acknowledged_to_avoid_queue_join_deadlock(app_module, app_state):
    with pytest.raises(KeyError):
        await process(app_module, app_state, [{}])
    await asyncio.wait_for(app_state.event_queue.join(), timeout=1)


async def test_recording_toggle_updates_button_status_and_manual_response(app_module, app_state):
    button, indicator = NS(label="Record"), NS(is_recording=False)
    app_state.query_one.side_effect = lambda selector, *args: button if selector == "#send-button" else indicator
    await app_module.RealtimeApp.toggle_recording(app_state)
    assert app_state.should_send_audio.is_set()
    assert indicator.is_recording is True
    assert button.label == "Stop"
    app_state.ai_service.request_response_if_manual_mode.assert_not_awaited()
    await app_module.RealtimeApp.toggle_recording(app_state)
    assert not app_state.should_send_audio.is_set()
    assert indicator.is_recording is False
    assert button.label == "Record"
    app_state.ai_service.request_response_if_manual_mode.assert_awaited_once()


@pytest.mark.parametrize("settings", [None, {"prompt": "Updated"}])
async def test_config_save_restarts_session_but_cancel_does_nothing(app_module, app_state, settings):
    await app_module.RealtimeApp.apply_config(app_state, settings)
    if settings:
        assert app_state.user_config == settings
        app_state.config_manager.save.assert_called_once_with(settings)
        app_state.restart_workers.assert_awaited_once()
    else:
        app_state.config_manager.save.assert_not_called()
        app_state.restart_workers.assert_not_awaited()


async def test_provider_failure_cleans_resources_and_signals_disconnect(app_module, app_state, monkeypatch):
    service = NS(handle_realtime_connection=AsyncMock(side_effect=RuntimeError("Disconnected")),
                 cleanup_resources=AsyncMock())
    monkeypatch.setattr(app_module, "get_ai_service", Mock(return_value=service))
    app_state.session_updated.set()
    with pytest.raises(RuntimeError, match="Disconnected"):
        await app_module.RealtimeApp.handle_realtime_connection(app_state)
    assert app_state.connection_cancelled.is_set()
    assert not app_state.session_updated.is_set()
    service.cleanup_resources.assert_awaited_once()
