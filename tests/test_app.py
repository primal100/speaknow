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
        send_mic_audio_worker=None, show_connection_error=Mock(),
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
    app_state.session_updated.set()
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
    await app_module.RealtimeApp.handle_realtime_connection(app_state)
    app_state.show_connection_error.assert_called_once()
    assert app_state.connection_cancelled.is_set()
    assert not app_state.session_updated.is_set()
    service.cleanup_resources.assert_awaited_once()


async def test_constructor_failure_signals_disconnect_and_allows_retry(app_module, app_state, monkeypatch):
    monkeypatch.setattr(app_module, "get_ai_service", Mock(side_effect=ValueError("Bad settings")))
    await app_module.RealtimeApp.handle_realtime_connection(app_state)
    assert app_state.connection_cancelled.is_set()
    assert app_state.ai_service is None
    app_state.show_connection_error.assert_called_once()


async def test_cleanup_failure_does_not_hide_original_error_or_block_retry(app_module, app_state, monkeypatch):
    service = NS(handle_realtime_connection=AsyncMock(side_effect=RuntimeError("Connection failed")),
                 cleanup_resources=AsyncMock(side_effect=RuntimeError("Cleanup failed")))
    monkeypatch.setattr(app_module, "get_ai_service", Mock(return_value=service))
    await app_module.RealtimeApp.handle_realtime_connection(app_state)
    assert str(app_state.show_connection_error.call_args.args[0]) == "Connection failed"
    assert app_state.connection_cancelled.is_set()


async def test_cancelled_connection_does_not_show_error(app_module, app_state, monkeypatch):
    service = NS(handle_realtime_connection=AsyncMock(side_effect=asyncio.CancelledError()),
                 cleanup_resources=AsyncMock())
    monkeypatch.setattr(app_module, "get_ai_service", Mock(return_value=service))
    with pytest.raises(asyncio.CancelledError):
        await app_module.RealtimeApp.handle_realtime_connection(app_state)
    app_state.show_connection_error.assert_not_called()
    assert app_state.connection_cancelled.is_set()
    service.cleanup_resources.assert_awaited_once()


async def test_session_timeout_shows_error_instead_of_exiting(app_module, app_state):
    app_state._send_mic_audio = AsyncMock(side_effect=TimeoutError())
    app_state.handle_realtime_connection_worker = Mock(is_finished=False)
    await app_module.RealtimeApp.send_mic_audio(app_state)
    app_state.show_connection_error.assert_called_once()
    app_state.handle_realtime_connection_worker.cancel.assert_called_once()


@pytest.mark.parametrize("missing_key", [False, True])
async def test_launch_error_dialog_keeps_app_open_and_can_retry(app_module, monkeypatch, missing_key):
    monkeypatch.setattr(app_module, "AudioPlayerAsync", Mock())
    monkeypatch.setattr(app_module.RealtimeApp, "_send_mic_audio", AsyncMock())
    for name in ("realtime_app", "realtime_tokens"):
        monkeypatch.setattr(app_module.logging.getLogger(name), "handlers", [])
    if missing_key:
        monkeypatch.delenv("OPENAI_API_KEY")

    async def connected(queue):
        queue.put_nowait({"type": "session_updated"})
        await asyncio.Event().wait()

    service = NS(handle_realtime_connection=AsyncMock(side_effect=connected), cleanup_resources=AsyncMock())
    rejected = RuntimeError("Authentication rejected")
    rejected.status_code = 401
    failed_service = NS(handle_realtime_connection=AsyncMock(side_effect=rejected), cleanup_resources=AsyncMock())
    factory = Mock(side_effect=[failed_service, service] if not missing_key else [service])
    monkeypatch.setattr(app_module, "get_ai_service", factory)
    app = app_module.RealtimeApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        assert isinstance(app.screen, app_module.ConnectionErrorModal)
        assert app.connection_cancelled.is_set()
        if missing_key:
            assert "OPENAI_API_KEY" in app.screen.message
            factory.assert_not_called()
            monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        else:
            assert "Authentication was rejected" in app.screen.message
        await pilot.click("#error-retry")
        await pilot.pause()
        assert not isinstance(app.screen, app_module.ConnectionErrorModal)
        assert app.session_updated.is_set()


@pytest.mark.parametrize("status", [401, 403, 429])
def test_error_message_explains_status_without_leaking_credentials(config, status):
    from speaknow_ai_realtime_text_to_speech.connection_errors import connection_error_message

    error = RuntimeError("secret-api-key must never be displayed")
    error.status_code = status
    message = connection_error_message(error, config)
    assert "secret-api-key" not in message
    assert ("Authentication" if status != 429 else "quota") in message
