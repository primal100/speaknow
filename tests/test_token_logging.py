from datetime import datetime
from unittest.mock import Mock

import pytest


@pytest.mark.parametrize("kind", ["realtime", "realtime_transcription"])
@pytest.mark.parametrize("save_result", [False, True])
async def test_token_logging_respects_result_privacy(openai_service, kind, save_result):
    service = openai_service
    service.user_config["save_result"] = save_result
    log = Mock()
    csv = Mock()
    setattr(service, f"token_logger_{kind}", log)
    setattr(service, f"csv_token_logger_{kind}", csv)
    usage = object()
    method = service.write_realtime_tokens_wrapper if kind == "realtime" else service.write_realtime_transcribe_tokens_wrapper
    await method("model", "private answer", usage)
    expected = "private answer" if save_result else ""
    log.record.assert_called_once_with("model", expected, usage)
    csv.record.assert_called_once_with("model", expected, usage)


@pytest.mark.parametrize("kind", ["realtime", "realtime_transcription"])
@pytest.mark.parametrize("disabled", ["policy", "log", "csv"])
async def test_token_logging_skips_disabled_or_missing_writers(openai_service, kind, disabled):
    service = openai_service
    log, csv = Mock(), Mock()
    service.log_token_usage = disabled != "policy"
    setattr(service, f"token_logger_{kind}", None if disabled == "log" else log)
    setattr(service, f"csv_token_logger_{kind}", None if disabled == "csv" else csv)
    method = service.write_realtime_tokens_wrapper if kind == "realtime" else service.write_realtime_transcribe_tokens_wrapper
    await method("model", "answer", object())
    log.record.assert_not_called()
    csv.record.assert_not_called()


async def test_disabled_token_storage_does_not_create_csv_files(openai_service, tmp_path):
    assert openai_service.token_logger_realtime is None
    assert openai_service.token_logger_realtime_transcription is None
    await openai_service.cleanup_resources()
    assert list(tmp_path.rglob("*.csv")) == []


async def test_cleanup_closes_present_loggers(openai_service):
    loggers = [Mock() for _ in range(4)]
    (openai_service.token_logger_realtime,
     openai_service.csv_token_logger_realtime,
     openai_service.token_logger_realtime_transcription,
     openai_service.csv_writer_realtime_transcribe) = loggers
    await openai_service.cleanup_resources()
    for logger in loggers:
        logger.close.assert_called_once()


@pytest.mark.parametrize("timed_pricing", [False, True])
async def test_connection_duration_only_recorded_for_timed_pricing(openai_service, timed_pricing):
    service = openai_service
    service.log_per_interval_pricing = timed_pricing
    service.token_logger_realtime = Mock()
    service.csv_token_logger_realtime = Mock()
    started = datetime(2020, 1, 1)
    await service.write_connection_time("model", started)
    if timed_pricing:
        model, interval = service.token_logger_realtime.record_timed.call_args.args
        assert model == "model"
        assert interval.start_time == started
        assert interval.end_time >= started
        service.csv_token_logger_realtime.record_timed.assert_called_once_with(model, interval)
    else:
        service.token_logger_realtime.record_timed.assert_not_called()
        service.csv_token_logger_realtime.record_timed.assert_not_called()
