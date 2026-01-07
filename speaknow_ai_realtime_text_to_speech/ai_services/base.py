from abc import ABC, abstractmethod
import asyncio
from datetime import datetime
from numpy import ndarray
from gpt_token_tracker.models import Timed
from gpt_token_tracker.writers.types import Writer
from gpt_token_tracker.writers.log_writer import LogWriter
from gpt_token_tracker.token_logger import TokenLogger
from gpt_token_tracker.writers.csv_writer import CSVWriter
from pathlib import Path
from typing import Any, Type, TYPE_CHECKING
from ..directories import get_token_dir
from .prices import realtime_costs, transcription_costs

# Avoid circular import
if TYPE_CHECKING:
    from ..widgets import SessionDisplay
else:
    SessionDisplay = object


TOKENS_DIR = get_token_dir()
log_writer = LogWriter("realtime_tokens")


class BaseAIService(ABC):
    prefix: str = ""
    client: Any
    realtime_pricing_cls: Type
    transcription_pricing_cls: Type
    log_token_usage: bool = True
    log_per_interval_pricing: bool = False

    @classmethod
    def set_default_config_options_on_change(cls) -> dict[str, Any]:
        return {}

    def __init__(self, user_config: dict[str, Any]):
        self.user_config = user_config
        if self.user_config.get("save_token_data") and self.user_config["model"] in realtime_costs:
            self.realtime_convo_csv = Path(TOKENS_DIR) / f"{self.prefix}_realtime_conversation_tokens.csv"
            self.token_logger_realtime = TokenLogger(log_writer, self.realtime_pricing_cls(realtime_costs))
            self.csv_writer_realtime = CSVWriter(self.realtime_convo_csv)
            self.csv_token_logger_realtime = TokenLogger(self.csv_writer_realtime, self.realtime_pricing_cls(realtime_costs))
        else:
            self.realtime_convo_csv = None
            self.realtime_tokens_csv = None
            self.token_logger_realtime = None
            self.csv_writer_realtime = None
            self.csv_token_logger_realtime = None
        if self.user_config.get("save_token_data") and self.transcription_pricing_cls and self.user_config['transcription_model'] in transcription_costs:
            self.realtime_transcribe_tokens_csv = Path(TOKENS_DIR) / f"{self.prefix}_realtime_transcribe_tokens.csv"
            self.csv_writer_realtime_transcribe = CSVWriter(self.realtime_transcribe_tokens_csv)
            self.token_logger_realtime_transcription = TokenLogger(log_writer, self.transcription_pricing_cls(transcription_costs))
            self.csv_token_logger_realtime_transcription = TokenLogger(self.csv_writer_realtime_transcribe, self.transcription_pricing_cls(transcription_costs))
        else:
            self.token_logger_realtime_transcription = None
            self.csv_token_logger_realtime_transcription = None
            self.token_logger_realtime_transcription = None
            self.csv_writer_realtime_transcribe = None
            self.csv_token_logger_realtime_transcription = None

    @staticmethod
    async def _close_logger_if_present(logger: Writer | None):
        if logger:
            await asyncio.to_thread(logger.close)

    async def cleanup_resources(self):
        await asyncio.gather(
            self._close_logger_if_present(self.token_logger_realtime),
            self._close_logger_if_present(self.csv_token_logger_realtime),
            self._close_logger_if_present(self.token_logger_realtime_transcription),
            self._close_logger_if_present(self.csv_writer_realtime_transcribe),
        )

    async def write_connection_time(self, model: str, start_time: datetime):
        if self.log_per_interval_pricing and self.token_logger_realtime and self.csv_token_logger_realtime:
            end_time = datetime.now()
            timed = Timed(start_time=start_time, end_time=end_time)
            self.token_logger_realtime.record_timed(model, timed)
            await asyncio.to_thread(self.csv_token_logger_realtime.record_timed, model, timed)

    async def write_realtime_tokens(self, model: str, result: str, usage: Any) -> None:
        self.token_logger_realtime.record(model, result, usage)
        await asyncio.to_thread(self.csv_token_logger_realtime.record, model, result, usage)

    async def write_realtime_transcribe_tokens(self, model: str, result: str, usage: Any) -> None:
        self.token_logger_realtime_transcription.record(model, result, usage)
        await asyncio.to_thread(self.csv_token_logger_realtime_transcription.record, model, result, usage)

    async def write_realtime_tokens_wrapper(self, model: str, result: str, usage: Any) -> None:
        if not self.user_config['save_result']:
            result = ""
        if self.log_token_usage and self.token_logger_realtime and self.csv_token_logger_realtime:
            await self.write_realtime_tokens(model, result, usage)

    async def write_realtime_transcribe_tokens_wrapper(self, model: str, result: str, usage: Any) -> None:
        if not self.user_config['save_result']:
            result = ""
        if self.log_token_usage and self.token_logger_realtime_transcription and self.csv_token_logger_realtime_transcription:
            await self.write_realtime_transcribe_tokens(model, result, usage)

    @abstractmethod
    async def send_audio(self, data: ndarray, sent_audio: bool) -> bool: ...

    @abstractmethod
    async def handle_realtime_connection(self, event_queue: asyncio.Queue[dict[str, Any]]) -> None: ...

    @abstractmethod
    async def request_response(self) -> None: ...

    @abstractmethod
    async def request_response_if_manual_mode(self) -> None: ...
