# Originally based on openai-python.examples.realtime.push_to_talk.py

import base64
import asyncio
import logging.config
import shutil
from typing import Any, cast, override
from textual import events
import sounddevice as sd

from speaknow_ai_realtime_text_to_speech.audio_util import CHANNELS, SAMPLE_RATE, AudioPlayerAsync
import audioop
import time
from pathlib import Path
from datetime import datetime

from speaknow_ai_realtime_text_to_speech.app_css import CSS
from speaknow_ai_realtime_text_to_speech.directories import HOME, get_log_config_file, get_log_dir, get_recordings_dir, get_token_dir
from speaknow_ai_realtime_text_to_speech.widgets import (AmplitudeGraph, SessionDisplay, AudioStatusIndicator,
                                                         TextualLogMessage, TextualPaneLogHandler, ConfigModal)
from speaknow_ai_realtime_text_to_speech.config import ConfigManager
from speaknow_ai_realtime_text_to_speech.utils import update_log_config, save_wav_chunk

from textual.app import App, ComposeResult
from textual import on
from textual.events import Unmount
from textual.logging import TextualHandler
from textual.widgets import Button, RichLog
from textual.worker import Worker
from textual.containers import Container, Horizontal

from openai import AsyncOpenAI
from openai.types.realtime.session_update_event_param import Session  # https://github.com/openai/openai-python/pull/2803
from openai.resources.realtime.realtime import AsyncRealtimeConnection # Another bug?
from openai.types.realtime.realtime_audio_input_turn_detection_param import ServerVad, SemanticVad
from openai.types.realtime.realtime_response_usage import RealtimeResponseUsage
from openai.types.realtime.conversation_item_input_audio_transcription_completed_event import UsageTranscriptTextUsageTokens


from gpt_token_tracker.token_logger import TokenLogger
from gpt_token_tracker.writers.log_writer import LogWriter
from gpt_token_tracker.pricing import PricingRealtime, PricingAudioTranscription
from gpt_token_tracker.writers.csv_writer import CSVWriter


LOG_CONFIG_FILE = get_log_config_file()
BASE_LOG_DIR = get_log_dir()
AUDIO_DIR = get_recordings_dir()
TOKENS_DIR = get_token_dir()


REALTIME_COSTS = {
        "text_in": 0.60,
        "cached_text_in": 0.06,
        "text_out": 2.40,
        "audio_in": 10.00,
        "audio_out": 20.00,
        "image_in": 5.00,
        "cached_image_in": 0.50,
        "cached_audio_in": 0.40,
    }


REALTIME_CONVO_CSV = Path(TOKENS_DIR) / "realtime_conversation_tokens.csv"
REALTIME_TOKENS_CSV = Path(TOKENS_DIR) / "realtime_transcribe_tokens.csv"

if not LOG_CONFIG_FILE.exists():
    PACKAGE_DIR = Path(__file__).resolve().parent
    PACKAGE_LOG_CONFIG_DIR = PACKAGE_DIR / "logging.conf"
    shutil.copy(PACKAGE_LOG_CONFIG_DIR, LOG_CONFIG_FILE)

update_log_config(LOG_CONFIG_FILE, BASE_LOG_DIR)
# Load logging config (.ini)
logging.config.fileConfig(LOG_CONFIG_FILE, disable_existing_loggers=False)
log = logging.getLogger("realtime_app")
events_log = logging.getLogger("events")


log_writer = LogWriter("realtime_tokens")
token_logger_realtime = TokenLogger(log_writer, PricingRealtime(REALTIME_COSTS))
token_logger_realtime_transcription = TokenLogger(log_writer, PricingAudioTranscription(REALTIME_COSTS))
csv_writer_realtime = CSVWriter(REALTIME_CONVO_CSV)
csv_writer_realtime_transcribe = CSVWriter(REALTIME_TOKENS_CSV)
csv_token_logger_realtime = TokenLogger(csv_writer_realtime, PricingRealtime(REALTIME_COSTS))
csv_token_logger_realtime_transcription = TokenLogger(csv_writer_realtime_transcribe, PricingAudioTranscription(REALTIME_COSTS))


log.info('Using application directory: %s', HOME)


def write_realtime_tokens(model: str, result: str, usage: RealtimeResponseUsage) -> None:
    token_logger_realtime.record(model, result, usage)
    csv_token_logger_realtime.record(model, result, usage)


def write_realtime_transcribe_tokens(model: str, result: str, usage: UsageTranscriptTextUsageTokens) -> None:
    token_logger_realtime_transcription.record(model, result, usage)
    csv_token_logger_realtime_transcription.record(model, result, usage)


class RealtimeApp(App[None]):
    TITLE = "SpeakNow"
    SUB_TITLE = "Realtime AI Voice Interface"
    CSS = CSS
    client: AsyncOpenAI
    should_send_audio: asyncio.Event
    audio_player: AudioPlayerAsync
    last_audio_item_id: str | None
    connection: AsyncRealtimeConnection | None
    session: Session | None
    connected: asyncio.Event
    handle_realtime_connection_worker: Worker | None = None
    send_mic_audio_worker: Worker | None = None

    def __init__(self) -> None:
        super().__init__()
        self.config_manager = ConfigManager()
        self.user_config = self.config_manager.load()
        self.connection = None
        self.session = None
        self.client = AsyncOpenAI()
        self.audio_player = AudioPlayerAsync()

        self.session_updated = asyncio.Event()
        self.response_in_progress = asyncio.Event()
        self.speech_ongoing = asyncio.Event()
        self.speech_done = asyncio.Event()
        self.connection_cancelled = asyncio.Event()
        self.connection_cancelled.set()
        self.last_audio_item_id = None
        self.should_send_audio = asyncio.Event()
        self.connected = asyncio.Event()
        self.start_time = time.time()

    @on(Unmount)
    async def cleanup_resources(self):
        self.log("The app is unmounting now!")
        await asyncio.gather(
            asyncio.to_thread(token_logger_realtime.close),
            asyncio.to_thread(token_logger_realtime_transcription.close),
            asyncio.to_thread(csv_token_logger_realtime.close),
            asyncio.to_thread(csv_writer_realtime_transcribe.close)
        )

    @override
    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        with Container():
            with Horizontal(id="session-row"):
                yield SessionDisplay(id="session-display")
                yield AmplitudeGraph(id="amp-graph")
            with Horizontal(id="status-row"):
                yield AudioStatusIndicator(id="status-indicator")
                yield Button("Record", id="send-button")
                yield Button("Config", id="config-button")
                yield Button("Quit", id="quit-button")
            yield RichLog(id="middle-pane", wrap=True, highlight=True, markup=True)
            yield RichLog(id="lower-middle-pane", wrap=True, highlight=True, markup=True)
            yield RichLog(id="bottom-pane", wrap=True, highlight=True, markup=True)

    def worker_callback(self, worker) -> None:
        if exc := worker.get_exception():
            log.error("Worker %s failed with exception: %s", worker, exc)

    async def on_mount(self) -> None:
        # Attach log handler

        handler = TextualPaneLogHandler(self)
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s]: %(message)s"))

        logging.getLogger("realtime_app").addHandler(handler)
        logging.getLogger("realtime_tokens").addHandler(handler)
        textual_handler = TextualHandler()
        textual_handler.setLevel(logging.DEBUG)
        textual_handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s]: %(message)s"))
        logging.getLogger("realtime_app").addHandler(textual_handler)

        await self.restart_workers()

    async def restart_workers(self):
        log.debug("Cancelling mic audio worker")
        if self.send_mic_audio_worker and not self.send_mic_audio_worker.is_finished:
            self.send_mic_audio_worker.cancel()

        log.debug("Cancelling realtime connection worker")
        if self.handle_realtime_connection_worker and not self.handle_realtime_connection_worker.is_finished:
            self.handle_realtime_connection_worker.cancel()

        await self.connection_cancelled.wait()

        self.handle_realtime_connection_worker = self.run_worker(self.handle_realtime_connection())
        self.send_mic_audio_worker = self.run_worker(self.send_mic_audio())

        async def listen_immediately():
            await asyncio.sleep(1)
            # Simulate pressing lowercase "k"
            await self.toggle_recording()

        if self.user_config['immediate_initialisation']:
            self.run_worker(listen_immediately())

    async def on_textual_log_message(self, message: TextualLogMessage) -> None:
        """Receive log messages sent by the log handler and write them to the pane."""
        pane = self.query_one("#bottom-pane", RichLog)
        pane.write(message.text)

    async def handle_realtime_connection(self) -> None:
        self.connection_cancelled.clear()
        try:
            async with self.client.realtime.connect(model=self.user_config['model']) as conn:
                self.connection = conn
                self.connected.set()

                # note: this is the default and can be omitted
                # if you want to manually handle VAD yourself, then set `'turn_detection': None`

                turn_detection: ServerVad | SemanticVad | None = None
                mode = self.user_config['mode']

                if mode == "server":
                    turn_detection: ServerVad = {
                        "type": "server_vad",
                        "idle_timeout_ms": 5000
                    }
                elif mode == "semantic":
                    turn_detection: SemanticVad = {
                        "type": "semantic_vad",
                        "eagerness": "high"
                    }

                if self.user_config['transcription_enabled']:
                    transcription_info = {
                        "language": self.user_config['language'],
                        "model": self.user_config['transcription_model'],
                    }
                else:
                    transcription_info = None

                log.info("Updating session with model %s, prompt %s",
                         self.user_config['model'], self.user_config['prompt'])
                log.info("Turn Detection: %s", turn_detection)
                log.info("Transcription Info: %s", transcription_info)

                await conn.session.update(
                    session={
                        "audio": {
                            "input": {"turn_detection": turn_detection,
                                      "transcription": transcription_info
                                      },
                        },
                        "instructions": self.user_config['prompt'],
                        "output_modalities": self.user_config.get('output_modalities'),
                        "model": self.user_config['model'],
                        "type": "realtime",
                    }
                )

                acc_items: dict[str, Any] = {}
                transcription_items: dict[str, Any] = {}
                speech_start_times: dict[str, datetime] = {}

                async for event in conn:
                    events_log.info("Event Type: %s. Item Id: %s", event.type, getattr(event, "item_id", ""))
                    events_log.debug(event)
                    if event.type == "session.created":
                        self.session = event.session
                        session_display = self.query_one(SessionDisplay)
                        assert event.session.id is not None
                        session_display.session_id = event.session.id
                        continue

                    if event.type == "session.updated":
                        self.session_updated.set()
                        self.session = event.session
                        continue

                    if event.type == "response.output_audio.delta":
                        if event.item_id != self.last_audio_item_id:
                            log.info("First audio response received for %s", event.item_id)
                            self.audio_player.reset_frame_count()
                            self.last_audio_item_id = event.item_id

                        bytes_data = base64.b64decode(event.delta)
                        if self.user_config['play_audio']:
                            self.audio_player.add_data(bytes_data)
                        continue

                    if event.type == "response.output_audio_transcript.delta" or event.type == "response.output_text.delta":
                        try:
                            text = acc_items[event.item_id]
                        except KeyError:
                            parts = event.type.split(".")
                            category = parts[1].replace("_", " ") if len(parts) > 1 else "unknown"
                            log.info("First %s response received for %s", category, event.item_id)
                            acc_items[event.item_id] = event.delta
                        else:
                            acc_items[event.item_id] = text + event.delta

                        # Clear and update the entire content because RichLog otherwise treats each delta as a new line
                        lower_middle_pane = self.query_one("#lower-middle-pane", RichLog)
                        lower_middle_pane.clear()
                        lower_middle_pane.write(event.item_id)
                        lower_middle_pane.write(acc_items[event.item_id])
                        continue

                    if event.type == "response.output_audio_transcript.complete" or event.type == "response.output_text.complete" or event.type == "response.output_item.complete":
                        parts = event.type.split(".")
                        category = parts[1].replace("_", " ") if len(parts) > 1 else "unknown"
                        log.debug("%s done for %s", category, event.item_id)
                        final_text = event.item.text
                        try:
                            transcription_items[event.item_id]
                        except KeyError:
                            transcription_items[event.item_id] = final_text
                        log.info("Answer: %s", final_text)

                    if event.type == "conversation.item.input_audio_transcription.delta":
                        try:
                            text = transcription_items[event.item_id]
                        except KeyError:
                            log.info("First realtime audio transcription response received for %s", event.item_id)
                            transcription_items[event.item_id] = event.delta
                        else:
                            transcription_items[event.item_id] = text + event.delta

                        # Clear and update the entire content because RichLog otherwise treats each delta as a new line
                        middle_pane = self.query_one("#middle-pane", RichLog)
                        middle_pane.clear()
                        middle_pane.write(event.item_id)
                        middle_pane.write(transcription_items[event.item_id])
                        continue

                    if event.type == "conversation.item.input_audio_transcription.completed":
                        log.debug("Audio realtime transcription response done for %s", event.item_id)
                        log.debug("Type: %s", type(event))
                        try:
                            text = transcription_items[event.item_id]
                        except KeyError:
                            transcription_items[event.item_id] = event.transcript
                            text = transcription_items[event.item_id]
                        log.info("[TRANSCRIPTION] REALTIME: %s", text)

                        if usage := getattr(event, "usage"):
                            log.debug("Type: %s", type(usage))
                            await asyncio.to_thread(write_realtime_transcribe_tokens, self.user_config['model'], text, usage)
                        else:
                            log.warning("No token usage info in transcription response.")
                            continue

                        continue

                    if event.type == "response.created":
                        self.response_in_progress.set()
                        log.info("%s Response is being created", event.response.id)
                        continue

                    if event.type == "input_audio_buffer.speech_started":
                        self.speech_ongoing.set()
                        self.speech_done.clear()
                        speech_start_times[event.item_id] = datetime.now()
                        log.info("%s Speech started", event.item_id)
                        continue

                    if event.type == "input_audio_buffer.speech_stopped":
                        self.speech_done.set()
                        self.speech_ongoing.clear()
                        end = datetime.now()
                        start = speech_start_times.get(event.item_id)
                        if start:
                            duration = (end - start).total_seconds()
                            log.info("Speech ended for %s, %.3f seconds detected", event.item_id, duration)
                        else:
                            log.warning("Speech ended event for %s with no matching start time", event.item_id)
                        continue

                    if event.type == "response.done":
                        self.response_in_progress.clear()
                        status = event.response.status
                        status_details = event.response.status_details
                        result = None
                        if output := event.response.output:
                            item = output[0]
                            if content := getattr(item, "content", None):
                                result = getattr(content[0], "text", None)
                        if status_details:
                            log.info("%s Response is done, status: %s, type: %s, reason: %s, error: %s, result: %s",
                                     event.response.id, status, status_details.type,
                                     status_details.reason, status_details.error, result)
                        else:
                            log.info("%s Response is done, status: %s, result: %s", event.response.id, status, result)
                        if usage := getattr(event.response, "usage"):
                            await asyncio.to_thread(write_realtime_tokens,
                                                    self.user_config['model'],
                                                    result,
                                                    usage
                                                    )
                        else:
                            log.warning("No token usage info in response.")
                            continue

                        if event.type == "rate_limits.updated":
                            for rl in event.rate_limits:
                                log.debug(
                                    "[RATE LIMIT] name=%s | limit=%s | remaining=%s | reset_in=%.3fs",
                                    rl.name,
                                    rl.limit,
                                    rl.remaining,
                                    rl.reset_seconds,
                                )

                        continue
        finally:
            log.debug('Clearing events')
            self.session_updated.clear()
            self.connected.clear()
            self.connection = None
            self.connection_cancelled.set()

    async def _get_connection(self) -> AsyncRealtimeConnection:
        await self.connected.wait()
        assert self.connection is not None
        return self.connection

    async def send_mic_audio(self) -> None:
        log.info("Starting mic audio task")
        recordings_dir = get_recordings_dir()
        try:
            await asyncio.wait_for(self.session_updated.wait(), timeout=10)
        except asyncio.TimeoutError:
            log.error("Failed to update session, existing")
            self.exit()

        async with asyncio.TaskGroup() as tg:
            amp_widget = self.query_one("#amp-graph", AmplitudeGraph)

            sent_audio = False

            device_info = sd.query_devices()
            print(device_info)

            read_size = int(SAMPLE_RATE * 0.02)

            stream = sd.InputStream(
                channels=CHANNELS,
                samplerate=SAMPLE_RATE,
                dtype="int16",
            )
            stream.start()

            status_indicator = self.query_one(AudioStatusIndicator)

            wav_stream = b''

            try:

                while True:

                    if stream.read_available < read_size:
                        await asyncio.sleep(0)
                        continue

                    await self.should_send_audio.wait()
                    status_indicator.is_recording = True

                    data, _ = stream.read(read_size)

                    connection = await self._get_connection()
                    if not sent_audio:
                        if self.response_in_progress.is_set():
                            log.info("Sending initial cancel response...")
                            await connection.send({"type": "response.cancel"})
                        sent_audio = True

                    await connection.input_audio_buffer.append(audio=base64.b64encode(cast(Any, data)).decode("utf-8"))

                    rms = audioop.rms(data, 2)  # 2 bytes = 16-bit
                    peak = min(rms / 30000.0, 1.0)  # normalize to 0–1 range

                    amp_widget.amplitude = peak

                    binary_data = data.tobytes()

                    wav_stream += binary_data

                    if not self.speech_ongoing.is_set():
                        if self.speech_done.is_set():
                            log.info("Saving speech done chunk")
                            tg.create_task(save_wav_chunk(wav_stream, "speech_done", CHANNELS, SAMPLE_RATE, recordings_dir))
                            wav_stream = b''
                            self.speech_done.clear()
                        elif self.user_config["save_silence_multiplier"] and len(wav_stream) > (
                                read_size * self.user_config["save_silence_multiplier"]):
                            log.debug("Saving silence chunk")
                            tg.create_task(save_wav_chunk(wav_stream, "periodic", CHANNELS, SAMPLE_RATE, recordings_dir))
                            wav_stream = b''
                    elif self.user_config["save_speech_multiplier"] and len(wav_stream) > (
                            read_size * self.user_config["save_speech_multiplier"]):
                        log.debug("Saving speech chunk")
                        tg.create_task(save_wav_chunk(wav_stream, "speech", CHANNELS, SAMPLE_RATE, recordings_dir))
                        wav_stream = b''

                    await asyncio.sleep(0)
            except KeyboardInterrupt:
                pass
            finally:
                log.debug("Stopping mic stream in finally...")
                if wav_stream:
                    log.debug("Saving final chunk")
                    await save_wav_chunk(wav_stream, "periodic", CHANNELS, SAMPLE_RATE, recordings_dir)
                stream.stop()
                stream.close()
                log.debug(tg._tasks)

    def show_config(self) -> None:
        self.push_screen(ConfigModal(), self.apply_config)

    async def apply_config(self, new_config: dict | None) -> None:
        if new_config:
            self.user_config = new_config
            self.config_manager.save(new_config)
            log.info("Settings Updated! Restarting session.")
            log.info(self.user_config)
            self.notify("Settings Updated! Restarting session...")

            # Logic: Refresh the OpenAI connection with new prompt/model
            # You might need to trigger a session.update event here
            await self.refresh_session()

    async def refresh_session(self):
        """Helper to push new config to the active OpenAI connection."""
        await self.restart_workers()

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "send-button":
            log.info("Button pressed, toggle recording...")
            await self.toggle_recording()
            return
        if event.button.id == "quit-button":
            log.info("Button pressed, quitting...")
            self.exit()
        if event.button.id == "config-button":
            self.show_config()

    async def toggle_recording(self) -> None:
        send_button = self.query_one("#send-button", Button)
        status_indicator = self.query_one(AudioStatusIndicator)
        if status_indicator.is_recording:
            log.debug("Toggle recording off...")
            self.should_send_audio.clear()
            send_button.label = "Record"
            status_indicator.is_recording = False

            if self.session and self.session.audio.input.turn_detection is None:  # Bugfix
                # The default in the API is that the model will automatically detect when the user has
                # stopped talking and then start responding itself.
                #
                # However if we're in manual `turn_detection` mode then we need to
                # manually tell the model to commit the audio buffer and start responding.
                conn = await self._get_connection()
                log.info("Requesting response")
                await conn.input_audio_buffer.commit()
                await conn.response.create()
        else:
            log.debug("Toggle recording on...")
            send_button.label = "Stop"
            self.should_send_audio.set()
            status_indicator.is_recording = True

    async def on_key(self, event: events.Key) -> None:
        """Handle key press events."""
        log.debug("Key event: %s", event.key)
        if event.key == "enter":
            log.info("Enter pressed, toggle recording...")
            await self.toggle_recording()
            return

        if event.key == "q":
            log.info("Q pressed, quitting...")
            self.exit()
            return

        if event.key == "o":
            # To log for timing purposes
            log.info("Question starting...")
            return

        if event.key == "l":
            # To log for timing purposes
            log.info("Question sent...")
            return

        if event.key == "c":
            self.show_config()
            return

        if event.key == "s":
            # To log for timing purposes
            conn = await self._get_connection()
            log.info("Requesting response manually")
            await conn.input_audio_buffer.commit()
            await conn.response.create()
            return

        if event.key == "k":
            log.info("k pressed, toggle recording...")
            await self.toggle_recording()
            return


def run():
    app = RealtimeApp()
    app.run()

if __name__ == "__main__":
    run()
