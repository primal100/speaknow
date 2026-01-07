# Originally based on openai-python.examples.realtime.push_to_talk.py

import asyncio
import logging.config
import shutil
from datetime import datetime
from typing import Any
from typing_extensions import override
from textual import events

try:
    import sounddevice as sd
except OSError:
    print("Install postaudio19-dev and ffmpeg")
    print("For example in Debian or Ubuntu:")
    print("sudo apt install portaudio19-dev ffmpeg")
    raise

from speaknow_ai_realtime_text_to_speech.audio_util import CHANNELS, SAMPLE_RATE, AudioPlayerAsync
import audioop
import time

from speaknow_ai_realtime_text_to_speech.app_css import CSS
from speaknow_ai_realtime_text_to_speech import version
from speaknow_ai_realtime_text_to_speech.directories import (APP_NAME, HOME, get_log_config_file,
                                                             get_default_log_config_file, get_log_dir,
                                                             get_token_dir)
from speaknow_ai_realtime_text_to_speech.widgets import (AmplitudeGraph, SessionDisplay, AudioStatusIndicator,
                                                         TextualLogMessage, TextualPaneLogHandler, ConfigModal)
from speaknow_ai_realtime_text_to_speech.config import ConfigManager
from speaknow_ai_realtime_text_to_speech.utils import update_log_config

from textual.app import App, ComposeResult
from textual.logging import TextualHandler
from textual.widgets import Button, RichLog, Static
from textual.worker import Worker
from textual.containers import Container, Horizontal

from speaknow_ai_realtime_text_to_speech.ai_services import get_ai_service, BaseAIService

# Ignore pydub warning in python 3.14
# site-packages\pydub\utils.py:300: SyntaxWarning: "\(" is an invalid escape sequence. Such sequences will not work in the future. Did you mean "\\("? A raw string is also an option.
import warnings

warnings.filterwarnings(
    "ignore",
    category=SyntaxWarning,
    module=r"pydub\..*",
)

LOG_CONFIG_FILE = get_log_config_file()
BASE_LOG_DIR = get_log_dir()
TOKENS_DIR = get_token_dir()

if not LOG_CONFIG_FILE.exists():
    PACKAGE_LOG_CONFIG_PATH = get_default_log_config_file()
    shutil.copy(PACKAGE_LOG_CONFIG_PATH, LOG_CONFIG_FILE)

update_log_config(LOG_CONFIG_FILE, BASE_LOG_DIR)
# Load logging config (.ini)
logging.config.fileConfig(LOG_CONFIG_FILE, disable_existing_loggers=False)
log = logging.getLogger("realtime_app")

log.info('Using application directory: %s', HOME)


class RealtimeApp(App[None]):
    TITLE = "SpeakNow"
    SUB_TITLE = "Realtime AI Voice Interface"
    CSS = CSS
    ai_service: BaseAIService | None
    should_send_audio: asyncio.Event
    audio_player: AudioPlayerAsync
    connected: asyncio.Event
    event_queue: asyncio.Queue[dict[str, Any]]
    event_queue_worker: Worker | None = None
    handle_realtime_connection_worker: Worker | None = None
    send_mic_audio_worker: Worker | None = None

    def __init__(self) -> None:
        super().__init__()
        self.config_manager = ConfigManager()
        self.user_config = self.config_manager.load()
        self.ai_service = None

        self.audio_player = AudioPlayerAsync()

        self.session_updated = asyncio.Event()
        self.speech_ongoing = asyncio.Event()
        self.speech_done = asyncio.Event()
        self.connection_cancelled = asyncio.Event()
        self.connection_cancelled.set()
        self.should_send_audio = asyncio.Event()
        self.event_queue = asyncio.Queue()
        self.start_time = time.time()

    @override
    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        with Container():
            with Horizontal(id="session-row"):
                yield Static(id="version-display", content=f"{APP_NAME} v{version}")
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
        log.info("Starting %s version %s", APP_NAME, version)
        await self.restart_workers()

    async def restart_workers(self):

        if self.send_mic_audio_worker and not self.send_mic_audio_worker.is_finished:
            log.debug("Cancelling mic audio worker")
            self.send_mic_audio_worker.cancel()

        if self.handle_realtime_connection_worker and not self.handle_realtime_connection_worker.is_finished:
            log.debug("Cancelling realtime connection worker")
            self.handle_realtime_connection_worker.cancel()

        await self.connection_cancelled.wait()

        if self.event_queue_worker and not self.event_queue_worker.is_finished:
            log.debug("Cancelling event queue process worker, will wait for queue to complete first")

            try:
                await asyncio.wait_for(self.event_queue.join(), timeout=5)
            except asyncio.TimeoutError:
                log.error("Not all events in event queue were processed")

            self.event_queue_worker.cancel()

        self.event_queue_worker = self.run_worker(self.process_event_queue())
        self.handle_realtime_connection_worker = self.run_worker(self.handle_realtime_connection())
        self.send_mic_audio_worker = self.run_worker(self.send_mic_audio())

        if self.user_config['immediate_initialisation']:
            await asyncio.sleep(1)
            await self.toggle_recording()

    async def on_textual_log_message(self, message: TextualLogMessage) -> None:
        """Receive log messages sent by the log handler and write them to the pane."""
        pane = self.query_one("#bottom-pane", RichLog)
        pane.write(message.text)

    async def handle_realtime_connection(self) -> None:
        self.connection_cancelled.clear()
        self.ai_service = get_ai_service(self.user_config)
        try:
            await self.ai_service.handle_realtime_connection(self.event_queue)
        finally:
            log.debug('Clearing events')
            self.session_updated.clear()
            self.connection_cancelled.set()
            await self.ai_service.cleanup_resources()

    async def process_event_queue(self) -> None:
        speech_start_times: dict[str, datetime] = {}

        while True:
            event = await self.event_queue.get()
            try:
                if event['type'] == "session_id":
                    session_display = self.query_one(SessionDisplay)
                    session_display.session_id = event['session_id']

                elif event['type'] == "session_updated":
                    self.session_updated.set()

                elif event['type'] == "audio_response":
                    if event["is_first_in_response"]:
                        self.audio_player.reset_frame_count()
                    if self.user_config['play_audio']:
                        self.audio_player.add_data(event["data"])

                elif event["type"] == "response_received":
                    # Clear and update the entire content because RichLog otherwise treats each delta as a new line
                    lower_middle_pane = self.query_one("#lower-middle-pane", RichLog)
                    lower_middle_pane.clear()
                    if event_id := event.get("item_id", None):
                        lower_middle_pane.write(event_id)
                    lower_middle_pane.write(event["text"])

                elif event["type"] == "transcription_received":
                    # Clear and update the entire content because RichLog otherwise treats each delta as a new line
                    middle_pane = self.query_one("#middle-pane", RichLog)
                    middle_pane.clear()
                    if item_id := event.get("item_id", None):
                        middle_pane.write(item_id)
                    middle_pane.write(event["text"])
                    continue

                elif event["type"] == "speech_started":
                    self.speech_ongoing.set()
                    self.speech_done.clear()
                    speech_start_times[event["item_id"]] = datetime.now()
                    log.info("%s Speech started", event["item_id"])

                elif event["type"] == "speech_stopped":
                    self.speech_done.set()
                    self.speech_ongoing.clear()
                    end = datetime.now()
                    start = speech_start_times.get(event["item_id"])
                    if start:
                        duration = (end - start).total_seconds()
                        log.info("Speech ended for %s, %.3f seconds detected", event["item_id"], duration)
                    else:
                        log.warning("Speech ended event for %s with no matching start time", event["item_id"])
            finally:
                self.event_queue.task_done()

    async def send_mic_audio(self) -> None:
        log.info("Starting mic audio task")
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

            try:

                while True:

                    if stream.read_available < read_size:
                        await asyncio.sleep(0)
                        continue

                    await self.should_send_audio.wait()
                    status_indicator.is_recording = True

                    data, _ = stream.read(read_size)

                    sent_audio = await self.ai_service.send_audio(data, sent_audio)

                    rms = audioop.rms(data, 2)  # 2 bytes = 16-bit
                    peak = min(rms / 30000.0, 1.0)  # normalize to 0–1 range

                    amp_widget.amplitude = peak

                    await asyncio.sleep(0)
            except KeyboardInterrupt:
                pass
            finally:
                log.debug("Stopping mic stream...")
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

            await self.ai_service.request_response_if_manual_mode()
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
            log.info("Requesting response manually")
            await self.ai_service.request_response()
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
