# Originally based on openai-python.examples.realtime.push_to_talk.py

import base64
import asyncio
import os
from logging import Handler, LogRecord
import logging.config
from typing import Any, cast, override, Literal
from textual import events

from ai_realtime_text_to_speech_gui.audio_util import CHANNELS, SAMPLE_RATE, AudioPlayerAsync
import audioop
import time
from pathlib import Path
import wave
from datetime import datetime

from textual.app import App, ComposeResult
from textual import on
from textual.events import Unmount
from textual.logging import TextualHandler
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Button, Static, RichLog
from textual.reactive import reactive
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


class TextualLogMessage(Message):
    """A message carrying log text for the UI."""

    def __init__(self, text: str) -> None:
        self.text = text
        super().__init__()


MODEL: str = "gpt-realtime-mini"
LOG_CONFIG_FILE = "logging.conf"
BASE_LOG_DIR = "logs"
PLAY_AUDIO: bool = False
AUDIO_DIR = "recordings"
TOKENS_DIR = "tokens"
SAVE_SILENCE_AFTER_BYTES_MULTIPLIER: int | None = 1000
SAVE_SPEECH_AFTER_BYTES_MULTIPLIER: int | None = None

MODE: Literal["manual", "server", "semantic"] = "manual"
# PROMPT: str = "You are a quiz contestant who answers concisely and clearly a"nd as quickly as possible with just the answer, no need for a full sentence or json, just plaintext"
PROMPT: str = "You are a quiz contestant who answers concisely and clearly and as quickly as possible with just the answer, no need for a full sentence. If the question is a statement or a true/false question, answer true or false first and then give extra context"
IMMEDIATE_INITIALISATION_TIME: int | None = 10
OUTPUT_MODALITIES: list[Literal["text", "audio"]] = ["text"]  # Audio is way more responsive
TRANSCRIPTION_REALTIME_ENABLED = True
TRANSCRIPTION_MODEL = "gpt-4o-mini-transcribe"
LANGUAGE = "en"

if TRANSCRIPTION_REALTIME_ENABLED:
    TRANSCRIPTION_INFO = {
        "language": LANGUAGE,
        "model": TRANSCRIPTION_MODEL,
    }
else:
    TRANSCRIPTION_INFO = None

TURN_DETECTION: ServerVad | SemanticVad | None = None

if MODE == "server":
    TURN_DETECTION: ServerVad = {
        "type": "server_vad",
        "idle_timeout_ms": 5000
    }
elif MODE == "semantic":
    TURN_DETECTION: SemanticVad = {
        "type": "semantic_vad",
        "eagerness": "high"
    }


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

os.makedirs(BASE_LOG_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(TOKENS_DIR, exist_ok=True)

REALTIME_CONVO_CSV = Path(TOKENS_DIR) / "realtime_conversation_tokens.csv"
REALTIME_TOKENS_CSV = Path(TOKENS_DIR) / "realtime_transcribe_tokens.csv"

# Load logging config (.ini)
logging.config.fileConfig(LOG_CONFIG_FILE, disable_existing_loggers=False)
log = logging.getLogger("realtime_app")
transcript_log = logging.getLogger("transcripts")
events_log = logging.getLogger("events")

session_updated = asyncio.Event()
response_in_progress = asyncio.Event()
speech_ongoing = asyncio.Event()
speech_done = asyncio.Event()


log_writer = LogWriter("realtime_tokens")
token_logger_realtime = TokenLogger(log_writer, PricingRealtime(REALTIME_COSTS))
token_logger_realtime_transcription = TokenLogger(log_writer, PricingAudioTranscription(REALTIME_COSTS))
csv_writer_realtime = CSVWriter(REALTIME_CONVO_CSV)
csv_writer_realtime_transcribe = CSVWriter(REALTIME_TOKENS_CSV)
csv_token_logger_realtime = TokenLogger(csv_writer_realtime, PricingRealtime(REALTIME_COSTS))
csv_token_logger_realtime_transcription = TokenLogger(csv_writer_realtime_transcribe, PricingAudioTranscription(REALTIME_COSTS))


class AmplitudeGraph(Widget):
    """Displays a simple bar graph of audio amplitude."""
    amplitude = reactive(0.0)  # 0.0 → 1.0

    def render(self) -> str:
        bar_width = max(1, self.size.width - 4)
        scaled = self.amplitude * 3
        clamped = min(scaled, 1.0)
        filled = int(clamped * bar_width)
        empty = bar_width - filled

        bar = "█" * filled + " " * empty
        return f"[{bar}] {self.amplitude:.2f}"


def write_realtime_tokens(model: str, result: str, usage: RealtimeResponseUsage) -> None:
    token_logger_realtime.record(model, result, usage)
    csv_token_logger_realtime.record(model, result, usage)


def write_realtime_transcribe_tokens(model: str, result: str, usage: UsageTranscriptTextUsageTokens) -> None:
    token_logger_realtime_transcription.record(model, result, usage)
    csv_token_logger_realtime_transcription.record(model, result, usage)


async def save_wav_chunk(pcm_bytes: bytes, suffix: str) -> str:
    """Save PCM16 audio to a WAV file asynchronously."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
    filename = f"audio_{timestamp}_{suffix}.wav"

    bytes_per_frame = CHANNELS * 2  # 2 bytes per sample (16-bit PCM)
    num_frames = len(pcm_bytes) // bytes_per_frame
    duration_sec = num_frames / SAMPLE_RATE

    path = os.path.join(AUDIO_DIR, filename)

    def _save():
        log.debug('Saving wav chunk to %s', filename)
        with wave.open(path, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(pcm_bytes)
        log.info('Saved wav chunk to %s, length %.3f seconds', path, duration_sec)

    await asyncio.to_thread(_save)
    return path


class SessionDisplay(Static):
    """A widget that shows the current session ID."""

    session_id = reactive("")

    @override
    def render(self) -> str:
        return f"Session ID: {self.session_id}" if self.session_id else "Connecting..."


class AudioStatusIndicator(Static):
    """A widget that shows the current audio recording status."""

    is_recording = reactive(False)

    @override
    def render(self) -> str:
        status = (
            "🔴 Recording... (Press K to stop)" if self.is_recording else "⚪ Press K to start recording (Q to quit)"
        )
        return status


class RealtimeApp(App[None]):
    CSS = """
        Screen {
            background: #1a1b26;  /* Dark blue-grey background */
        }

        Container {
            layout: vertical;
            height: 100%;
            border: double rgb(91, 164, 91);
        }

        Horizontal {
            width: 100%;
        }

        #middle-pane {
            width: 100%;
            height: 25%;  /* Reduced to make room for session display */
            border: round rgb(205, 133, 63);
            content-align: center middle;
        }

        #lower-middle-pane {
            width: 100%;
            height: 25%;  /* Reduced to make room for session display */
            border: round rgb(205, 133, 63);
            content-align: center middle;
        }

        #bottom-pane {
            width: 100%;
            height: 75%;  /* Reduced to make room for session display */
            border: round rgb(205, 133, 63);
        }

        #status-row {
            height: 3;
            width: 100%;
            layout: horizontal;
            content-align: center middle;
            margin: 1 1;
        }

        #session-row {
            height: 3;
            width: 100%;
            layout: horizontal;
            content-align: center middle;
            margin: 1 1;
        }

        #status-indicator {
            content-align: center middle;
            width: 1fr;
            height: 3;
            background: #2a2b36;
            border: solid rgb(91, 164, 91);
            padding: 0 1;
        }

        #send-button {
            width: 12;
            height: 3;
            margin-left: 1;
            background: #2a2b36;
            border: solid rgb(91, 164, 91);
        }

        #quit-button {
            width: 12;
            height: 3;
            margin-left: 1;
            background: #2a2b36;
            border: solid rgb(91, 164, 91);
        }

        #session-display {
            height: 3;
            width: 1fr;
            content-align: center middle;
            background: #2a2b36;
            border: solid rgb(91, 164, 91);
            padding: 0 1;
        }

        #amp-graph {
            width: 24;
            height: 3;
            margin-left: 1;
            background: #2a2b36;
            border: solid rgb(91, 164, 91);
        }

        Static {
            color: white;
        }
    """

    client: AsyncOpenAI
    should_send_audio: asyncio.Event
    audio_player: AudioPlayerAsync
    last_audio_item_id: str | None
    connection: AsyncRealtimeConnection | None
    session: Session | None
    connected: asyncio.Event

    def __init__(self) -> None:
        super().__init__()
        self.connection = None
        self.session = None
        self.client = AsyncOpenAI()
        self.audio_player = AudioPlayerAsync()
        self.last_audio_item_id = None
        self.should_send_audio = asyncio.Event()
        self.connected = asyncio.Event()
        self.start_time = time.time()

    @on(Unmount)
    async def cleanup_resources(self):
        self.log("The app is unmounting now!")
        await asyncio.gather(
            asyncio.to_thread(token_logger_realtime.close()),
            asyncio.to_thread(token_logger_realtime_transcription.close()),
            asyncio.to_thread(csv_token_logger_realtime.close()),
            asyncio.to_thread(csv_writer_realtime_transcribe.close())
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
        textual_handler = TextualHandler()
        textual_handler.setLevel(logging.DEBUG)
        textual_handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s]: %(message)s"))
        logging.getLogger("realtime_app").addHandler(textual_handler)

        self.run_worker(self.handle_realtime_connection())
        self.run_worker(self.send_mic_audio())

        async def listen_once():
            await asyncio.sleep(1)
            # Simulate pressing lowercase "k"
            await self.toggle_recording()

            await asyncio.sleep(IMMEDIATE_INITIALISATION_TIME)
            log.info('Runtime finshed, existing...')
            await self.toggle_recording()

        if IMMEDIATE_INITIALISATION_TIME:
            self.run_worker(listen_once())

    async def on_textual_log_message(self, message: TextualLogMessage) -> None:
        """Receive log messages sent by the log handler and write them to the pane."""
        pane = self.query_one("#bottom-pane", RichLog)
        pane.write(message.text)

    async def handle_realtime_connection(self) -> None:
        async with self.client.realtime.connect(model=MODEL) as conn:
            self.connection = conn
            self.connected.set()

            # note: this is the default and can be omitted
            # if you want to manually handle VAD yourself, then set `'turn_detection': None`

            log.info("Starting session with model %s, prompt %s", MODEL, PROMPT)
            log.info("Turn Detection: %s", TURN_DETECTION)

            await conn.session.update(
                session={
                    "audio": {
                        "input": {"turn_detection": TURN_DETECTION,
                                  "transcription": TRANSCRIPTION_INFO
                                  },
                    },
                    "instructions": PROMPT,
                    "output_modalities": OUTPUT_MODALITIES,
                    "model": MODEL,
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
                    session_updated.set()
                    self.session = event.session
                    continue

                if event.type == "response.output_audio.delta":
                    if event.item_id != self.last_audio_item_id:
                        log.info("First audio response received for %s", event.item_id)
                        self.audio_player.reset_frame_count()
                        self.last_audio_item_id = event.item_id

                    bytes_data = base64.b64decode(event.delta)
                    if PLAY_AUDIO:
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
                        await asyncio.to_thread(write_realtime_transcribe_tokens, MODEL, text, usage)
                    else:
                        log.warning("No token usage info in transcription response.")
                        continue

                    continue

                if event.type == "response.created":
                    response_in_progress.set()
                    log.info("%s Response is being created", event.response.id)
                    continue

                if event.type == "input_audio_buffer.speech_started":
                    speech_ongoing.set()
                    speech_done.clear()
                    speech_start_times[event.item_id] = datetime.now()
                    log.info("%s Speech started", event.item_id)
                    continue

                if event.type == "input_audio_buffer.speech_stopped":
                    speech_done.set()
                    speech_ongoing.clear()
                    end = datetime.now()
                    start = speech_start_times.get(event.item_id)
                    if start:
                        duration = (end - start).total_seconds()
                        log.info("Speech ended for %s, %.3f seconds detected", event.item_id, duration)
                    else:
                        log.warning("Speech ended event for %s with no matching start time", event.item_id)
                    continue

                if event.type == "response.done":
                    response_in_progress.clear()
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
                                                MODEL,
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

    async def _get_connection(self) -> AsyncRealtimeConnection:
        await self.connected.wait()
        assert self.connection is not None
        return self.connection

    async def send_mic_audio(self) -> None:
        try:
            await asyncio.wait_for(session_updated.wait(), timeout=10)
        except asyncio.TimeoutError:
            log.error("Failed to update session, existing")
            self.exit()
        async with asyncio.TaskGroup() as tg:
            import sounddevice as sd  # type: ignore

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
                        if response_in_progress.is_set():
                            log.info("Sending initial cancel response...")
                            await connection.send({"type": "response.cancel"})
                        sent_audio = True

                    await connection.input_audio_buffer.append(audio=base64.b64encode(cast(Any, data)).decode("utf-8"))

                    rms = audioop.rms(data, 2)  # 2 bytes = 16-bit
                    peak = min(rms / 30000.0, 1.0)  # normalize to 0–1 range

                    amp_widget = self.query_one("#amp-graph", AmplitudeGraph)
                    amp_widget.amplitude = peak

                    binary_data = data.tobytes()

                    wav_stream += binary_data

                    if not speech_ongoing.is_set():
                        if speech_done.is_set():
                            tg.create_task(save_wav_chunk(wav_stream, "speech_done"))
                            wav_stream = b''
                            speech_done.clear()
                        elif SAVE_SILENCE_AFTER_BYTES_MULTIPLIER and len(wav_stream) > (
                                read_size * SAVE_SILENCE_AFTER_BYTES_MULTIPLIER):
                            tg.create_task(save_wav_chunk(wav_stream, "periodic"))
                            wav_stream = b''
                    elif SAVE_SPEECH_AFTER_BYTES_MULTIPLIER and len(wav_stream) > (
                            read_size * SAVE_SPEECH_AFTER_BYTES_MULTIPLIER):
                        tg.create_task(save_wav_chunk(wav_stream, "speech"))
                        wav_stream = b''

                    await asyncio.sleep(0)
            except KeyboardInterrupt:
                pass
            finally:
                if wav_stream:
                    tg.create_task(
                        save_wav_chunk(wav_stream, "periodic"))
                stream.stop()
                stream.close()

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "send-button":
            log.info("Button pressed, toggle recording...")
            await self.toggle_recording()
            return
        if event.button.id == "quit-button":
            log.info("Button pressed, quitting...")
            self.exit()

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


class TextualPaneLogHandler(Handler):
    """
    Logging handler that forwards log messages into the Textual app
    using message posting (thread-safe).
    """

    def __init__(self, app: "RealtimeApp"):
        super().__init__()
        self.app = app

    def emit(self, record: LogRecord):
        try:
            msg = self.format(record)
            # Post message safely into Textual's event queue
            self.app.post_message(TextualLogMessage(msg))
        except Exception:
            self.handleError(record)


if __name__ == "__main__":
    app = RealtimeApp()
    app.run()

