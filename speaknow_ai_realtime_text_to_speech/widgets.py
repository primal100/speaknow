from textual.app import App
from textual import on
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.widget import Widget
from textual.widgets import Static, Input, Label, Select, Checkbox, Button
from textual.app import ComposeResult
from logging import Handler, LogRecord
from .ai_services import ai_services, AI_SERVICES_SELECTION
from typing_extensions import override


default_service = "openai"


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


class SessionDisplay(Static):
    """A widget that shows the current session ID."""

    session_id = reactive("")

    @override
    def render(self) -> str:
        return f"Session: {self.session_id}" if self.session_id else "Connecting..."


class AudioStatusIndicator(Static):
    """A widget that shows the current audio recording status."""

    is_recording = reactive(False)

    @override
    def render(self) -> str:
        status = (
            "🔴 Recording... (Press K to stop)" if self.is_recording else "⚪ Press K to start recording (Q to quit)"
        )
        return status


class TextualLogMessage(Message):
    """A message carrying log text for the UI."""

    def __init__(self, text: str) -> None:
        self.text = text
        super().__init__()


class TextualPaneLogHandler(Handler):
    """
    Logging handler that forwards log messages into the Textual app
    using message posting (thread-safe).
    """

    def __init__(self, app: App[None]):
        super().__init__()
        self.app = app

    def emit(self, record: LogRecord):
        try:
            msg = self.format(record)
            # Post message safely into Textual's event queue
            self.app.post_message(TextualLogMessage(msg))
        except Exception:
            self.handleError(record)


class ConfigModal(ModalScreen[dict]):
    """A modal screen to edit application settings."""

    def __init__(self):
        # Flag to prevent default values from overwriting existing config during startup
        self._is_loading = True
        super().__init__()

    def on_mount(self) -> None:
        # After mounting and setting initial values, we clear the flag
        # We use call_after_refresh to ensure the initial 'Changed' events have fired
        self.call_after_refresh(self._finish_loading)

    def _finish_loading(self):
        self._is_loading = False

    def compose(self) -> ComposeResult:
        cfg = self.app.user_config

        # Sticky Header
        yield Label("Application Settings", id="config-title")

        # Scrollable Middle Section
        with VerticalScroll(id="config-body"):
            yield Label("AI Service")
            yield Select(
                AI_SERVICES_SELECTION,
                value=cfg["ai_service"],
                name="ai_service",
                id="ai-service-select"
            )

            with Horizontal(id="model-settings-row"):
                with Vertical():
                    yield Label("Model Name")
                    yield Input(value=cfg["model"], name="model")

                with Vertical():
                    yield Label("Base Url")
                    yield Input(value=cfg["base_url"], name="base_url", placeholder="Default URL")

                with Vertical():
                    yield Label("API Key Environment Variable")
                    yield Input(value=cfg["api_key_env"], name="api_key_env")

            yield Label("System Prompt")
            yield Input(value=cfg["prompt"], name="prompt")

            yield Label("Mode")
            yield Select(
                [("Manual", "manual"), ("Server VAD", "server"), ("Semantic VAD", "semantic")],
                value=cfg["mode"],
                name="mode",
            )

            current_modalities = cfg.get("output_modalities", ["text", "audio"])
            yield Label("Output Modalities")
            with Horizontal(id="modalities-row"):
                yield Checkbox("Audio", value="audio" in current_modalities, name="audio", classes="modality-check")

            yield Label("Transcription Settings")
            yield Checkbox("Transcription Enabled", value=cfg["transcription_enabled"], name="transcription_enabled")
            yield Label("Transcription Model")
            yield Input(value=cfg["transcription_model"], name="transcription_model", placeholder="Model Name")
            yield Label("Transcription Language")
            yield Input(value=cfg["language"], name="language", placeholder="Language (en, fr, etc)")

            yield Label("Miscellaneous Settings")
            yield Checkbox("Play Audio", value=cfg["play_audio"], name="play_audio")
            yield Checkbox("Immediate Initialization", value=cfg["immediate_initialisation"],
                           name="immediate_initialisation")

            yield Label("Save Token Usage Data")
            with Horizontal(classes="input-row"):
                with Vertical():
                    yield Label("Save tokens")
                    yield Checkbox("Save Token Data", value=cfg["save_token_data"], name="save_token_data")
                with Vertical():
                    yield Label("Save chat output result in tokens file")
                    yield Checkbox("Save Result", value=cfg["save_result"], name="save_result")

        # Sticky Footer
        with Horizontal(id="config-buttons"):
            yield Button("Save", variant="success", id="save_config")
            yield Button("Cancel", variant="error", id="cancel_config")

    def get_field(self, fieldname: str) -> Widget:
        return self.query_one(f"Input[name='{fieldname}'], Select[name='{fieldname}']")

    @on(Select.Changed, "#ai-service-select")
    def handle_service_change(self, event: Select.Changed) -> None:
        new_service = event.value
        ai_service_cls = ai_services[new_service]
        changes = ai_service_cls.set_default_config_options_on_change()

        disabled_list = changes.get("DISABLED", [])

        # Query all interactive controls in the modal
        for control in self.query("Input, Select, Checkbox"):
            fieldname = control.name
            if not fieldname:
                continue

            # 1. Update Disabled state
            # This re-enables fields NOT in the list and disables those that ARE
            control.disabled = (fieldname in disabled_list)

            # 2. Update Value if specified
            if not self._is_loading and fieldname in changes:
                new_val = changes[fieldname]

                # Checkbox specific logic: ensure it gets a boolean
                if isinstance(control, Checkbox):
                    control.value = bool(new_val)
                else:
                    control.value = str(new_val)

    @on(Button.Pressed)
    def handle_buttons(self, event: Button.Pressed) -> None:
        if event.button.id == "save_config":
            new_settings = {}
            for control in self.query("Input, Select, Checkbox").exclude(".modality-check"):
                val = control.value

                # Special handling for our number fields
                if input_type := getattr(control, "type", None):
                    if any(input_type == number_type for number_type in ("integer", "number")):
                        try:
                            val = int(val) if val.strip() != "" else None
                        except ValueError:
                            val = None

                new_settings[control.name] = val
            selected_modalities = [
                cb.name for cb in self.query(".modality-check") if cb.value is True
            ]
            if not selected_modalities:
                selected_modalities = ["text"]
            new_settings["output_modalities"] = selected_modalities
            self.dismiss(new_settings)
        else:
            self.dismiss()
