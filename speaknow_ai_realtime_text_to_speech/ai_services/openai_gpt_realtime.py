import asyncio
import base64
import datetime
import logging
import os

from numpy import ndarray
from gpt_token_tracker.pricing import PricingRealtime, PricingAudioTranscription

from openai import AsyncOpenAI
from openai.types.realtime.session_update_event_param import Session  # https://github.com/openai/openai-python/pull/2803
from openai.resources.realtime.realtime import AsyncRealtimeConnection  # Another bug?
from openai.types.realtime.realtime_audio_input_turn_detection_param import ServerVad, SemanticVad
from typing import Any, cast
from .base import BaseAIService, LogWriter


log_writer = LogWriter("realtime_tokens")
log = logging.getLogger("realtime_app")
events_log = logging.getLogger("events")


class OpenAIGPTRealtime(BaseAIService):
    prefix = "openai"
    client: AsyncOpenAI
    realtime_pricing_cls = PricingRealtime
    transcription_pricing_cls = PricingAudioTranscription
    connection: AsyncRealtimeConnection | None
    response_in_progress: asyncio.Event
    session: Session | None
    last_audio_item_id: str | None

    @classmethod
    def set_default_config_options_on_change(cls) -> dict[str, Any]:
        return {
            'model': "gpt-realtime-mini",
            'base_url': os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1",
            'api_key_env': "OPENAI_API_KEY",
            'transcription_model': "gpt-4o-mini-transcribe",
        }

    def __init__(self, user_config: dict[str, Any]):
        super().__init__(user_config)
        self.client = AsyncOpenAI(
            api_key=os.environ[self.user_config["api_key_env"]],
            base_url=self.user_config["base_url"] or None
        )
        self.connection = None
        self.session = None
        self.connected = asyncio.Event()
        self.response_in_progress = asyncio.Event()
        self.last_audio_item_id = None

    async def _get_connection(self) -> AsyncRealtimeConnection:
        await self.connected.wait()
        assert self.connection is not None
        return self.connection

    async def send_audio(self, data: ndarray, sent_audio: bool) -> bool:
        connection = await self._get_connection()
        if not sent_audio:
            if self.response_in_progress.is_set():
                log.info("Sending initial cancel response...")
                await connection.send({"type": "response.cancel"})
            sent_audio = True

        await connection.input_audio_buffer.append(audio=base64.b64encode(cast(Any, data)).decode("utf-8"))
        return sent_audio

    async def handle_realtime_connection(self, event_queue: asyncio.Queue[dict[str, Any]]) -> None:
        try:
            async with self.client.realtime.connect(model=self.user_config['model']) as conn:
                connection_start_time = datetime.datetime.now()
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

                async for event in conn:
                    events_log.info("Event Type: %s. Item Id: %s", event.type, getattr(event, "item_id", ""))
                    events_log.debug(event)
                    if event.type == "session.created":
                        # Not sent by Grok
                        self.session = event.session
                        assert self.session.id is not None
                        event_queue.put_nowait({"type": "session_id", "session_id": self.session.id})
                        continue

                    if event.type == "session.updated":
                        event_queue.put_nowait({"type": "session_updated"})
                        self.session = event.session
                        # Grok has no session ID
                        event_queue.put_nowait({"type": "session_id",
                                                "session_id": f'{getattr(self.session, "model", "")} {getattr(self.session, "id", "")}'})
                        continue

                    if event.type == "response.output_audio.delta":
                        is_first = False
                        if event.item_id != self.last_audio_item_id:
                            log.info("First audio response received for %s", event.item_id)
                            is_first = True
                        bytes_data = base64.b64decode(event.delta)
                        event_queue.put_nowait({"type": "audio_response", "item_id": event.item_id,
                                                "is_first_in_response": is_first, "data": bytes_data})
                        self.last_audio_item_id = event.item_id
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
                        event_queue.put_nowait({"type": "response_received", "item_id": event.item_id, "text": acc_items[event.item_id]})
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

                        event_queue.put_nowait({"type": "transcription_received", "item_id": event.item_id, "text": transcription_items[event.item_id]})

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
                            await self.write_realtime_transcribe_tokens_wrapper(
                                self.user_config['transcription_model'],
                                text, usage)
                        continue

                    if event.type == "response.created":
                        self.response_in_progress.set()
                        log.info("%s Response is being created", event.response.id)
                        continue

                    if event.type == "input_audio_buffer.speech_started":
                        event_queue.put_nowait({"type": "speech_started", "item_id": event.item_id})
                        continue

                    if event.type == "input_audio_buffer.speech_stopped":
                        event_queue.put_nowait({"type": "speech_stopped", "item_id": event.item_id})
                        continue

                    if event.type == "response.done":
                        self.response_in_progress.clear()
                        status = event.response.status
                        status_details = event.response.status_details
                        result = None
                        if output := event.response.output:
                            item = output[0]
                            if content := getattr(item, "content", None):
                                result = getattr(content[0], "text", None) or getattr(content[0], "transcript", None)
                        if status_details and hasattr(status_details, "type"):  # Grok has status_details=='unimplemented')
                            log.info("%s Response is done, status: %s, type: %s, reason: %s, error: %s, result: %s",
                                     event.response.id, status, getattr(status_details, "type", ""),
                                     getattr(status_details, "reason", ""), getattr(status_details, "error", ""), result)
                        else:
                            log.info("%s Response is done, status: %s, result: %s", event.response.id, status, result)
                        if usage := getattr(event.response, "usage"):
                            if getattr(usage, "total_tokens", None) is not None:
                                await self.write_realtime_tokens_wrapper(
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
            await self.write_connection_time(self.user_config['model'], connection_start_time)
            log.debug('Clearing events')
            self.connected.clear()
            self.connection = None

    async def request_response(self) -> None:
        conn = await self._get_connection()
        await conn.input_audio_buffer.commit()
        await conn.response.create()

    async def request_response_if_manual_mode(self) -> None:
        if self.session and self.session.audio.input.turn_detection is None:  # Bugfix
            # The default in the API is that the model will automatically detect when the user has
            # stopped talking and then start responding itself.
            #
            # However if we're in manual `turn_detection` mode then we need to
            # manually tell the model to commit the audio buffer and start responding.
            await self.request_response()
