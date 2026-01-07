from typing import Any
from .openai_gpt_realtime import OpenAIGPTRealtime
from gpt_token_tracker.pricing_timed import PricingTimeInterval


class XAIGrokVoice(OpenAIGPTRealtime):
    """
    Grok Voice returns Usage for compatibiltiy but all values are None.
    Grok Voice charges by time, not tokens.
    """

    prefix = "xai_grok"
    realtime_pricing_cls = PricingTimeInterval
    transcription_pricing_cls = None
    log_token_usage = False
    log_per_interval_pricing = True

    @classmethod
    def set_default_config_options_on_change(cls) -> dict[str, Any]:
        return {
            'model': "grok-4",
            'base_url': "https://api.x.ai/v1",
            'api_key_env': "XAI_API_KEY",
            'transcription_model': "gpt-4o-mini-transcribe",
        }
