realtime_costs = {
    "gpt-realtime": {
        "text_in": 4.00,
        "cached_text_in": 0.40,
        "text_out": 16.00,
        "audio_in": 32.00,
        "audio_out": 64.0,
        "image_in": 5.00,
        "cached_image_in": 0.50,
        "cached_audio_in": 0.40,
    },
    "gpt-realtime-mini": {
        "text_in": 0.60,
        "cached_text_in": 0.06,
        "text_out": 2.40,
        "audio_in": 10.00,
        "audio_out": 20.00,
        "image_in": 0.80,
        "cached_image_in": 0.08,
        "cached_audio_in": 0.30,
    },
    "gemini-2.5-flash-native-audio-preview-12-2025": {
        "text_in": 0.10,
        "image_in": 0.10,
        "video_in": 0.10,
        "audio_in": 0.30,
        "cached_text_in": 0.01,
        "cached_video_in": 0.01,
        "cached_image_in": 0.01,
        "cached_audio_in": 0.03,
        "text_out": 0.40,
        "audio_out": 0.40,
        "thinking": 0.40
    }
}


transcription_costs: dict[str, dict[str, float]] = {
    "gpt-4o-transcribe": {
        "audio_in": 2.50,
        "text_out": 10.00
    },
    "gpt-4o-mini-transcribe": {
        "audio_in": 1.25,
        "text_out": 5.00
    },
}