import asyncio
import logging
import os
import wave
from datetime import datetime
from pathlib import Path


log = logging.getLogger("realtime_app")


async def save_wav_chunk(pcm_bytes: bytes, suffix: str, channels: int, sample_rate: int, audio_dir: str | Path) -> str:
    """Save PCM16 audio to a WAV file asynchronously."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
    filename = f"audio_{timestamp}_{suffix}.wav"

    bytes_per_frame = channels * 2  # 2 bytes per sample (16-bit PCM)
    num_frames = len(pcm_bytes) // bytes_per_frame
    duration_sec = num_frames / sample_rate

    path = os.path.join(audio_dir, filename)

    def _save():
        log.debug('Saving wav chunk to %s', filename)
        with wave.open(path, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_bytes)
        log.info('Saved wav chunk to %s, length %.3f seconds', path, duration_sec)

    await asyncio.to_thread(_save)
    return path
