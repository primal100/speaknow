import asyncio
import logging
import os
import wave
from datetime import datetime
from pathlib import Path


log = logging.getLogger("realtime_app")


def update_log_config(log_config_file: str | Path = None, base_log_dir: str | Path = None):
    config_path = Path(log_config_file)

    if not config_path.exists():
        print(f"Error: {log_config_file} not found.")
        return

    content = config_path.read_text()

    # 2. Replace the placeholder with the actual variable value
    updated_content = content.replace("%BASE_LOG_DIR%", base_log_dir.as_posix())

    # 3. Write it back (or to a new file)
    config_path.write_text(updated_content)


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
