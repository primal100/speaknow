# SpeakNow

SpeakNow is a high-performance, real-time AI voice interface built on the OpenAI Realtime API. It provides a seamless, low-latency speech-to-speech conversational experience directly in your terminal.

This project is based on and inspired by the `push_to_talk_app.py` example from the [openai-python](https://github.com/openai/openai-python/blob/main/examples/realtime/push_to_talk_app.py) repository.

## Features

* **Low-Latency Speech-to-Speech:** Direct multimodal interaction using the `gpt-4o-realtime` model for near-instant responses.
* **Real-time Transcription:** View live streaming transcripts of your conversation as you speak.
* **Dual Mode Support:** * **Manual (Push-to-Talk):** Total control over when the AI listens.
    * **Server VAD:** Automatic voice activity detection for a more natural, hands-free flow.
* **Advanced Audio Handling:** Save input speech to local WAV files for record-keeping or debugging.
* **Configurable Parameters:** Easily adjust system prompts, model names, and transcription languages through a built-in TUI settings menu.
* **Professional TUI:** A clean, "sticky" interface with persistent headers, footers, and scrollable settings panes.

## Installation

SpeakNow requires **Python 3.11 or greater**.

To install the latest version from PyPI, run:

```bash
pip install speaknow
```

## Usage

SpeakNow provides two main entry points for different use cases.

### 1. Standard Application
Launch the main TUI application to start a real-time session:

```bash
speaknow
```

2. Web Service Mode
Run a server-side version optimized for shared or remote environments:

Bash

speaknow-serve
Configuration
Before running, ensure your OPENAI_API_KEY is set in your environment variables:

Bash

export OPENAI_API_KEY="your-api-key-here"
Acknowledgments
Special thanks to the OpenAI team for the original push_to_talk_app.py example which served as the foundation for this interface.

License
This project is licensed under the Apache License 2.0. See the LICENSE file for details.