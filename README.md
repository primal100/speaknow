# SpeakNow

SpeakNow is a high-performance, real-time AI voice interface built on the OpenAI Realtime API. It provides a seamless, low-latency speech-to-speech conversational experience directly in your terminal.

This project is based on and inspired by the `push_to_talk_app.py` example from the [openai-python](https://github.com/openai/openai-python/blob/main/examples/realtime/push_to_talk_app.py) repository but with lots of features added.

## Features

* **Low-Latency Speech-to-Speech:** Direct multimodal interaction using the `gpt-realtime` or `gpt-realtime-mini` models for near-instant responses.
* **Real-time Transcription:** View live streaming transcripts of your conversation as you speak.
* **Advanced Audio Handling:** Save input speech to local WAV files for record-keeping or debugging.
* **Configurable Parameters:** Easily adjust system prompts, model names, mode, and transcription options through a built-in TUI settings menu.
* **Professional TUI:** A clean, "sticky" interface with persistent headers, footers, and scrollable settings panes using ```textual```.
* **Voice Amplitude Monitor:** Monitor the volume of the voice input

## Installation

SpeakNow requires **Python 3.11 or greater**.

To install the latest version from PyPI, run:

```bash
pip install speaknow-gui
```

For linux, portaudio19-dev and ffmpeg are required. For example, to install on Ubuntu:

```bash
sudo apt install portaudio19-dev ffmpeg
```

## Usage

### Configuration
Before running, ensure your OPENAI_API_KEY is set in your environment variables. 

In Windows open the Edit Environmnent Variables GUI and add it there.

In Linux:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

SpeakNow provides two main entry points for different use cases.

### 1. Standard Application
Launch the main TUI application to start a real-time session:

Windows:
```powershell
speaknow.exe
```

Linux:
```bash
speaknow
```

![GUI preview](images/gui_preview.png)
![Config preview](images/config_preview.png)

### 2. Web Service Mode
Run a server-side version optimized for shared or remote environments:

Windows:
```powershell
speaknow-serve.exe
```

Linux:
```bash
speaknow-serve
```

### Modes:
The mode can be changed in configuration.
Manual mode is triggered by hitting "Start," speaking and then hitting "Stop." to send the audio.
Server VAD and uses periods of silence to automatically chunk the audio.
Semantic VAD  uses a semantic classifier to detect when the user has finished speaking, based on the words they have uttered.