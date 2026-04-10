# live-translator

Records computer or microphone audio in real time and translates it to Brazilian Portuguese using OpenAI Whisper and GPT-4o mini.

## How it works

1. Audio is captured from the selected input device in 30-second chunks
2. Silent chunks are skipped automatically
3. Each chunk is transcribed by Whisper (language-agnostic)
4. The transcription is translated to pt-BR by GPT-4o mini

## Requirements

- Python 3.13+
- [uv](https://github.com/astral-sh/uv)
- An OpenAI API key

## Installation

```bash
git clone <repo>
cd live-translator
uv sync
```

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-...
```

## Capturing computer audio (macOS)

To translate audio playing on your computer (not just your microphone), you need a loopback device.

1. Install [BlackHole 2ch](https://github.com/ExistingRealAudio/BlackHole) (free)
2. Open **Audio MIDI Setup** (Spotlight → "Audio MIDI Setup")
3. Click **+** → **Create Multi-Output Device**
4. Check both **BlackHole 2ch** and your speakers/headphones
5. In **System Preferences → Sound → Output**, select the new Multi-Output Device

Audio will now play through your speakers and be routed into BlackHole simultaneously.

> **Note:** volume keys stop working with a Multi-Output Device. To fix it, check "Use this device for sound output" on the speakers entry inside the Multi-Output Device in Audio MIDI Setup.

## Usage

```bash
# List available input devices
uv run live-translator --list-devices

# Translate from BlackHole (computer audio)
uv run live-translator --device-name blackhole

# Or select by device index
uv run live-translator --device 3

# Adjust chunk size (default: 30s)
uv run live-translator --device-name blackhole --chunk 10
```

### Options

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--list-devices` | `-l` | — | List available input devices and exit |
| `--device` | `-d` | system default | Input device index |
| `--device-name` | `-n` | — | Select device by name substring (e.g. `blackhole`) |
| `--chunk` | `-c` | `30` | Audio buffer length in seconds |
| `--silence-threshold` | `-s` | `0.005` | RMS level below which chunks are skipped |
