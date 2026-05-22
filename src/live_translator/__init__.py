"""Live audio translator — records audio and outputs Brazilian Portuguese text."""

import argparse
import io
import queue
import sys
import termios
import threading
import tty
import wave

import numpy as np
import sounddevice as sd
from dotenv import load_dotenv
from openai import OpenAI

SAMPLE_RATE = 16000
CHANNELS = 1


def list_devices() -> None:
    devices = sd.query_devices()
    default_in = sd.default.device[0]
    print("Available input devices:")
    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            marker = " <-- default" if i == default_in else ""
            print(f"  [{i:2d}] {dev['name']}{marker}")


def _to_wav_bytes(audio: np.ndarray) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes((audio * 32767).astype(np.int16).tobytes())
    buf.seek(0)
    return buf.read()


def _translate(client: OpenAI, audio: np.ndarray) -> str | None:
    wav = io.BytesIO(_to_wav_bytes(audio))
    wav.name = "audio.wav"
    try:
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=wav,
            response_format="text",
        )
        source_text = (
            transcription.strip()
            if isinstance(transcription, str)
            else str(transcription).strip()
        )
        if not source_text:
            return None

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a translator. Translate the user's text to Brazilian Portuguese (pt-BR). "
                        "Output only the translated text, nothing else. "
                        "If the text is already in Brazilian Portuguese, output it as-is."
                    ),
                },
                {"role": "user", "content": source_text},
            ],
            temperature=0,
        )
        return response.choices[0].message.content.strip() or None
    except Exception as exc:
        print(f"\n[error] {exc}", file=sys.stderr)
        return None


def _read_key() -> str:
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Record audio and translate it to Brazilian Portuguese in real time."
    )
    parser.add_argument(
        "--list-devices",
        "-l",
        action="store_true",
        help="List available input devices and exit",
    )
    parser.add_argument(
        "--device",
        "-d",
        type=int,
        default=None,
        metavar="INDEX",
        help="Input device index (default: system default)",
    )
    parser.add_argument(
        "--device-name",
        "-n",
        type=str,
        default=None,
        metavar="NAME",
        help="Select input device by name substring (case-insensitive), e.g. 'blackhole'",
    )
    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return

    if args.device_name:
        name_lower = args.device_name.lower()
        matches = [
            i
            for i, d in enumerate(sd.query_devices())
            if d["max_input_channels"] > 0 and name_lower in d["name"].lower()
        ]
        if not matches:
            print(
                f"[error] No input device matching '{args.device_name}'.",
                file=sys.stderr,
            )
            print("Run with --list-devices to see available devices.", file=sys.stderr)
            sys.exit(1)
        if len(matches) > 1:
            print(
                f"[warning] Multiple devices match '{args.device_name}', using first:",
                file=sys.stderr,
            )
            for i in matches:
                print(f"  [{i}] {sd.query_devices(i)['name']}", file=sys.stderr)
        args.device = matches[0]

    client = OpenAI()

    raw_queue: queue.Queue[np.ndarray] = queue.Queue()
    translate_queue: queue.Queue[np.ndarray] = queue.Queue()
    cut_event = threading.Event()

    def _audio_callback(indata, frames, time_info, status):
        if status:
            print(f"\n[audio] {status}", file=sys.stderr)
        raw_queue.put(indata[:, 0].copy())

    def _record_loop():
        buf = np.zeros(0, dtype=np.float32)
        while True:
            try:
                buf = np.concatenate([buf, raw_queue.get(timeout=0.05)])
            except queue.Empty:
                pass

            if cut_event.is_set():
                cut_event.clear()
                if len(buf) > 0:
                    translate_queue.put(buf)
                buf = np.zeros(0, dtype=np.float32)

    def _translate_loop():
        while True:
            chunk = translate_queue.get()
            text = _translate(client, chunk)
            if text:
                print(f"\n{text}", flush=True)

    device_name = sd.query_devices(args.device or sd.default.device[0])["name"]
    print(f"Device : {device_name}")
    print("Press any key to send chunk. Ctrl+C to stop.\n")

    for target in (_record_loop, _translate_loop):
        threading.Thread(target=target, daemon=True).start()

    try:
        with sd.InputStream(
            device=args.device,
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=np.float32,
            callback=_audio_callback,
            blocksize=1024,
        ):
            print("Recording...", flush=True)
            while True:
                _read_key()
                cut_event.set()
                print()
    except KeyboardInterrupt:
        print("\n\nStopped.")
    except Exception as exc:
        print(f"\n[error] {exc}", file=sys.stderr)
        sys.exit(1)
