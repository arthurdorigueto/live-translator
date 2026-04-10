"""Live audio translator — records audio and outputs English text."""

import argparse
import io
import queue
import sys
import threading
import time
import wave

import numpy as np
import sounddevice as sd
from dotenv import load_dotenv
from openai import OpenAI

SAMPLE_RATE = 16000
CHANNELS = 1
DEFAULT_CHUNK_SECONDS = 30
DEFAULT_SILENCE_THRESHOLD = 0.005  # RMS below this is silence


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


def _is_silent(audio: np.ndarray, threshold: float) -> bool:
    return float(np.sqrt(np.mean(audio**2))) < threshold


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


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Record audio and translate it to English in real time."
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
    parser.add_argument(
        "--chunk",
        "-c",
        type=float,
        default=DEFAULT_CHUNK_SECONDS,
        metavar="SECONDS",
        help=f"Audio chunk length in seconds (default: {DEFAULT_CHUNK_SECONDS})",
    )
    parser.add_argument(
        "--silence-threshold",
        "-s",
        type=float,
        default=DEFAULT_SILENCE_THRESHOLD,
        metavar="RMS",
        help=f"Skip chunks quieter than this RMS level (default: {DEFAULT_SILENCE_THRESHOLD})",
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

    client = OpenAI()  # reads OPENAI_API_KEY from env

    raw_queue: queue.Queue[np.ndarray] = queue.Queue()
    translate_queue: queue.Queue[np.ndarray] = queue.Queue()
    chunk_samples = int(args.chunk * SAMPLE_RATE)

    # --- Thread 1: audio callback (runs in sounddevice's thread) ---
    def _audio_callback(indata, frames, time_info, status):
        if status:
            print(f"\n[audio] {status}", file=sys.stderr)
        raw_queue.put(indata[:, 0].copy())

    # --- Thread 2: accumulate raw audio into fixed-size chunks ---
    def _record_loop():
        buf = np.zeros(0, dtype=np.float32)
        while True:
            try:
                buf = np.concatenate([buf, raw_queue.get(timeout=0.5)])
            except queue.Empty:
                continue

            while len(buf) >= chunk_samples:
                chunk, buf = buf[:chunk_samples], buf[chunk_samples:]
                if _is_silent(chunk, args.silence_threshold):
                    print("·", end="", flush=True)
                else:
                    translate_queue.put(chunk)

    # --- Thread 3: translate chunks and print ---
    def _translate_loop():
        while True:
            chunk = translate_queue.get()
            text = _translate(client, chunk)
            if text:
                print(f"\n{text}", flush=True)

    device_name = sd.query_devices(args.device or sd.default.device[0])["name"]
    print(f"Device : {device_name}")
    print(f"Chunk  : {args.chunk}s | Silence threshold: {args.silence_threshold} RMS")
    print("Listening... (Ctrl+C to stop)\n")

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
            while True:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n\nStopped.")
    except Exception as exc:
        print(f"\n[error] {exc}", file=sys.stderr)
        sys.exit(1)
