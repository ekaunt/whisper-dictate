#!/usr/bin/env python3
"""Local Whisper dictation daemon.

Loads a faster-whisper model once and serves transcription requests over a
Unix socket at $XDG_RUNTIME_DIR/whisper-dictate.sock. One JSON request per
connection, newline-terminated:

    {"cmd": "transcribe", "wav": "/tmp/dictate-XXXX.wav"}  -> {"text": "...", "ms": 420}
    {"cmd": "ping"}                                         -> {"ok": true}
"""
from __future__ import annotations

import json
import os
import signal
import socketserver
import sys
import threading
import time
import wave
from pathlib import Path

import ctranslate2
from faster_whisper import WhisperModel

MODEL_ID = "Systran/faster-distil-whisper-large-v3"
SOCKET_PATH = Path(os.environ["XDG_RUNTIME_DIR"]) / "whisper-dictate.sock"


def load_model() -> WhisperModel:
    # Default off so the daemon doesn't hold the dGPU awake (~5W idle on
    # laptops with NVIDIA runtime PM). Set WHISPER_DICTATE_GPU=1 to opt in.
    gpu = os.environ.get("WHISPER_DICTATE_GPU") == "1" and ctranslate2.get_cuda_device_count() > 0
    device = "cuda" if gpu else "cpu"
    compute_type = "float16" if gpu else "int8"
    print(f"loading {MODEL_ID} on {device} ({compute_type})", flush=True)
    model = WhisperModel(
        MODEL_ID,
        device=device,
        compute_type=compute_type,
        cpu_threads=8,
    )
    print("warming up", flush=True)
    silence = SOCKET_PATH.parent / "whisper-warmup.wav"
    with wave.open(str(silence), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 16000)
    for _ in model.transcribe(str(silence), language="en")[0]:
        pass
    silence.unlink(missing_ok=True)
    print("ready", flush=True)
    return model


def send(wfile, obj) -> None:
    wfile.write((json.dumps(obj) + "\n").encode())
    wfile.flush()


def transcribe_stream(model: WhisperModel, wav: str, wfile) -> None:
    """Stream {"partial": "..."} per segment as it's decoded; finish with {"done": true, ...}."""
    t0 = time.monotonic()
    accumulated = ""
    segments, _ = model.transcribe(wav, language="en", vad_filter=True)
    for seg in segments:
        accumulated += seg.text
        send(wfile, {"partial": seg.text})
    send(wfile, {"done": True, "text": accumulated.strip(), "ms": int((time.monotonic() - t0) * 1000)})


def make_handler(model: WhisperModel):
    class Handler(socketserver.StreamRequestHandler):
        def handle(self) -> None:
            line = self.rfile.readline()
            if not line:
                return
            try:
                req = json.loads(line)
                cmd = req.get("cmd")
                if cmd == "ping":
                    send(self.wfile, {"ok": True})
                elif cmd == "transcribe":
                    transcribe_stream(model, req["wav"], self.wfile)
                else:
                    send(self.wfile, {"error": f"unknown cmd: {cmd!r}"})
            except Exception as e:
                try:
                    send(self.wfile, {"error": f"{type(e).__name__}: {e}"})
                except Exception:
                    pass
    return Handler


def cleanup_socket() -> None:
    try:
        SOCKET_PATH.unlink()
    except FileNotFoundError:
        pass


def main() -> int:
    cleanup_socket()
    model = load_model()
    server = socketserver.UnixStreamServer(str(SOCKET_PATH), make_handler(model))
    os.chmod(SOCKET_PATH, 0o600)

    def stop(signum, frame):
        # shutdown() blocks until serve_forever() returns; calling it directly
        # from the signal handler deadlocks. Dispatch to a thread.
        threading.Thread(target=server.shutdown, daemon=True).start()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    print(f"listening on {SOCKET_PATH}", flush=True)
    try:
        server.serve_forever()
    finally:
        cleanup_socket()
    return 0


if __name__ == "__main__":
    sys.exit(main())
