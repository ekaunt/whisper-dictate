# whisper-dictate

Local, push-to-talk voice dictation on Linux, powered by [faster-whisper](https://github.com/SYSTRAN/faster-whisper). Hold a hotkey, speak, release — the text appears in whatever window has focus. No cloud, no API keys.

- ~80 lines of Python, ~70 lines of bash
- Streaming transcription: words appear as the model decodes them
- Runs as a small systemd user service so model load (~3s) happens once at login, not per keypress
- CPU by default (~2GB RAM, ~1× realtime on a modern CPU); optional GPU mode via env var

## Requirements

- Linux with X11 (uses `xdotool` to type into the focused window)
- PipeWire or PulseAudio (uses `parec`)
- `sxhkd` or any other hotkey daemon
- `systemd` user instance
- Python 3.12 (faster-whisper wheels lag for cp314+) — `uv python install 3.12` if you don't have it
- These CLI tools on PATH: `uv`, `xdotool`, `parec`, `jq`, `nc` (openbsd-style), `notify-send`

## Install

```sh
git clone https://github.com/<you>/whisper-dictate
cd whisper-dictate
./install.sh         # CPU mode (default)
./install.sh --gpu   # NVIDIA CUDA mode (Linux + nvidia driver only)
```

The installer creates a uv venv at `~/.local/share/whisper-dictate/.venv`, downloads `Systran/faster-distil-whisper-large-v3` (~1.5 GB), installs `~/.local/bin/dictate`, installs a systemd user unit, and enables it.

## Hotkey setup

Add to `~/.config/sxhkd/sxhkdrc`:

```
super + z
    ~/.local/bin/dictate start
@super + z
    ~/.local/bin/dictate stop
```

Then `pkill -USR1 sxhkd` to reload. The `@` prefix means "fire on key release" — that's what makes it push-to-talk.

For other hotkey daemons / desktop environments, bind any key's press to `dictate start` and its release to `dictate stop`.

## Usage

Hold your hotkey, speak, release. You'll see:

- `🎤 Recording...` (stays visible while you talk)
- `✍️ Transcribing...` (instant on release)
- `✍️ <text so far>` updating as each segment decodes — the same text streams into your focused window
- `✓ <ms> — <text>` final confirmation

## GPU mode

By default the daemon runs on CPU. On a modern desktop CPU this is ~1× realtime for `distil-large-v3`, which is plenty for interactive dictation.

GPU mode is faster but holds a CUDA context resident, which on laptops prevents the dGPU from suspending (~5 W idle drain via NVIDIA runtime PM). Recommended only if you're plugged in or have a desktop.

To enable: `./install.sh --gpu` (installs `nvidia-cublas-cu12` + `nvidia-cudnn-cu12` pip wheels, sets `WHISPER_DICTATE_GPU=1` in the unit). To flip back: re-run `./install.sh` without `--gpu`.

**NVIDIA Blackwell (RTX 50-series) note:** `compute_type` must be `"float16"` — int8 variants hit `CUBLAS_STATUS_NOT_SUPPORTED`. The daemon already does the right thing automatically.

## Tweaking

Edit `~/.local/share/whisper-dictate/daemon.py`:

- **Model:** change `MODEL_ID`. Smaller (`distil-medium.en`) is faster and lighter; `large-v3` is more accurate but bigger.
- **Language:** drop `language="en"` from `model.transcribe()` to autodetect.
- **VAD:** `vad_filter=True` trims silence; set `False` if you want to capture everything verbatim.

Restart after editing: `systemctl --user restart whisper-dictate`.

## Why a daemon?

Cold model load is 3–8s. Without a resident daemon, every hotkey press would pay that cost. The daemon keeps the model in RAM and answers each `dictate stop` over a Unix socket in milliseconds.
