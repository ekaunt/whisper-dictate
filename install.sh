#!/usr/bin/env bash
# whisper-dictate installer.
#   ./install.sh         # CPU (default; safe everywhere)
#   ./install.sh --gpu   # NVIDIA CUDA; faster but holds the dGPU awake
set -euo pipefail

GPU=0
for arg in "$@"; do
    case "$arg" in
        --gpu) GPU=1 ;;
        --cpu) GPU=0 ;;
        -h|--help) sed -n '2,5p' "$0"; exit 0 ;;
        *) echo "unknown flag: $arg" >&2; exit 2 ;;
    esac
done

REPO_DIR=$(cd "$(dirname "$0")" && pwd)
SHARE_DIR=$HOME/.local/share/whisper-dictate
BIN_DIR=$HOME/.local/bin
UNIT_DIR=$HOME/.config/systemd/user
VENV_DIR=$SHARE_DIR/.venv

need() { command -v "$1" >/dev/null || { echo "missing required dependency: $1" >&2; exit 1; }; }

echo "==> Checking dependencies"
for cmd in uv xdotool parec jq nc notify-send systemctl; do need "$cmd"; done

echo "==> Locating Python 3.12 (faster-whisper wheels lag for cp314+)"
PY312=$(uv python find 3.12 2>/dev/null || command -v python3.12 || true)
if [[ -z "$PY312" ]]; then
    echo "Python 3.12 not found. Install with: uv python install 3.12" >&2
    exit 1
fi
echo "    using $PY312"

mkdir -p "$SHARE_DIR" "$BIN_DIR" "$UNIT_DIR"

echo "==> Creating venv at $VENV_DIR"
uv venv --python "$PY312" "$VENV_DIR"
VENV_PY=$VENV_DIR/bin/python

echo "==> Installing faster-whisper"
uv pip install --python "$VENV_PY" faster-whisper

if [[ "$GPU" == "1" ]]; then
    echo "==> Installing CUDA pip wheels (cuBLAS + cuDNN)"
    uv pip install --python "$VENV_PY" nvidia-cublas-cu12 nvidia-cudnn-cu12
fi

echo "==> Pre-downloading Whisper model (~1.5 GB on first run)"
"$VENV_PY" -c 'from faster_whisper import WhisperModel; WhisperModel("Systran/faster-distil-whisper-large-v3", device="cpu", compute_type="int8")' >/dev/null

echo "==> Installing files"
install -m 644 "$REPO_DIR/daemon.py"   "$SHARE_DIR/daemon.py"
install -m 755 "$REPO_DIR/bin/dictate" "$BIN_DIR/dictate"

echo "==> Writing systemd user unit"
{
    cat <<EOF
[Unit]
Description=Local Whisper dictation daemon
After=graphical-session.target

[Service]
ExecStart=%h/.local/share/whisper-dictate/.venv/bin/python %h/.local/share/whisper-dictate/daemon.py
Restart=on-failure
RestartSec=3
Environment=HF_HOME=%h/.cache/huggingface
EOF
    if [[ "$GPU" == "1" ]]; then
        cat <<EOF
Environment=WHISPER_DICTATE_GPU=1
Environment=LD_LIBRARY_PATH=%h/.local/share/whisper-dictate/.venv/lib/python3.12/site-packages/nvidia/cublas/lib:%h/.local/share/whisper-dictate/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib
EOF
    fi
    cat <<EOF

[Install]
WantedBy=default.target
EOF
} > "$UNIT_DIR/whisper-dictate.service"

echo "==> Enabling and starting service"
systemctl --user daemon-reload
systemctl --user enable --now whisper-dictate

cat <<EOF

✓ Installed (mode: $( [[ $GPU == 1 ]] && echo GPU || echo CPU )).

Next: bind a hotkey. For sxhkd, add to ~/.config/sxhkd/sxhkdrc:

    super + z
        ~/.local/bin/dictate start
    @super + z
        ~/.local/bin/dictate stop

Then: pkill -USR1 sxhkd

Hold the hotkey, speak, release — the text types into your focused window.
EOF
