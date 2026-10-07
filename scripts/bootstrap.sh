#!/usr/bin/env bash
# Sets up everything Vaani Studio needs on this Mac:
#   - command-line tools (ffmpeg, rubberband, uv, xcodegen) via Homebrew
#   - the Python engine (Python 3.11 venv with the ML libraries)
#   - the isolated voice runtime (Chatterbox, pinned torch/transformers)
# Models are not downloaded here; the app fetches them on first use or from the Models screen.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

step() { printf '\n\033[1m==> %s\033[0m\n' "$1"; }

if [[ "$(uname -m)" != "arm64" ]]; then
  echo "Vaani Studio needs an Apple Silicon Mac (M1 or newer)." >&2
  exit 1
fi

step "Command-line tools"
if ! command -v brew >/dev/null; then
  echo "Homebrew is required: https://brew.sh" >&2
  exit 1
fi
for tool in ffmpeg rubberband uv xcodegen; do
  if ! command -v "$tool" >/dev/null; then brew install "$tool"; else echo "$tool ✓"; fi
done

step "Python 3.11"
uv python install 3.11

step "Engine (this downloads PyTorch and friends, ~2 GB)"
cd "$ROOT/engine"
uv sync --python 3.11 --extra ml

step "Voice runtimes (voice cloning on Apple's GPU, voice conversion, Marathi speech)"
for runtime in mlxvoice voice; do
  (cd "$ROOT/engine/runtimes/$runtime" && uv sync --python 3.11)
done

step "Checking"
cd "$ROOT/engine"
.venv/bin/python -m pytest -q
.venv/bin/vaani models list

cat <<'EOF'

Done. Next:
  • Build the app:   scripts/build_app.sh
  • Or try the CLI:  engine/.venv/bin/vaani enhance "your file.wav"
  • For the best local Marathi models (IndicConformer, IndicTrans2), create a free
    Hugging Face token, accept the model terms, and add the token in Settings.
EOF
