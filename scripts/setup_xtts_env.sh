#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python3.11 -m venv .venv-xtts
source .venv-xtts/bin/activate
python -m pip install --upgrade pip
python -m pip install "TTS==0.22.0"

echo "XTTS environment ready: $PWD/.venv-xtts/bin/python"
echo "Use XTTS_PYTHON=$PWD/.venv-xtts/bin/python if you run the app from another working directory."
