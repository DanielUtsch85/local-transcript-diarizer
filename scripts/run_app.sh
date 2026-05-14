#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONNOUSERSITE=1
export STREAMLIT_SERVER_FILE_WATCHER_TYPE=none
export STREAMLIT_SERVER_MAX_UPLOAD_SIZE=1000
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

echo "Starting local transcription app..."
exec streamlit run app.py "$@"
