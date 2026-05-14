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

PORT="${STREAMLIT_PORT:-8501}"
HOST="${STREAMLIT_HOST:-localhost}"
URL="http://${HOST}:${PORT}"

if command -v curl >/dev/null 2>&1 && curl -fsS "${URL}/_stcore/health" >/dev/null 2>&1; then
  echo "App is already running: ${URL}"
  if command -v open >/dev/null 2>&1; then
    open "${URL}?reload=$(date +%s)"
  fi
  exit 0
fi

streamlit run app.py \
  --server.headless true \
  --server.port "${PORT}" \
  "$@" &
APP_PID=$!

cleanup() {
  kill "${APP_PID}" >/dev/null 2>&1 || true
}
trap cleanup INT TERM EXIT

for _ in {1..60}; do
  if command -v curl >/dev/null 2>&1 && curl -fsS "${URL}/_stcore/health" >/dev/null 2>&1; then
    echo "App is ready: ${URL}"
    if command -v open >/dev/null 2>&1; then
      open "${URL}?reload=$(date +%s)"
    fi
    wait "${APP_PID}"
    exit $?
  fi

  if ! kill -0 "${APP_PID}" >/dev/null 2>&1; then
    wait "${APP_PID}"
    exit $?
  fi
  sleep 1
done

echo "Streamlit did not become ready at ${URL} within 60 seconds." >&2
exit 1
