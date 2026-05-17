#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
if [[ ! -f .venv/bin/activate ]]; then
  echo "Virtual environment missing. Create it first with: python -m venv .venv" >&2
  exit 1
fi

source .venv/bin/activate
export PYTHONNOUSERSITE=1
export STREAMLIT_SERVER_FILE_WATCHER_TYPE=none
export STREAMLIT_SERVER_RUN_ON_SAVE=false
export STREAMLIT_SERVER_MAX_UPLOAD_SIZE=1000
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
export STREAMLIT_GLOBAL_DEVELOPMENT_MODE=false
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

PORT="${STREAMLIT_PORT:-8501}"
HOST="${STREAMLIT_HOST:-127.0.0.1}"
URL="http://${HOST}:${PORT}"
OPEN_BROWSER="${STREAMLIT_OPEN_BROWSER:-1}"
LOG_DIR="${STREAMLIT_LOG_DIR:-data/logs}"
LOG_FILE="${STREAMLIT_LOG_FILE:-${LOG_DIR}/streamlit.log}"
PID_FILE="${STREAMLIT_PID_FILE:-${LOG_DIR}/streamlit.pid}"
mkdir -p "${LOG_DIR}"

stop_app() {
  local stopped=0
  if [[ -f "${PID_FILE}" ]]; then
    local pid
    pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
    if [[ "${pid}" =~ ^[0-9]+$ ]] && kill -0 "${pid}" >/dev/null 2>&1; then
      echo "Stopping app process ${pid}..."
      kill "${pid}" >/dev/null 2>&1 || true
      for _ in {1..20}; do
        if ! kill -0 "${pid}" >/dev/null 2>&1; then
          break
        fi
        sleep 0.2
      done
      if kill -0 "${pid}" >/dev/null 2>&1; then
        echo "Process ${pid} did not stop cleanly; forcing shutdown."
        kill -9 "${pid}" >/dev/null 2>&1 || true
      fi
      stopped=1
    fi
    rm -f "${PID_FILE}"
  fi

  if command -v lsof >/dev/null 2>&1; then
    local port_pids
    port_pids="$(lsof -tiTCP:"${PORT}" -sTCP:LISTEN 2>/dev/null || true)"
    if [[ -n "${port_pids}" ]]; then
      echo "Stopping process listening on ${URL}: ${port_pids}"
      kill ${port_pids} >/dev/null 2>&1 || true
      stopped=1
    fi
  fi

  if [[ "${stopped}" == "1" ]]; then
    echo "App stopped."
  else
    echo "No running app found for ${URL}."
  fi
}

case "${1:-}" in
  stop)
    stop_app
    exit 0
    ;;
  restart)
    stop_app
    shift
    ;;
esac

echo "Starting local transcription app..."

python -c "from importlib.metadata import version; v=version('streamlit'); parts=tuple(int(p) for p in v.split('.')[:2]); print(f'Streamlit {v} detected.'); print('Warning: project expects streamlit>=1.40,<1.50. Run .venv/bin/python -m pip install -r requirements.txt if startup is slow.' if parts >= (1, 50) else '')"

if command -v curl >/dev/null 2>&1 && curl -fsS "${URL}/_stcore/health" >/dev/null 2>&1; then
  echo "App is already running: ${URL}"
  echo "Stop it with: bash scripts/run_app.sh stop"
  echo "Restart it with: bash scripts/run_app.sh restart"
  if [[ "${OPEN_BROWSER}" != "0" ]] && command -v open >/dev/null 2>&1; then
    open "${URL}?reload=$(date +%s)"
  fi
  exit 0
fi

python -m streamlit run app.py \
  --server.headless true \
  --server.address "${HOST}" \
  --server.port "${PORT}" \
  "$@" >"${LOG_FILE}" 2>&1 &
APP_PID=$!
echo "${APP_PID}" >"${PID_FILE}"

cleanup() {
  kill "${APP_PID}" >/dev/null 2>&1 || true
  rm -f "${PID_FILE}"
}
trap cleanup INT TERM EXIT

for _ in {1..60}; do
  if command -v curl >/dev/null 2>&1 && curl -fsS "${URL}/_stcore/health" >/dev/null 2>&1; then
    echo "App is ready: ${URL}"
    echo "Streamlit log: ${LOG_FILE}"
    if [[ "${OPEN_BROWSER}" != "0" ]] && command -v open >/dev/null 2>&1; then
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
echo "Last Streamlit log lines from ${LOG_FILE}:" >&2
tail -n 80 "${LOG_FILE}" >&2 || true
exit 1
