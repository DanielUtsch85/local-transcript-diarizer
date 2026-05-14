#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

missing=()
for command in ffmpeg; do
  if ! command -v "$command" >/dev/null 2>&1; then
    missing+=("$command")
  fi
done

if [ "${#missing[@]}" -gt 0 ]; then
  echo "Missing required system tools for OpenVoice: ${missing[*]}" >&2
  echo "Install them on macOS with:" >&2
  echo "  brew install ffmpeg" >&2
  exit 1
fi

python3.11 -m venv .venv-openvoice
source .venv-openvoice/bin/activate
python -m pip install --upgrade pip "setuptools<81" wheel

python -m pip install "torch>=2.2,<2.4" "torchaudio>=2.2,<2.4"
python -m pip install \
  "setuptools<81" \
  "numpy>=1.26,<2" \
  "librosa==0.9.1" \
  "soundfile>=0.12" \
  "pydub==0.25.1" \
  "eng_to_ipa==0.0.2" \
  "inflect==7.0.0" \
  "unidecode==1.3.7" \
  "pypinyin==0.50.0" \
  "cn2an==0.5.22" \
  "jieba==0.42.1" \
  "langid==1.1.6" \
  "wavmark==0.0.3"
python -m pip install --no-deps git+https://github.com/myshell-ai/OpenVoice.git
python -m pip install "git+https://github.com/myshell-ai/MeloTTS.git"
python -m unidic download || true

mkdir -p data/models/openvoice

cat <<'MSG'
OpenVoice environment ready.

Python:
  .venv-openvoice/bin/python

Next required step:
  Download OpenVoice V2 checkpoints and place them at:
  data/models/openvoice/checkpoints_v2/

Expected files include:
  data/models/openvoice/checkpoints_v2/converter/config.json
  data/models/openvoice/checkpoints_v2/converter/checkpoint.pth
  data/models/openvoice/checkpoints_v2/base_speakers/ses/*.pth

If you use another location:
  export OPENVOICE_PYTHON="$PWD/.venv-openvoice/bin/python"
  export OPENVOICE_CHECKPOINT_DIR="/absolute/path/to/checkpoints_v2"
MSG
