# Local Transcript Diarizer

A local macOS-first app for transcribing audio files, detecting speakers locally, renaming speaker labels, and exporting clean conversation transcripts as DOCX or HTML.

## What Runs Locally

- Transcription: WhisperX / faster-whisper
- Speaker diarization: local `diarize`, no account required
- Voice activity detection: pyannote VAD by default; Silero is optional when cached locally
- Export: DOCX und HTML
- No OpenAI API, no cloud transcription API

Speaker detection runs locally with `diarize`. No token, paid service, or cloud account is required.

## Requirements on macOS

1. Install Homebrew if needed.
2. Install system packages:

```bash
brew install ffmpeg python@3.11
```

3. Create and install into a virtual environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

## Run

```bash
source .venv/bin/activate
python scripts/check_setup.py
streamlit run app.py
```

If Python cannot find the local package, use the bundled launcher:

```bash
bash scripts/run_app.sh
```

Streamlit will open the local app in your browser.

## Usage

1. Upload an MP3, WAV, M4A, or MP4 file.
2. Choose model and language.
3. Keep memory mode enabled for longer files.
4. Optionally set min/max speaker counts.
5. Start transcription.
6. Rename speaker labels after processing.
7. Download DOCX, HTML, and feedback CSV outputs.

## Recommended Settings

For longer recordings on macOS, start with:

- Memory mode: on
- Whisper model: `medium` for a strong default, `small` for fast checks
- Batch size: `1`
- Chunk size: `20`
- Word alignment: on for more readable segmentation
- VAD method: `pyannote` for stable offline runs in this app
- Speaker detection: only enable it when speaker labels are needed

`small` is useful for quick checks, `medium` is currently the best default tradeoff, and `large-v3` is intended for final higher-quality runs. `tiny` and `base` are intentionally not part of the normal UI because early tests produced weak results for interview material.

## Local Speaker Detection

When speaker detection is enabled, the app:

1. Transcribes the audio with WhisperX.
2. Runs local diarization with `diarize`.
3. Assigns speaker labels back to transcript segments by timestamp overlap.

You can then rename labels like `SPEAKER_00` to real names before exporting.

## Feedback CSV

Each completed run creates a feedback CSV with:

- runtime and audio-duration ratio
- selected settings
- transcript segment and speaker counts
- speaker duration balance
- CPU, RAM, and memory-pressure peaks/averages
- quality flags for suspicious outputs
- optional short text samples, disabled by default for privacy

This helps evaluate whether a run looks technically and structurally plausible without manually inspecting the full transcript.

## Privacy

Audio files, generated transcripts, feedback CSVs, local runs, model caches, and virtual environments are not meant to be committed. The `.gitignore` excludes local uploads and run artifacts by default.

Before publishing or sharing logs, check that they do not contain private transcript excerpts.

---

# Voice Pipeline

`voice_pipeline` is a modular CLI pipeline for turning existing diarization output into clean target-speaker reference clips and synthetic voice samples. It assumes speaker diarization and initial speaker localization have already happened.

## Consent and Synthetic Media Disclosure

This tool must only be used with explicit consent from the target speaker.
All generated audio is synthetic and must be disclosed as such.
The pipeline writes JSON sidecar metadata for generated files.

Synthesis commands require `--consent-confirmed`. Without that flag, synthesis fails before any backend is called.

## Installation

Install `ffmpeg` first:

```bash
brew install ffmpeg
```

Then install the project:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Optional heavy dependencies are intentionally not required for tests:

- `faster-whisper` for transcription inside scoring
- Silero VAD via `silero-vad` when available, with an energy-based fallback
- Coqui `TTS` for XTTS-v2 synthesis
- OpenVoice V2 plus checkpoints for OpenVoice synthesis

## Input Format

JSON:

```json
[
  {
    "speaker": "SPEAKER_02",
    "start": 12.34,
    "end": 19.82,
    "source_file": "input/interview.mp3"
  }
]
```

JSONL is also supported:

```json
{"speaker":"SPEAKER_02","start":12.34,"end":19.82,"source_file":"input/interview.mp3"}
```

## Examples

Prepare audio:

```bash
voice-pipeline prepare-audio \
  --input data/input/interview.mp3 \
  --output data/work/interview.wav \
  --sample-rate 24000 \
  --mono
```

Segment curation only:

```bash
voice-pipeline extract-speaker \
  --segments data/input/diarization.json \
  --speaker SPEAKER_02 \
  --out data/output/SPEAKER_02/raw_segments

voice-pipeline score-segments \
  --input data/output/SPEAKER_02/raw_segments \
  --speaker SPEAKER_02 \
  --out data/output/SPEAKER_02/manifests/segments.jsonl

voice-pipeline curate \
  --manifest data/output/SPEAKER_02/manifests/segments.jsonl \
  --accepted-dir data/output/SPEAKER_02/clean_segments \
  --rejected-dir data/output/SPEAKER_02/rejected_segments
```

Build reference clips:

```bash
voice-pipeline build-reference-pack \
  --clean-segments data/output/SPEAKER_02/clean_segments \
  --manifest data/output/SPEAKER_02/manifests/accepted.jsonl \
  --out data/output/SPEAKER_02/voice_refs \
  --target-total-sec 60
```

XTTS synthesis:

```bash
voice-pipeline synthesize \
  --backend xtts \
  --speaker-dir data/output/SPEAKER_02 \
  --text "Hallo, dies ist ein synthetisch erzeugter Testsatz." \
  --language de \
  --out data/output/SPEAKER_02/generated_samples/sample_xtts_001.wav \
  --consent-confirmed
```

OpenVoice synthesis:

```bash
voice-pipeline synthesize \
  --backend openvoice \
  --speaker-dir data/output/SPEAKER_02 \
  --text "Hallo, dies ist ein synthetisch erzeugter Testsatz." \
  --language de \
  --out data/output/SPEAKER_02/generated_samples/sample_openvoice_001.wav \
  --consent-confirmed
```

End-to-end with the lightweight mock backend:

```bash
voice-pipeline run-all \
  --audio data/input/interview.mp3 \
  --segments data/input/diarization.json \
  --speaker SPEAKER_02 \
  --text "Dies ist ein synthetischer Test." \
  --backend mock \
  --language de \
  --consent-confirmed \
  --out data/output/SPEAKER_02
```

## Output Structure

```text
data/output/SPEAKER_02/
  raw_segments/
  clean_segments/
  rejected_segments/
  voice_refs/
  generated_samples/
  manifests/
    raw_segments.jsonl
    segments.jsonl
    accepted.jsonl
    rejected.jsonl
    synthesis_runs.jsonl
  report.md
```

Each generated sample also gets a sidecar file such as:

```text
sample_xtts_001.wav.synthetic.json
```

## Configuration

Defaults live in `configs/default.yaml`. You can pass another file to commands that accept `--config`; unspecified keys fall back to built-in defaults.

## Troubleshooting

- `ffmpeg is required`: install `ffmpeg` and make sure it is on `PATH`.
- `Install faster-whisper`: transcription is optional; run scoring without `--transcribe` or install the package.
- `Install coqui-tts/TTS`: XTTS-v2 is isolated behind the adapter and only needed for real XTTS synthesis.
- `OpenVoice V2 ... checkpoints`: OpenVoice requires a separate repository/checkpoint setup; the adapter boundary is present but checkpoint-specific wiring must be configured.
- `synthesis requires --consent-confirmed`: confirm explicit target-speaker consent and rerun with the required flag.
