# Local Transcript Diarizer

A local macOS-first app for transcribing audio files, detecting speakers locally, renaming speaker labels, and exporting clean conversation transcripts as DOCX or HTML.

## What Runs Locally

- Transcription: WhisperX / faster-whisper
- Speaker diarization: local `diarize`, no account required
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
