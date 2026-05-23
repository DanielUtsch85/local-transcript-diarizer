# Local Transcript Diarizer

A local macOS-first Streamlit app for transcribing audio files, detecting speakers locally, exporting clean conversation transcripts, and building consent-based target-speaker voice-reference packs with optional synthetic audio backends.

## What Runs Locally

- Transcription: WhisperX / faster-whisper
- Speaker diarization: local `diarize`, no account required
- Voice activity detection: pyannote VAD for transcription; Silero VAD for voice-segment scoring when available
- Export: DOCX, HTML, feedback CSV, diarization JSON, and diarization JSONL
- Voice pipeline: target-speaker curation, manual reference-pack review, optional reference denoise, mock/XTTS synthesis, metadata logging
- Run monitoring: bounded CPU/RAM history, live progress reporting, and partial-result recovery hints when WhisperX fails after writing JSON
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
7. Download DOCX, HTML, feedback CSV, and diarization JSON/JSONL outputs.
8. Optionally open the separate Voice-Pipeline page for target-speaker curation, reference review, and synthesis.

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

The app keeps the raw local diarization time ranges separately from the rendered transcript segments. Diarization JSON/JSONL exports and the Voice-Pipeline page use these raw speaker time ranges when available, because they are better suited for later segment curation.

## Diarization Export

After a run with a known source audio file, the export panel offers:

- diarization JSON
- diarization JSONL

Both formats use the voice-pipeline schema:

```json
{
  "speaker": "SPEAKER_00",
  "start": 12.34,
  "end": 19.82,
  "source_file": "data/uploads/interview.mp3"
}
```

These exports are useful when transcription and speaker detection are done in one session, but voice-reference curation happens later.

## Feedback CSV

Each completed run creates a feedback CSV with:

- runtime and audio-duration ratio
- run timestamp, audio filename, audio file size, and the executed WhisperX command
- selected settings
- transcript segment and speaker counts
- speaker duration balance
- Python, platform, and relevant package versions for troubleshooting
- CPU, RAM, and memory-pressure peaks/averages
- quality flags for suspicious outputs
- optional short text samples, disabled by default for privacy

This helps evaluate whether a run looks technically and structurally plausible without manually inspecting the full transcript.

Longer local runs keep resource monitoring bounded to the most recent hour of one-second samples. If WhisperX fails after writing a partial JSON result, the app surfaces that file name so the existing JSON loader can be used for recovery instead of losing the whole run context.

## Voice-Pipeline Page

The sidebar has two app areas:

- `Transkription`
- `Voice-Pipeline`

The Voice-Pipeline page can use the current audio and speaker segments from the transcription run, or it can accept an audio file plus diarization JSON/JSONL manually. It supports:

- target-speaker selection
- audio preparation to mono WAV at 16 kHz or 24 kHz
- target-speaker segment extraction
- quality scoring with RMS, peak, clipping, speech ratio, silence ratio, simple SNR, and overlap metrics
- overlap rejection for segments that collide with another speaker
- accepted/rejected segment folders
- automatic reference-pack building
- manual reference-pack curation with per-clip playback
- optional ffmpeg-based reference-clip denoise before reference-pack building
- optional mock, XTTS, or OpenVoice synthesis
- synthetic sidecar metadata and synthesis JSONL logging
- Voice-Pipeline feedback CSV export
- generated-sample review CSV export with human quality ratings

## Privacy and Consent

Audio files, generated transcripts, feedback CSVs, local runs, model caches, and virtual environments are not meant to be committed. The `.gitignore` excludes local uploads and run artifacts by default.

Before publishing or sharing logs, check that they do not contain private transcript excerpts. Feedback CSVs include local diagnostic metadata such as the uploaded audio filename and executed command. Files under `data/` and files downloaded to your local `Downloads` folder are not published to GitHub unless you explicitly add and commit them.

Voice synthesis and voice-cloning workflows must only be used with explicit consent from the target speaker. Generated audio must be disclosed as synthetic.

---

# Voice Pipeline

`voice_pipeline` is the modular pipeline behind the Voice-Pipeline app page and is also exposed as a CLI. It turns existing diarization output into clean target-speaker reference clips and synthetic voice samples. It assumes speaker diarization and initial speaker localization have already happened.

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
- Silero VAD via `silero-vad` when available, with internal 16 kHz resampling for 24 kHz clips and an energy-based fallback
- Coqui `TTS` for XTTS-v2 synthesis, installed in a separate environment
- OpenVoice V2 plus checkpoints for OpenVoice synthesis; the adapter boundary exists, but checkpoint-specific synthesis wiring is not configured in this repo yet

## Isolated XTTS Environment

Do not install Coqui `TTS` into the main `.venv`. Current WhisperX/pyannote dependencies need modern `numpy` and `pandas`, while `TTS 0.22.0` pulls older versions. Keep XTTS isolated:

```bash
bash scripts/setup_xtts_env.sh
```

The app looks for `.venv-xtts/bin/python` by default. If you keep XTTS somewhere else, set:

```bash
export XTTS_PYTHON=/absolute/path/to/xtts/python
```

XTTS may download model weights on first use. Review the XTTS-v2 model license and deployment terms before using generated audio outside local experiments.

## OpenVoice Status

The OpenVoice backend is currently a guarded adapter, not a complete local OpenVoice setup. If selected without a separate OpenVoice V2 installation and configured checkpoints, the app fails fast with a clear error instead of silently producing invalid output.

Use XTTS or the mock backend for German end-to-end tests. OpenVoice V2 is wired through an isolated runner for the MeloTTS-supported synthesis languages `en`, `es`, `fr`, `zh`, `jp`, and `kr`. German OpenVoice output requires an external German base TTS path and is intentionally rejected by the current adapter.

Set up the isolated OpenVoice environment with:

```bash
bash scripts/setup_openvoice_env.sh
```

Then place OpenVoice V2 checkpoints under:

```text
data/models/openvoice/checkpoints_v2/
  converter/config.json
  converter/checkpoint.pth
  base_speakers/ses/*.pth
```

If your checkpoints live elsewhere, set:

```bash
export OPENVOICE_PYTHON=/absolute/path/to/openvoice/python
export OPENVOICE_CHECKPOINT_DIR=/absolute/path/to/checkpoints_v2
```

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

In the Streamlit UI, the automatic reference pack can be refined manually:

1. Open `Reference-Pack manuell kuratieren`.
2. Listen to accepted clips.
3. Keep only clean, unmistakable target-speaker clips.
4. Click `Auswahl als Reference-Pack verwenden`.
5. Run synthesis with the current reference pack without rerunning the full pipeline.

The UI can also create denoised reference copies before building a pack. This uses ffmpeg `afftdn`, keeps original clips untouched, and records denoise settings in feedback metadata.

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

The OpenVoice command currently requires a separate OpenVoice V2 installation and `checkpoints_v2`. Without that setup it exits with an explanatory error. In this app, OpenVoice language choices are limited to the MeloTTS-supported languages.

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
  denoised_segments/
  rejected_segments/
  voice_refs/
  generated_samples/
  manifests/
    raw_segments.jsonl
    segments.jsonl
    accepted.jsonl
    rejected.jsonl
    reference_source.jsonl
    manual_reference_selection.jsonl
    synthesis_runs.jsonl
    voice_feedback.csv
  report.md
```

Each generated sample also gets a sidecar file such as:

```text
sample_xtts_001.wav.synthetic.json
```

Manual sample reviews can export files such as:

```text
SPEAKER_02-sample_xtts_001-quality-review.csv
```

## Configuration

Defaults live in `configs/default.yaml`. You can pass another file to commands that accept `--config`; unspecified keys fall back to built-in defaults.

Important quality defaults include:

- `min_duration_sec: 2.0`
- `max_duration_sec: 15.0`
- `min_speech_ratio: 0.75`
- `max_clipping_ratio: 0.001`
- `max_overlap_sec: 0.0`

The default overlap policy rejects target-speaker clips that overlap another speaker in the diarization export.

The UI exposes additional voice-reference controls:

- sample rate: 24 kHz for XTTS-oriented quality, 16 kHz for smaller files
- target reference-pack duration
- preferred reference-clip duration range
- minimum speech ratio
- clipping, RMS, and peak thresholds
- optional reference denoise strength

In practical tests, smaller manually selected packs can outperform large automatic packs. Start with 3-6 clean clips and compare against the automatic pack.

## Voice Feedback CSV

Each Voice-Pipeline run writes `manifests/voice_feedback.csv`. It captures:

- selected speaker and source files
- pipeline settings
- raw, scored, accepted, and rejected segment counts
- acceptance rate
- accepted duration totals and distribution
- overlap counts and rejected-overlap counts
- reject-reason counts
- reference-pack target and actual duration
- reference-pack selection mode and denoise settings when available
- synthesis backend and output path when synthesis was run

Use this file to compare whether pipeline changes improve curation quality over repeated runs.

Generated samples can also be reviewed from the UI. The review export combines technical run metadata with human ratings for overall quality, speaker similarity, target-speaker accuracy, intelligibility, naturalness, artifact level, optimization needs, and notes.

## Troubleshooting

- `ffmpeg is required`: install `ffmpeg` and make sure it is on `PATH`.
- `Install faster-whisper`: transcription is optional; run scoring without `--transcribe` or install the package.
- `XTTS requires a separate Python environment`: run `bash scripts/setup_xtts_env.sh` or set `XTTS_PYTHON`.
- `OpenVoice requires a separate Python environment`: run `bash scripts/setup_openvoice_env.sh` or set `OPENVOICE_PYTHON`.
- `OpenVoice V2 checkpoints are not configured`: set `OPENVOICE_CHECKPOINT_DIR` to a `checkpoints_v2` folder with converter and base-speaker embeddings.
- `OpenVoice V2 with MeloTTS does not natively support this synthesis language`: use `en`, `es`, `fr`, `zh`, `jp`, or `kr`, or use XTTS for German.
- `synthesis requires --consent-confirmed`: confirm explicit target-speaker consent and rerun with the required flag.
- `accepted_count` is unexpectedly `0`: check `voice_feedback.csv` reject reasons. If `low_speech_ratio` dominates, confirm Silero VAD is installed and that the clips are readable WAV files. If `overlaps_other_speaker` dominates, try disabling overlap rejection for diagnosis, but keep it enabled for high-quality reference packs.
- Synthetic voice is understandable but not similar enough: try a smaller manually curated reference pack, then A/B test with reference denoise. If similarity remains weak, test another backend instead of only increasing reference duration.
- Denoise sounds metallic: lower denoise strength or disable it. The original clean segments remain available.
