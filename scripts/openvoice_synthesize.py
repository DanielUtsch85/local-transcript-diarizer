from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path


LANGUAGE_TO_MELO = {
    "en": "EN_NEWEST",
    "es": "ES",
    "fr": "FR",
    "zh": "ZH",
    "jp": "JP",
    "kr": "KR",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run OpenVoice V2 synthesis in an isolated Python environment.")
    parser.add_argument("--request", required=True, type=Path)
    args = parser.parse_args()

    payload = json.loads(args.request.read_text(encoding="utf-8"))
    output = Path(payload["output_wav"])
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(payload["checkpoint_dir"])
    language = payload["language"]
    melo_language = LANGUAGE_TO_MELO[language]

    import torch  # type: ignore
    from melo.api import TTS  # type: ignore
    from openvoice.api import ToneColorConverter  # type: ignore
    import wavmark  # type: ignore

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    _log(f"device={device}")
    _log("loading tone color converter")
    wavmark.load_model = lambda: _NoWatermarkModel()
    converter = ToneColorConverter(str(checkpoint_dir / "converter" / "config.json"), device=device)
    converter.watermark_model = None
    converter.load_ckpt(str(checkpoint_dir / "converter" / "checkpoint.pth"))

    _log(f"loading MeloTTS language={melo_language}")
    tts = TTS(language=melo_language, device=device)
    speaker_key = _choose_speaker(tts.hps.data.spk2id)
    source_se = _source_se_path(checkpoint_dir, speaker_key)
    speaker_id = tts.hps.data.spk2id[speaker_key]
    _log(f"extracting target speaker embedding from {payload['reference_wav']}")
    target_se = converter.extract_se([str(payload["reference_wav"])])

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
        source_wav = Path(handle.name)
    try:
        _log("generating base TTS audio")
        tts.tts_to_file(payload["text"], speaker_id, str(source_wav), speed=1.0)
        _log("converting tone color")
        converter.convert(
            audio_src_path=str(source_wav),
            src_se=torch.load(source_se, map_location=device),
            tgt_se=target_se,
            output_path=str(output),
        )
        _log(f"done output={output}")
    finally:
        source_wav.unlink(missing_ok=True)
    return 0


def _choose_speaker(speaker_ids: dict[str, int]) -> str:
    preferred = ["EN-Newest", "EN-US", "EN-BR", "ES", "FR", "ZH", "JP", "KR"]
    for key in preferred:
        if key in speaker_ids:
            return key
    return next(iter(speaker_ids))


class _NoWatermarkModel:
    def to(self, device: str):
        return None


def _log(message: str) -> None:
    print(f"[openvoice] {message}", file=sys.stderr, flush=True)


def _source_se_path(checkpoint_dir: Path, speaker_key: str) -> Path:
    ses_dir = checkpoint_dir / "base_speakers" / "ses"
    candidates = [
        ses_dir / f"{speaker_key}.pth",
        ses_dir / f"{speaker_key.lower()}.pth",
        ses_dir / f"{speaker_key.replace('_', '-').lower()}.pth",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    available = ", ".join(path.name for path in sorted(ses_dir.glob("*.pth")))
    raise RuntimeError(f"No source speaker embedding found for {speaker_key}. Available: {available}")


if __name__ == "__main__":
    raise SystemExit(main())
