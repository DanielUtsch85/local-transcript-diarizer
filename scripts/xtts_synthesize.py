from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run XTTS synthesis in an isolated Python environment.")
    parser.add_argument("--request", required=True, type=Path)
    args = parser.parse_args()

    payload = json.loads(args.request.read_text(encoding="utf-8"))
    output = Path(payload["output_wav"])
    output.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output.parent.parent / "work" / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(output.parent.parent / "work" / "xdg_cache"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

    from TTS.api import TTS  # type: ignore

    tts = TTS(payload["model_name"])
    tts.tts_to_file(
        text=payload["text"],
        speaker_wav=payload["reference_wavs"],
        language=payload["language"],
        file_path=str(output),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
