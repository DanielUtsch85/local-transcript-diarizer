from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .logging_utils import append_jsonl, write_json


def validate_consent(consent_confirmed: bool) -> None:
    if not consent_confirmed:
        raise PermissionError("synthesis requires --consent-confirmed")


def synthesis_metadata(
    speaker: str,
    backend: str,
    text: str,
    language: str,
    reference_files: Sequence[Path],
    output_file: Path,
    consent_confirmed: bool,
) -> dict[str, Any]:
    validate_consent(consent_confirmed)
    return {
        "timestamp": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "speaker": speaker,
        "backend": backend,
        "text": text,
        "language": language,
        "reference_files": [str(path) for path in reference_files],
        "output_file": str(output_file),
        "synthetic": True,
        "consent_confirmed": True,
    }


def write_synthesis_log(log_path: Path, metadata: dict[str, Any], output_wav: Path) -> None:
    append_jsonl(log_path, metadata)
    write_json(output_wav.with_suffix(output_wav.suffix + ".synthetic.json"), metadata)
