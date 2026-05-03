from __future__ import annotations

import shutil
import wave
from pathlib import Path
from typing import Any

from .logging_utils import read_jsonl, write_json


def _rank(row: dict[str, Any], preferred_min: float, preferred_max: float) -> tuple[float, float, float]:
    duration = float(row.get("duration_sec") or 0)
    duration_penalty = 0.0 if preferred_min <= duration <= preferred_max else min(abs(duration - preferred_min), abs(duration - preferred_max))
    speech = float(row.get("speech_ratio") or 0)
    clipping = float(row.get("clipping_ratio") or 1)
    rms = float(row.get("rms_db") or -120)
    rms_penalty = abs(rms + 22)
    return (speech - clipping * 10 - duration_penalty * 0.1 - rms_penalty * 0.01, duration, -clipping)


def build_reference_pack(
    clean_segments_dir: Path,
    manifest_path: Path,
    out_dir: Path,
    target_total_sec: float = 60,
    preferred_min_duration_sec: float = 4,
    preferred_max_duration_sec: float = 12,
) -> dict[str, Any]:
    rows = [row for row in read_jsonl(manifest_path) if row.get("accepted", True)]
    rows.sort(key=lambda row: _rank(row, preferred_min_duration_sec, preferred_max_duration_sec), reverse=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    refs: list[dict[str, Any]] = []
    total = 0.0
    for row in rows:
        if total >= target_total_sec and refs:
            break
        source = Path(str(row["file"]))
        if not source.is_absolute():
            source = clean_segments_dir / source.name
        if not source.exists():
            source = clean_segments_dir / Path(str(row["file"])).name
        if not source.exists():
            continue
        dest = out_dir / f"ref_{len(refs) + 1:03d}.wav"
        shutil.copy2(source, dest)
        duration = float(row.get("duration_sec") or 0.0)
        total += duration
        refs.append({"file": str(dest), "source": str(source), "duration_sec": duration})

    combined = out_dir / "combined_ref.wav"
    if refs:
        _concat_wavs([Path(ref["file"]) for ref in refs], combined)
    metadata = {
        "speaker": out_dir.parent.name,
        "target_total_sec": target_total_sec,
        "actual_total_sec": round(total, 3),
        "refs": refs,
        "combined_ref": str(combined) if refs else None,
    }
    write_json(out_dir / "reference_pack.json", metadata)
    return metadata


def _concat_wavs(sources: list[Path], output: Path) -> None:
    params = None
    frames: list[bytes] = []
    for source in sources:
        with wave.open(str(source), "rb") as handle:
            current = handle.getparams()
            if params is None:
                params = current
            elif current[:3] != params[:3]:
                continue
            frames.append(handle.readframes(handle.getnframes()))
    if params is None:
        return
    with wave.open(str(output), "wb") as handle:
        handle.setparams(params)
        for frame_bytes in frames:
            handle.writeframes(frame_bytes)
