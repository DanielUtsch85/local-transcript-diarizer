from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class DiarizationSegment:
    speaker: str
    start: float
    end: float
    source_file: Path

    @property
    def duration_sec(self) -> float:
        return self.end - self.start


def _parse_segment(raw: dict[str, Any], base_dir: Path) -> DiarizationSegment:
    missing = {"speaker", "start", "end", "source_file"} - set(raw)
    if missing:
        raise ValueError(f"Segment is missing fields: {sorted(missing)}")
    speaker = str(raw["speaker"])
    start = float(raw["start"])
    end = float(raw["end"])
    if not speaker:
        raise ValueError("Segment speaker must not be empty")
    if start < 0 or end <= start:
        raise ValueError(f"Invalid segment timing: start={start}, end={end}")
    source_file = Path(str(raw["source_file"]))
    if not source_file.is_absolute():
        cwd_candidate = Path.cwd() / source_file
        source_file = cwd_candidate if cwd_candidate.exists() else base_dir / source_file
    return DiarizationSegment(speaker=speaker, start=start, end=end, source_file=source_file)


def load_segments(path: Path) -> list[DiarizationSegment]:
    base_dir = path.parent
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        rows = json.loads(text)
    if isinstance(rows, dict):
        rows = rows.get("segments", [])
    if not isinstance(rows, list):
        raise ValueError("Diarization input must be a JSON array, JSONL file, or object with 'segments'")
    return [_parse_segment(row, base_dir) for row in rows]


def filter_segments(
    segments: Iterable[DiarizationSegment],
    speaker: str,
    min_duration_sec: float = 2.0,
    max_duration_sec: float = 15.0,
) -> list[DiarizationSegment]:
    return [
        segment
        for segment in segments
        if segment.speaker == speaker and min_duration_sec <= segment.duration_sec <= max_duration_sec
    ]


def segment_filename(index: int, segment: DiarizationSegment) -> str:
    return f"seg_{index:06d}_{segment.start:08.3f}_{segment.end:08.3f}.wav"
