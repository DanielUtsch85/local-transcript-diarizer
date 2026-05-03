from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from statistics import median


@dataclass(frozen=True)
class ProcessingEstimate:
    seconds: float | None
    source: str
    samples: int = 0
    ratio: float | None = None


def estimate_processing_seconds(
    *,
    audio_seconds: float | None,
    model: str,
    speaker_backend: str,
    memory_mode: bool,
    no_align: bool,
    batch_size: int | None = None,
    chunk_size: int | None = None,
    threads: int | None = None,
    vad_method: str | None = None,
    history_dir: Path,
) -> ProcessingEstimate:
    if audio_seconds is None:
        return ProcessingEstimate(seconds=None, source="keine Audiodauer")

    candidates = load_feedback_runs(history_dir)
    exact = [
        run
        for run in candidates
        if run.get("model") == model
        and run.get("speaker_backend") == speaker_backend
        and run.get("memory_mode") == str(memory_mode)
        and run.get("no_align") == str(no_align)
        and _matches_optional(run, "batch_size", batch_size)
        and _matches_optional(run, "chunk_size", chunk_size)
        and _matches_optional(run, "threads", threads)
        and _matches_optional_text(run, "vad_method", vad_method)
    ]
    aligned = [
        run
        for run in candidates
        if run.get("model") == model
        and run.get("speaker_backend") == speaker_backend
        and run.get("memory_mode") == str(memory_mode)
        and run.get("no_align") == str(no_align)
    ]
    similar = [
        run
        for run in candidates
        if run.get("model") == model and run.get("speaker_backend") == speaker_backend
    ]

    for label, runs in [
        ("exakt gleiche lokale Einstellungen", exact),
        ("gleiches Modell/Backend/Alignment", aligned),
        ("Modell/Backend-Historie", similar),
    ]:
        ratios = [_runtime_ratio(run) for run in runs]
        ratios = [ratio for ratio in ratios if ratio is not None]
        if ratios:
            ratio = median(ratios)
            return ProcessingEstimate(
                seconds=max(60.0, audio_seconds * ratio),
                source=label,
                samples=len(ratios),
                ratio=ratio,
            )

    ratio = default_ratio(model=model, speaker_backend=speaker_backend, memory_mode=memory_mode, no_align=no_align)
    return ProcessingEstimate(
        seconds=max(60.0, audio_seconds * ratio),
        source="Standardwert",
        samples=0,
        ratio=ratio,
    )


def load_feedback_runs(history_dir: Path) -> list[dict[str, str]]:
    runs = []
    for path in sorted(history_dir.glob("*/feedback.csv")):
        run = parse_feedback_csv(path)
        if run:
            runs.append(run)
    return runs


def parse_feedback_csv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            category = row.get("category", "")
            metric = row.get("metric", "")
            value = row.get("value", "")
            if category == "audio" and metric == "duration":
                values["audio_duration"] = value
            elif category == "timing" and metric == "processing_time":
                values["processing_time"] = value
            elif category == "run" and metric == "source_name":
                values["source_name"] = value
            elif category == "transcript" and metric in {
                "segment_count",
                "speaker_count",
                "unknown_speaker_segments",
                "word_count",
                "character_count",
                "avg_segment_duration",
                "max_segment_duration",
            }:
                values[metric] = value
            elif category == "quality":
                values[f"quality_{metric}"] = value
            elif category == "resources" and metric.startswith(("peak_", "avg_")):
                values[metric] = value
            elif category == "settings":
                values[metric] = value
    return values


def default_ratio(*, model: str, speaker_backend: str, memory_mode: bool, no_align: bool) -> float:
    base = {
        "tiny": 0.25,
        "base": 0.4,
        "small": 0.6,
        "medium": 0.9,
        "large-v3": 1.6,
    }.get(model, 0.9)
    if speaker_backend in {"Ohne Token", "diarize"}:
        base += 0.15
    if not memory_mode:
        base *= 0.85
    if no_align:
        base *= 0.9
    return base


def _runtime_ratio(run: dict[str, str]) -> float | None:
    try:
        duration = float(run.get("audio_duration", ""))
        processing = float(run.get("processing_time", ""))
    except ValueError:
        return None
    if duration <= 0 or processing <= 0:
        return None
    return processing / duration


def _matches_optional(run: dict[str, str], key: str, value: int | None) -> bool:
    if value is None:
        return True
    return run.get(key) == str(value)


def _matches_optional_text(run: dict[str, str], key: str, value: str | None) -> bool:
    if value is None:
        return True
    return run.get(key, value) == value
