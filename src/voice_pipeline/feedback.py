from __future__ import annotations

import csv
from collections import Counter
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any


def build_voice_feedback_csv(
    *,
    speaker: str,
    source_audio: Path | str,
    diarization_path: Path | str,
    settings: dict[str, Any],
    raw_rows: list[dict[str, Any]],
    scored_rows: list[dict[str, Any]],
    accepted_rows: list[dict[str, Any]],
    rejected_rows: list[dict[str, Any]],
    reference_metadata: dict[str, Any] | None = None,
    synthesis_metadata: dict[str, Any] | None = None,
    synthesis_error: str | None = None,
) -> str:
    buffer = StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["category", "metric", "value", "unit", "notes"])

    def add(category: str, metric: str, value: Any, unit: str = "", notes: str = "") -> None:
        writer.writerow([category, metric, value, unit, notes])

    all_durations = [float(row.get("duration_sec") or 0) for row in scored_rows]
    accepted_durations = [float(row.get("duration_sec") or 0) for row in accepted_rows]
    reject_reasons = Counter(reason for row in rejected_rows for reason in row.get("reject_reasons", []))
    overlap_rows = [row for row in scored_rows if float(row.get("overlap_sec") or 0) > 0]

    add("run", "speaker", speaker)
    add("run", "source_audio", source_audio)
    add("run", "diarization_path", diarization_path)
    for key, value in settings.items():
        add("settings", key, value)
    add("segments", "raw_count", len(raw_rows), "segments")
    add("segments", "scored_count", len(scored_rows), "segments")
    add("segments", "accepted_count", len(accepted_rows), "segments")
    add("segments", "rejected_count", len(rejected_rows), "segments")
    add("segments", "acceptance_rate", _ratio(len(accepted_rows), len(scored_rows)))
    add("duration", "raw_total", round(sum(all_durations), 3), "seconds")
    add("duration", "accepted_total", round(sum(accepted_durations), 3), "seconds")
    add("duration", "accepted_median", _median(accepted_durations), "seconds")
    add("duration", "accepted_min", min(accepted_durations) if accepted_durations else 0, "seconds")
    add("duration", "accepted_max", max(accepted_durations) if accepted_durations else 0, "seconds")
    add("overlap", "overlapping_segments", len(overlap_rows), "segments")
    add("overlap", "overlap_total", round(sum(float(row.get("overlap_sec") or 0) for row in scored_rows), 3), "seconds")
    add("overlap", "rejected_for_overlap", reject_reasons.get("overlaps_other_speaker", 0), "segments")
    for reason, count in sorted(reject_reasons.items()):
        add("reject_reason", reason, count, "segments")

    if reference_metadata:
        refs = reference_metadata.get("refs", [])
        add("reference_pack", "ref_count", len(refs), "files")
        add("reference_pack", "target_total_sec", reference_metadata.get("target_total_sec", 0), "seconds")
        add("reference_pack", "actual_total_sec", reference_metadata.get("actual_total_sec", 0), "seconds")
    if synthesis_metadata:
        add("synthesis", "backend", synthesis_metadata.get("backend", ""))
        add("synthesis", "synthetic", synthesis_metadata.get("synthetic", ""))
        add("synthesis", "output_file", synthesis_metadata.get("output_file", ""))
    if synthesis_error:
        add("synthesis", "error", synthesis_error)
    return buffer.getvalue()


def build_synthesis_review_csv(
    *,
    speaker: str,
    sample_wav: Path | str,
    technical_feedback_csv: str,
    review: dict[str, Any],
    sample_synthesis_metadata: dict[str, Any] | None = None,
) -> str:
    buffer = StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["category", "metric", "value", "unit", "notes"])

    def add(category: str, metric: str, value: Any, unit: str = "", notes: str = "") -> None:
        writer.writerow([category, metric, value, unit, notes])

    technical_rows = list(csv.DictReader(StringIO(technical_feedback_csv)))
    add("run", "speaker", speaker)
    add("run", "sample_wav", sample_wav)
    add("run", "review_timestamp", datetime.now(timezone.utc).isoformat())
    for row in technical_rows:
        category = row.get("category", "")
        metric = row.get("metric", "")
        if sample_synthesis_metadata and category == "synthesis":
            continue
        if category and metric:
            add(f"technical_{category}", metric, row.get("value", ""), row.get("unit", ""), row.get("notes", ""))
    if sample_synthesis_metadata:
        add("technical_synthesis", "backend", sample_synthesis_metadata.get("backend", ""))
        add("technical_synthesis", "synthetic", sample_synthesis_metadata.get("synthetic", ""))
        add("technical_synthesis", "output_file", sample_synthesis_metadata.get("output_file", ""))
        add("technical_synthesis", "reference_file_count", len(sample_synthesis_metadata.get("reference_files", [])), "files")
    for key, value in review.items():
        add("human_review", key, value)
    return buffer.getvalue()


def _ratio(part: int, total: int) -> float:
    return round(part / total, 4) if total else 0.0


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return round(ordered[len(ordered) // 2], 3)
