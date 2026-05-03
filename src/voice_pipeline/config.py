from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "audio": {"sample_rate": 24000, "mono": True, "padding_ms": 100},
    "segments": {"min_duration_sec": 2.0, "max_duration_sec": 15.0},
    "quality": {
        "min_duration_sec": 2.0,
        "max_duration_sec": 15.0,
        "min_speech_ratio": 0.75,
        "max_clipping_ratio": 0.001,
        "min_rms_db": -35,
        "max_peak_db": -0.5,
        "reject_empty_transcript": True,
    },
    "reference_pack": {
        "target_total_sec": 60,
        "preferred_min_duration_sec": 4,
        "preferred_max_duration_sec": 12,
    },
    "transcription": {"backend": "faster-whisper", "model": "medium", "language": "de"},
    "synthesis": {"default_backend": "xtts", "require_consent": True},
}


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: Path | None = None) -> dict[str, Any]:
    if path is None:
        return deepcopy(DEFAULT_CONFIG)
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return _deep_merge(DEFAULT_CONFIG, loaded)
