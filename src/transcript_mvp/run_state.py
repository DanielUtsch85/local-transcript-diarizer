from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any


def write_run_state(run_dir: Path, stage: str, **metadata: Any) -> Path | None:
    payload = {
        "stage": stage,
        "updated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        **metadata,
    }
    path = run_dir / "run_state.json"
    try:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    except OSError:
        return None
    return path
