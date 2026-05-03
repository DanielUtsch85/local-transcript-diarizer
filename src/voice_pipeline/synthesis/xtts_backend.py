from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Sequence

from .base import SynthesisBackend


class XTTSBackend(SynthesisBackend):
    """Adapter for Coqui XTTS-v2 running in an isolated Python environment."""

    def __init__(
        self,
        model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2",
        python_path: Path | None = None,
    ) -> None:
        self.model_name = model_name
        self.python_path = python_path

    def synthesize(
        self,
        text: str,
        reference_wavs: Sequence[Path],
        output_wav: Path,
        language: str = "de",
    ) -> Path:
        if not reference_wavs:
            raise ValueError("XTTS synthesis requires at least one reference wav")
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        python = self._resolve_python()
        runner = _repo_root() / "scripts" / "xtts_synthesize.py"
        if not runner.exists():
            raise RuntimeError(f"XTTS runner not found: {runner}")
        payload = {
            "model_name": self.model_name,
            "text": text,
            "reference_wavs": [str(path) for path in reference_wavs],
            "output_wav": str(output_wav),
            "language": language,
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", encoding="utf-8", delete=False) as handle:
            json.dump(payload, handle, ensure_ascii=False)
            request_path = Path(handle.name)
        try:
            env = os.environ.copy()
            env.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
            completed = subprocess.run(
                [str(python), str(runner), "--request", str(request_path)],
                check=False,
                capture_output=True,
                text=True,
                env=env,
            )
        finally:
            request_path.unlink(missing_ok=True)
        if completed.returncode != 0:
            details = (completed.stderr or completed.stdout).strip()
            raise RuntimeError(f"XTTS synthesis failed in isolated environment: {details}")
        return output_wav

    def _resolve_python(self) -> Path:
        env_python = os.environ.get("XTTS_PYTHON")
        configured = self.python_path or (Path(env_python) if env_python else None)
        if not configured:
            configured = _repo_root() / ".venv-xtts" / "bin" / "python"
        if not configured.exists():
            raise RuntimeError(
                "XTTS requires a separate Python environment. Run `bash scripts/setup_xtts_env.sh` "
                "or set XTTS_PYTHON to a Python executable with Coqui TTS installed."
            )
        return configured


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]
