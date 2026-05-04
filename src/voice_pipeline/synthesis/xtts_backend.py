from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Callable, Sequence

from .base import SynthesisBackend


class XTTSBackend(SynthesisBackend):
    """Adapter for Coqui XTTS-v2 running in an isolated Python environment."""

    def __init__(
        self,
        model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2",
        python_path: Path | None = None,
        timeout_sec: int = 30 * 60,
        pid_file: Path | None = None,
        license_confirmed: bool = False,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> None:
        self.model_name = model_name
        self.python_path = python_path
        self.timeout_sec = timeout_sec
        self.pid_file = pid_file
        self.license_confirmed = license_confirmed
        self.progress_callback = progress_callback

    def synthesize(
        self,
        text: str,
        reference_wavs: Sequence[Path],
        output_wav: Path,
        language: str = "de",
    ) -> Path:
        if not reference_wavs:
            raise ValueError("XTTS synthesis requires at least one reference wav")
        if not self.license_confirmed:
            raise RuntimeError(
                "XTTS requires explicit confirmation of the applicable Coqui license terms "
                "before synthesis can run."
            )
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
        process: subprocess.Popen[str] | None = None
        stdout = ""
        stderr = ""
        try:
            env = os.environ.copy()
            env.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
            env["COQUI_TOS_AGREED"] = "1"
            process = subprocess.Popen(
                [str(python), str(runner), "--request", str(request_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
            )
            if self.pid_file:
                self.pid_file.parent.mkdir(parents=True, exist_ok=True)
                self.pid_file.write_text(str(process.pid), encoding="utf-8")
            stdout, stderr = self._communicate_with_countdown(process)
        except subprocess.TimeoutExpired as exc:
            details = "\n".join(_as_text(part) for part in [exc.stdout, exc.stderr] if part).strip()
            suffix = f": {details}" if details else ""
            raise RuntimeError(f"XTTS synthesis timed out after {self.timeout_sec} seconds{suffix}") from exc
        finally:
            request_path.unlink(missing_ok=True)
            if self.pid_file:
                self.pid_file.unlink(missing_ok=True)
        if process is not None and process.returncode != 0:
            details = (stderr or stdout).strip() or f"process exited with code {process.returncode}"
            raise RuntimeError(f"XTTS synthesis failed in isolated environment: {details}")
        return output_wav

    def _communicate_with_countdown(self, process: subprocess.Popen[str]) -> tuple[str, str]:
        started_at = time.monotonic()
        stdout = ""
        stderr = ""
        while True:
            elapsed = int(time.monotonic() - started_at)
            remaining = max(0, int(self.timeout_sec - elapsed))
            if self.progress_callback:
                self.progress_callback(elapsed, remaining)
            try:
                stdout, stderr = process.communicate(timeout=min(1, max(remaining, 1)))
                return stdout, stderr
            except subprocess.TimeoutExpired as exc:
                if time.monotonic() - started_at >= self.timeout_sec:
                    process.kill()
                    stdout, stderr = process.communicate()
                    details = "\n".join(
                        _as_text(part) for part in [exc.stdout, exc.stderr, stdout, stderr] if part
                    ).strip()
                    suffix = f": {details}" if details else ""
                    raise RuntimeError(f"XTTS synthesis timed out after {self.timeout_sec} seconds{suffix}") from exc

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


def _as_text(value: str | bytes) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value
