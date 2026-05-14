from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Callable, Sequence

from .base import SynthesisBackend


_SUPPORTED_LANGUAGES = {"en", "es", "fr", "zh", "jp", "kr"}


class OpenVoiceBackend(SynthesisBackend):
    """OpenVoice V2 adapter running in an isolated Python environment."""

    def __init__(
        self,
        checkpoint_dir: Path | None = None,
        python_path: Path | None = None,
        timeout_sec: int = 30 * 60,
        pid_file: Path | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.python_path = python_path
        self.timeout_sec = timeout_sec
        self.pid_file = pid_file
        self.progress_callback = progress_callback

    def synthesize(
        self,
        text: str,
        reference_wavs: Sequence[Path],
        output_wav: Path,
        language: str = "de",
    ) -> Path:
        normalized_language = language.lower()
        if normalized_language == "en-us":
            normalized_language = "en"
        if normalized_language not in _SUPPORTED_LANGUAGES:
            raise RuntimeError(
                "OpenVoice V2 with MeloTTS does not natively support this synthesis language. "
                "Use one of: en, es, fr, zh, jp, kr. German output requires an external base TTS "
                "or a different backend."
            )
        if not reference_wavs:
            raise ValueError("OpenVoice synthesis requires at least one reference wav")
        python = self._resolve_python()
        checkpoint_dir = self._resolve_checkpoint_dir()
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        runner = _repo_root() / "scripts" / "openvoice_synthesize.py"
        if not runner.exists():
            raise RuntimeError(f"OpenVoice runner not found: {runner}")
        payload = {
            "text": text,
            "reference_wav": str(_openvoice_reference(reference_wavs, output_wav)),
            "output_wav": str(output_wav),
            "language": normalized_language,
            "checkpoint_dir": str(checkpoint_dir),
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", encoding="utf-8", delete=False) as handle:
            json.dump(payload, handle, ensure_ascii=False)
            request_path = Path(handle.name)
        process: subprocess.Popen[str] | None = None
        stdout = ""
        stderr = ""
        try:
            env = os.environ.copy()
            cache_dir = _repo_root() / "data" / "work" / "openvoice_cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            env.setdefault("HF_HOME", str(cache_dir / "huggingface"))
            env.setdefault("XDG_CACHE_HOME", str(cache_dir / "xdg"))
            env.setdefault("MPLCONFIGDIR", str(cache_dir / "matplotlib"))
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
        finally:
            request_path.unlink(missing_ok=True)
            if self.pid_file:
                self.pid_file.unlink(missing_ok=True)
        if process is not None and process.returncode != 0:
            details = _process_details(stderr, stdout) or f"process exited with code {process.returncode}"
            raise RuntimeError(f"OpenVoice synthesis failed in isolated environment: {details}")
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
                    details = _process_details(exc.stdout, exc.stderr, stdout, stderr)
                    suffix = f": {details}" if details else ""
                    raise RuntimeError(f"OpenVoice synthesis timed out after {self.timeout_sec} seconds{suffix}") from exc

    def _resolve_python(self) -> Path:
        env_python = os.environ.get("OPENVOICE_PYTHON")
        configured = self.python_path or (Path(env_python) if env_python else None)
        if not configured:
            configured = _repo_root() / ".venv-openvoice" / "bin" / "python"
        if not configured.exists():
            raise RuntimeError(
                "OpenVoice requires a separate Python environment. Run `bash scripts/setup_openvoice_env.sh` "
                "or set OPENVOICE_PYTHON to a Python executable with OpenVoice and MeloTTS installed."
            )
        return configured

    def _resolve_checkpoint_dir(self) -> Path:
        env_dir = os.environ.get("OPENVOICE_CHECKPOINT_DIR")
        configured = self.checkpoint_dir or (Path(env_dir) if env_dir else None)
        if not configured:
            configured = _repo_root() / "data" / "models" / "openvoice" / "checkpoints_v2"
        converter_config = configured / "converter" / "config.json"
        converter_ckpt = configured / "converter" / "checkpoint.pth"
        base_ses = configured / "base_speakers" / "ses"
        if not converter_config.exists() or not converter_ckpt.exists() or not base_ses.exists():
            raise RuntimeError(
                "OpenVoice V2 checkpoints are not configured. Set OPENVOICE_CHECKPOINT_DIR to a "
                "checkpoints_v2 directory containing converter/config.json, converter/checkpoint.pth, "
                "and base_speakers/ses."
            )
        return configured


def _openvoice_reference(reference_wavs: Sequence[Path], output_wav: Path) -> Path:
    combined = output_wav.parent.parent / "voice_refs" / "combined_ref.wav"
    if combined.exists():
        return combined
    return Path(reference_wavs[0])


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _as_text(value: str | bytes) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _process_details(*parts: str | bytes | None, max_chars: int = 12000) -> str:
    text = "\n".join(_as_text(part) for part in parts if part).replace("\r", "\n")
    lines = [line for line in text.splitlines() if line.strip()]
    cleaned = "\n".join(lines).strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return "[process output truncated]\n" + cleaned[-max_chars:]
