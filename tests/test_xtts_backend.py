from pathlib import Path
import subprocess

import pytest

from voice_pipeline.synthesis.xtts_backend import XTTSBackend


def test_xtts_backend_requires_isolated_python(tmp_path, monkeypatch):
    monkeypatch.delenv("XTTS_PYTHON", raising=False)
    backend = XTTSBackend(python_path=tmp_path / "missing-python", license_confirmed=True)

    with pytest.raises(RuntimeError, match="separate Python environment"):
        backend.synthesize("Hallo", [Path("ref.wav")], tmp_path / "out.wav", language="de")


def test_xtts_backend_times_out(tmp_path, monkeypatch):
    python = tmp_path / "python"
    python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    python.chmod(0o755)
    backend = XTTSBackend(python_path=python, timeout_sec=0, license_confirmed=True)

    class FakeProcess:
        pid = 1234
        returncode = None

        def __init__(self):
            self.calls = 0

        def communicate(self, timeout=None):
            self.calls += 1
            if self.calls == 1:
                raise subprocess.TimeoutExpired(cmd=["xtts"], timeout=timeout)
            self.returncode = -9
            return "", ""

        def kill(self):
            self.returncode = -9

    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: FakeProcess())

    with pytest.raises(RuntimeError, match="timed out after 0 seconds"):
        backend.synthesize("Hallo", [Path("ref.wav")], tmp_path / "out.wav", language="de")


def test_xtts_backend_requires_license_confirmation(tmp_path):
    python = tmp_path / "python"
    python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    python.chmod(0o755)
    backend = XTTSBackend(python_path=python)

    with pytest.raises(RuntimeError, match="license terms"):
        backend.synthesize("Hallo", [Path("ref.wav")], tmp_path / "out.wav", language="de")
