from pathlib import Path
import subprocess

import pytest

from voice_pipeline.synthesis.openvoice_backend import OpenVoiceBackend


def test_openvoice_backend_requires_supported_language(tmp_path):
    python = tmp_path / "python"
    python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    python.chmod(0o755)
    backend = OpenVoiceBackend(python_path=python, checkpoint_dir=tmp_path)

    with pytest.raises(RuntimeError, match="does not natively support"):
        backend.synthesize("Hallo", [Path("ref.wav")], tmp_path / "out.wav", language="de")


def test_openvoice_backend_requires_isolated_python(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENVOICE_PYTHON", raising=False)
    backend = OpenVoiceBackend(python_path=tmp_path / "missing-python")

    with pytest.raises(RuntimeError, match="separate Python environment"):
        backend.synthesize("Hello", [Path("ref.wav")], tmp_path / "out.wav", language="en")


def test_openvoice_backend_requires_checkpoints(tmp_path):
    python = tmp_path / "python"
    python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    python.chmod(0o755)
    backend = OpenVoiceBackend(python_path=python, checkpoint_dir=tmp_path / "missing-checkpoints")

    with pytest.raises(RuntimeError, match="checkpoints are not configured"):
        backend.synthesize("Hello", [Path("ref.wav")], tmp_path / "out.wav", language="en")


def test_openvoice_backend_times_out(tmp_path, monkeypatch):
    python = tmp_path / "python"
    python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    python.chmod(0o755)
    checkpoint_dir = tmp_path / "checkpoints_v2"
    (checkpoint_dir / "converter").mkdir(parents=True)
    (checkpoint_dir / "base_speakers" / "ses").mkdir(parents=True)
    (checkpoint_dir / "converter" / "config.json").write_text("{}", encoding="utf-8")
    (checkpoint_dir / "converter" / "checkpoint.pth").write_bytes(b"checkpoint")
    backend = OpenVoiceBackend(python_path=python, checkpoint_dir=checkpoint_dir, timeout_sec=0)

    class FakeProcess:
        pid = 1234
        returncode = None

        def __init__(self):
            self.calls = 0

        def communicate(self, timeout=None):
            self.calls += 1
            if self.calls == 1:
                raise subprocess.TimeoutExpired(cmd=["openvoice"], timeout=timeout)
            self.returncode = -9
            return "", ""

        def kill(self):
            self.returncode = -9

    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: FakeProcess())

    with pytest.raises(RuntimeError, match="timed out after 0 seconds"):
        backend.synthesize("Hello", [Path("ref.wav")], tmp_path / "out.wav", language="en")
