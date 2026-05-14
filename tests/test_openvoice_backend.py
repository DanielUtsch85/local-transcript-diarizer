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
    log_lines: list[str] = []
    backend = OpenVoiceBackend(
        python_path=python,
        checkpoint_dir=checkpoint_dir,
        timeout_sec=0,
        log_callback=log_lines.append,
    )

    class FakeProcess:
        pid = 1234
        returncode = None

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            if self.returncode is None:
                self.returncode = -9
            return self.returncode

        def kill(self):
            self.returncode = -9

    def fake_popen(*args, **kwargs):
        kwargs["stdout"].write("[openvoice] loading MeloTTS\n")
        kwargs["stdout"].flush()
        return FakeProcess()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    with pytest.raises(RuntimeError, match="loading MeloTTS"):
        backend.synthesize("Hello", [Path("ref.wav")], tmp_path / "out.wav", language="en")
    assert log_lines == [
        f"[openvoice] log={tmp_path / 'work' / 'openvoice_synthesis.log'}",
        "[openvoice] loading MeloTTS",
    ]
