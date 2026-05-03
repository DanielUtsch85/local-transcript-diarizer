from pathlib import Path

import pytest

from voice_pipeline.synthesis.xtts_backend import XTTSBackend


def test_xtts_backend_requires_isolated_python(tmp_path, monkeypatch):
    monkeypatch.delenv("XTTS_PYTHON", raising=False)
    backend = XTTSBackend(python_path=tmp_path / "missing-python")

    with pytest.raises(RuntimeError, match="separate Python environment"):
        backend.synthesize("Hallo", [Path("ref.wav")], tmp_path / "out.wav", language="de")
