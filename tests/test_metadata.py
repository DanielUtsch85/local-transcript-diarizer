import json

import pytest

from voice_pipeline.metadata import synthesis_metadata, validate_consent, write_synthesis_log


def test_consent_flag_is_validated():
    with pytest.raises(PermissionError):
        validate_consent(False)


def test_synthetic_sidecar_json_is_written(tmp_path):
    output = tmp_path / "sample_xtts_001.wav"
    output.write_bytes(b"RIFF")
    metadata = synthesis_metadata(
        speaker="SPEAKER_02",
        backend="mock",
        text="Hallo",
        language="de",
        reference_files=[tmp_path / "ref_001.wav"],
        output_file=output,
        consent_confirmed=True,
    )

    write_synthesis_log(tmp_path / "synthesis_runs.jsonl", metadata, output)

    sidecar = output.with_suffix(output.suffix + ".synthetic.json")
    assert sidecar.exists()
    assert json.loads(sidecar.read_text(encoding="utf-8"))["synthetic"] is True
    assert "consent_confirmed" in (tmp_path / "synthesis_runs.jsonl").read_text(encoding="utf-8")
