import wave
from pathlib import Path

from voice_pipeline.logging_utils import write_jsonl
from voice_pipeline.reference_pack import build_reference_pack, build_reference_pack_from_rows


def _write_wav(path: Path, duration_sec: float = 0.1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sample_rate = 16000
    frames = b"\x00\x00" * int(sample_rate * duration_sec)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(frames)


def test_reference_pack_clears_stale_refs(tmp_path: Path):
    clean = tmp_path / "clean"
    first = clean / "first.wav"
    second = clean / "second.wav"
    _write_wav(first)
    _write_wav(second)
    manifest = tmp_path / "accepted.jsonl"
    write_jsonl(
        manifest,
        [
            {"file": str(first), "accepted": True, "duration_sec": 5.0, "speech_ratio": 0.9, "clipping_ratio": 0.0, "rms_db": -22},
            {"file": str(second), "accepted": True, "duration_sec": 5.0, "speech_ratio": 0.8, "clipping_ratio": 0.0, "rms_db": -22},
        ],
    )
    out = tmp_path / "voice_refs"

    build_reference_pack(clean, manifest, out, target_total_sec=10)
    assert len(list(out.glob("ref_*.wav"))) == 2

    build_reference_pack_from_rows(clean, [{"file": str(first), "duration_sec": 5.0}], out, target_total_sec=5)

    refs = sorted(out.glob("ref_*.wav"))
    assert [ref.name for ref in refs] == ["ref_001.wav"]
    assert (out / "combined_ref.wav").exists()


def test_manual_reference_pack_records_selection_mode(tmp_path: Path):
    clean = tmp_path / "clean"
    source = clean / "source.wav"
    _write_wav(source)

    metadata = build_reference_pack_from_rows(clean, [{"file": str(source), "duration_sec": 4.2}], tmp_path / "voice_refs")

    assert metadata["selection_mode"] == "manual"
    assert metadata["actual_total_sec"] == 4.2
    assert len(metadata["refs"]) == 1
