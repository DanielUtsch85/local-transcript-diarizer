from __future__ import annotations

import shutil
from collections import Counter
from pathlib import Path
from typing import Optional

import typer

from .audio_io import cut_segment, prepare_audio
from .config import load_config
from .logging_utils import read_jsonl, write_jsonl
from .metadata import synthesis_metadata, validate_consent, write_synthesis_log
from .quality import score_segment
from .reference_pack import build_reference_pack as build_reference_pack_impl
from .segment_loader import filter_segments, load_segments, segment_filename
from .synthesis.mock_backend import MockBackend
from .synthesis.openvoice_backend import OpenVoiceBackend
from .synthesis.xtts_backend import XTTSBackend
from .transcription import FasterWhisperTranscriber

app = typer.Typer(no_args_is_help=True)


def _speaker_dir_from_out(out: Path) -> Path:
    parts = out.parts
    if "raw_segments" in parts:
        return Path(*parts[: parts.index("raw_segments")])
    return out


def _backend(name: str):
    normalized = name.lower()
    if normalized == "mock":
        return MockBackend()
    if normalized == "xtts":
        return XTTSBackend()
    if normalized == "openvoice":
        return OpenVoiceBackend()
    raise typer.BadParameter(f"Unsupported backend: {name}")


@app.command("prepare-audio")
def prepare_audio_command(
    input: Path = typer.Option(..., "--input", exists=True),
    output: Path = typer.Option(..., "--output"),
    sample_rate: int = typer.Option(24000, "--sample-rate"),
    mono: bool = typer.Option(True, "--mono/--stereo"),
) -> None:
    metadata = prepare_audio(input, output, sample_rate=sample_rate, mono=mono)
    typer.echo(f"prepared: {metadata['prepared_file']}")


@app.command("extract-speaker")
def extract_speaker_command(
    segments: Path = typer.Option(..., "--segments", exists=True),
    speaker: str = typer.Option(..., "--speaker"),
    out: Path = typer.Option(..., "--out"),
    min_duration_sec: float = typer.Option(2.0, "--min-duration-sec"),
    max_duration_sec: float = typer.Option(15.0, "--max-duration-sec"),
    padding_ms: int = typer.Option(100, "--padding-ms"),
) -> None:
    selected = filter_segments(load_segments(segments), speaker, min_duration_sec, max_duration_sec)
    rows = []
    for index, segment in enumerate(sorted(selected, key=lambda item: (str(item.source_file), item.start)), start=1):
        dest = out / segment_filename(index, segment)
        cut_segment(segment.source_file, dest, segment.start, segment.end, padding_ms=padding_ms)
        rows.append(
            {
                "file": str(dest),
                "speaker": speaker,
                "source_file": str(segment.source_file),
                "start": segment.start,
                "end": segment.end,
                "duration_sec": segment.duration_sec,
            }
        )
    write_jsonl(out.parent / "manifests" / "raw_segments.jsonl", rows)
    typer.echo(f"extracted: {len(rows)}")


@app.command("score-segments")
def score_segments_command(
    input: Path = typer.Option(..., "--input", exists=True),
    out: Path = typer.Option(..., "--out"),
    speaker: Optional[str] = typer.Option(None, "--speaker"),
    config: Optional[Path] = typer.Option(None, "--config"),
    transcribe: bool = typer.Option(False, "--transcribe/--no-transcribe"),
    language: str = typer.Option("de", "--language"),
) -> None:
    cfg = load_config(config)
    transcriber = None
    if transcribe:
        try:
            transcriber = FasterWhisperTranscriber(
                model=str(cfg["transcription"].get("model", "medium")),
                language=language,
            )
        except RuntimeError as exc:
            typer.echo(f"transcription disabled: {exc}", err=True)
    rows = []
    for wav in sorted(input.glob("*.wav")):
        transcript = None
        transcript_language = None
        confidence = None
        if transcriber is not None:
            result = transcriber.transcribe(wav)
            transcript = result.text
            transcript_language = result.language
            confidence = result.confidence
        rows.append(score_segment(wav, speaker=speaker, quality_config=cfg["quality"], transcript=transcript, language=transcript_language, transcript_confidence=confidence))
    write_jsonl(out, rows)
    typer.echo(f"scored: {len(rows)}")


def _copy_curated(row: dict, destination: Path) -> dict:
    source = Path(str(row["file"]))
    destination.mkdir(parents=True, exist_ok=True)
    dest = destination / source.name
    if source.resolve() != dest.resolve():
        shutil.copy2(source, dest)
    updated = dict(row)
    updated["file"] = str(dest)
    return updated


def _write_report(speaker_dir: Path, accepted: list[dict], rejected: list[dict]) -> None:
    all_rows = accepted + rejected
    durations = sorted(float(row.get("duration_sec") or 0) for row in all_rows)
    median = durations[len(durations) // 2] if durations else 0.0
    reasons = Counter(reason for row in rejected for reason in row.get("reject_reasons", []))
    speaker = speaker_dir.name
    lines = [
        f"# Speaker Report: {speaker}",
        "## Summary",
        f"- Raw segments: {len(all_rows)}",
        f"- Accepted: {len(accepted)}",
        f"- Rejected: {len(rejected)}",
        f"- Accepted duration: {sum(float(row.get('duration_sec') or 0) for row in accepted):.1f} s",
        f"- Median segment duration: {median:.1f} s",
        "## Rejection reasons",
    ]
    lines.extend(f"- {reason}: {count}" for reason, count in sorted(reasons.items()))
    (speaker_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


@app.command("curate")
def curate_command(
    manifest: Path = typer.Option(..., "--manifest", exists=True),
    accepted_dir: Path = typer.Option(..., "--accepted-dir"),
    rejected_dir: Path = typer.Option(..., "--rejected-dir"),
) -> None:
    rows = read_jsonl(manifest)
    accepted = [_copy_curated(row, accepted_dir) for row in rows if row.get("accepted")]
    rejected = [_copy_curated(row, rejected_dir) for row in rows if not row.get("accepted")]
    speaker_dir = accepted_dir.parent
    manifests = speaker_dir / "manifests"
    write_jsonl(manifests / "accepted.jsonl", accepted)
    write_jsonl(manifests / "rejected.jsonl", rejected)
    _write_report(speaker_dir, accepted, rejected)
    typer.echo(f"accepted: {len(accepted)}, rejected: {len(rejected)}")


@app.command("build-reference-pack")
def build_reference_pack_command(
    clean_segments: Path = typer.Option(..., "--clean-segments", exists=True),
    manifest: Path = typer.Option(..., "--manifest", exists=True),
    out: Path = typer.Option(..., "--out"),
    target_total_sec: float = typer.Option(60.0, "--target-total-sec"),
) -> None:
    metadata = build_reference_pack_impl(clean_segments, manifest, out, target_total_sec=target_total_sec)
    typer.echo(f"refs: {len(metadata['refs'])}")


@app.command("synthesize")
def synthesize_command(
    backend: str = typer.Option(..., "--backend"),
    speaker_dir: Path = typer.Option(..., "--speaker-dir", exists=True),
    text: str = typer.Option(..., "--text"),
    language: str = typer.Option("de", "--language"),
    out: Path = typer.Option(..., "--out"),
    consent_confirmed: bool = typer.Option(False, "--consent-confirmed"),
) -> None:
    try:
        validate_consent(consent_confirmed)
    except PermissionError as exc:
        raise typer.BadParameter(str(exc))
    reference_files = sorted((speaker_dir / "voice_refs").glob("ref_*.wav"))
    result = _backend(backend).synthesize(text, reference_files, out, language=language)
    metadata = synthesis_metadata(speaker_dir.name, backend, text, language, reference_files, result, consent_confirmed)
    write_synthesis_log(speaker_dir / "manifests" / "synthesis_runs.jsonl", metadata, result)
    typer.echo(f"synthetic: {result}")


@app.command("run-all")
def run_all_command(
    audio: Path = typer.Option(..., "--audio", exists=True),
    segments: Path = typer.Option(..., "--segments", exists=True),
    speaker: str = typer.Option(..., "--speaker"),
    text: str = typer.Option(..., "--text"),
    backend: str = typer.Option("xtts", "--backend"),
    language: str = typer.Option("de", "--language"),
    consent_confirmed: bool = typer.Option(False, "--consent-confirmed"),
    out: Path = typer.Option(..., "--out"),
    config: Optional[Path] = typer.Option(None, "--config"),
) -> None:
    cfg = load_config(config)
    validate_consent(consent_confirmed)
    prepared = Path("data/work") / f"{audio.stem}.wav"
    prepare_audio(audio, prepared, sample_rate=int(cfg["audio"]["sample_rate"]), mono=bool(cfg["audio"]["mono"]))
    raw = out / "raw_segments"
    clean = out / "clean_segments"
    rejected = out / "rejected_segments"
    manifests = out / "manifests"
    selected = filter_segments(load_segments(segments), speaker, cfg["segments"]["min_duration_sec"], cfg["segments"]["max_duration_sec"])
    rows = []
    for index, segment in enumerate(selected, start=1):
        dest = raw / segment_filename(index, segment)
        cut_segment(segment.source_file, dest, segment.start, segment.end, padding_ms=int(cfg["audio"]["padding_ms"]))
        rows.append(score_segment(dest, speaker=speaker, quality_config=cfg["quality"]))
    write_jsonl(manifests / "segments.jsonl", rows)
    accepted = [_copy_curated(row, clean) for row in rows if row.get("accepted")]
    rejected_rows = [_copy_curated(row, rejected) for row in rows if not row.get("accepted")]
    write_jsonl(manifests / "accepted.jsonl", accepted)
    write_jsonl(manifests / "rejected.jsonl", rejected_rows)
    _write_report(out, accepted, rejected_rows)
    build_reference_pack_impl(clean, manifests / "accepted.jsonl", out / "voice_refs", target_total_sec=float(cfg["reference_pack"]["target_total_sec"]))
    synth_out = out / "generated_samples" / f"sample_{backend}_001.wav"
    reference_files = sorted((out / "voice_refs").glob("ref_*.wav"))
    result = _backend(backend).synthesize(text, reference_files, synth_out, language=language)
    metadata = synthesis_metadata(speaker, backend, text, language, reference_files, result, consent_confirmed)
    write_synthesis_log(manifests / "synthesis_runs.jsonl", metadata, result)
    typer.echo(f"completed: {out}")


if __name__ == "__main__":
    app()
