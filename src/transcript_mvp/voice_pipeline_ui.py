from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from transcript_mvp.models import SpeakerSegment, TranscriptSegment
from voice_pipeline.audio_io import cut_segment, prepare_audio
from voice_pipeline.config import load_config
from voice_pipeline.feedback import build_voice_feedback_csv
from voice_pipeline.logging_utils import read_jsonl, write_jsonl
from voice_pipeline.metadata import synthesis_metadata, validate_consent, write_synthesis_log
from voice_pipeline.quality import score_segment, segment_overlap
from voice_pipeline.reference_pack import build_reference_pack
from voice_pipeline.segment_loader import filter_segments, load_segments, segment_filename
from voice_pipeline.synthesis.mock_backend import MockBackend
from voice_pipeline.synthesis.openvoice_backend import OpenVoiceBackend
from voice_pipeline.synthesis.xtts_backend import XTTSBackend


def render_voice_pipeline_page(data_dir: Path) -> None:
    st.header("Voice Reference & Synthese")
    st.caption("Erstellt aus vorhandenen Sprechersegmenten kuratierte Referenzclips und synthetisch markierte Samples.")

    source_audio = _resolve_source_audio(data_dir)
    diarization_path = _resolve_diarization_source(data_dir, source_audio)

    if source_audio is None or diarization_path is None:
        st.info("Voraussetzung: Audio-Datei und Sprechersegmente. Nutze die Transkriptionsseite zuerst oder lade beides hier hoch.")
        return

    try:
        segments = load_segments(diarization_path)
    except ValueError as exc:
        st.error(f"Diarization konnte nicht geladen werden: {exc}")
        return

    speakers = sorted({segment.speaker for segment in segments})
    if not speakers:
        st.warning("Keine Sprechersegmente gefunden.")
        return

    with st.form("voice-pipeline-settings"):
        speaker = st.selectbox("Zielsprecher", speakers)
        language = st.selectbox("Sprache", ["de", "en"], index=0)
        backend = st.selectbox("Synthese-Backend", ["mock", "xtts", "openvoice"], index=0)
        if backend != "mock":
            st.caption(_backend_hint(backend))
        text = st.text_area("Synthese-Text", value="Hallo, dies ist ein synthetisch erzeugter Testsatz.")
        consent_confirmed = st.checkbox("Explizite Einwilligung des Zielsprechers liegt vor")
        run_synthesis = st.checkbox("Nach Reference-Pack direkt Synthese erzeugen", value=False)
        with st.expander("Pipeline-Parameter"):
            sample_rate = st.selectbox("Sample Rate", [24000, 16000], index=0)
            min_duration = st.number_input("Min. Segmentlaenge Sekunden", min_value=0.1, max_value=30.0, value=2.0)
            max_duration = st.number_input("Max. Segmentlaenge Sekunden", min_value=0.5, max_value=60.0, value=15.0)
            padding_ms = st.number_input("Padding ms", min_value=0, max_value=1000, value=100, step=25)
            target_total_sec = st.number_input("Ziel-Laenge Reference-Pack Sekunden", min_value=1, max_value=600, value=60)
            reject_overlaps = st.checkbox(
                "Ueberlappungen mit anderen Sprechern ablehnen",
                value=True,
                help="Markiert Segmente als ungeeignet, wenn sich im Diarization-Export zur gleichen Zeit ein anderer Sprecher ueberschneidet.",
            )
        submitted = st.form_submit_button("Referenzdaten bauen", type="primary")

    output_dir = data_dir / "output" / speaker
    if submitted:
        if run_synthesis and not consent_confirmed:
            st.error("Synthese erfordert die bestaetigte Einwilligung des Zielsprechers.")
            return
        _run_pipeline_from_ui(
            source_audio=source_audio,
            diarization_path=diarization_path,
            speaker=speaker,
            output_dir=output_dir,
            sample_rate=int(sample_rate),
            min_duration_sec=float(min_duration),
            max_duration_sec=float(max_duration),
            padding_ms=int(padding_ms),
            target_total_sec=float(target_total_sec),
            reject_overlaps=reject_overlaps,
            run_synthesis=run_synthesis,
            backend=backend,
            text=text,
            language=language,
            consent_confirmed=consent_confirmed,
        )

    _render_existing_outputs(output_dir)


def _resolve_source_audio(data_dir: Path) -> Path | None:
    current = st.session_state.get("last_audio_path")
    if current and Path(current).exists():
        st.success(f"Aktuelle Audio-Datei aus der Transkription: `{current}`")
        return Path(current)

    uploaded = st.file_uploader(
        "Audio-Datei fuer Voice-Pipeline laden",
        type=["mp3", "wav", "m4a", "mp4", "flac"],
        key="voice-audio-upload",
    )
    if uploaded is None:
        return None
    target = data_dir / "uploads" / _safe_filename(uploaded.name)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(uploaded.getvalue())
    st.session_state.last_audio_path = str(target)
    return target


def _resolve_diarization_source(data_dir: Path, source_audio: Path | None) -> Path | None:
    source_segments = st.session_state.get("speaker_segments") or st.session_state.get("segments", [])
    usable_segments = [
        segment
        for segment in source_segments
        if isinstance(segment, (SpeakerSegment, TranscriptSegment))
        and segment.start is not None
        and segment.end is not None
        and segment.end > segment.start
    ]
    if usable_segments and source_audio is not None:
        if st.checkbox("Aktuelle Sprechersegmente aus der App verwenden", value=True):
            path = data_dir / "work" / f"{st.session_state.get('source_name', 'transkript')}-voice-diarization.json"
            rows = [
                {
                    "speaker": segment.speaker,
                    "start": float(segment.start),
                    "end": float(segment.end),
                    "source_file": str(source_audio),
                }
                for segment in usable_segments
            ]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
            return path

    uploaded = st.file_uploader(
        "Oder Diarization JSON/JSONL laden",
        type=["json", "jsonl"],
        key="voice-diarization-upload",
    )
    if uploaded is None:
        return None
    target = data_dir / "work" / _safe_filename(uploaded.name)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(uploaded.getvalue())
    return target


def _run_pipeline_from_ui(
    *,
    source_audio: Path,
    diarization_path: Path,
    speaker: str,
    output_dir: Path,
    sample_rate: int,
    min_duration_sec: float,
    max_duration_sec: float,
    padding_ms: int,
    target_total_sec: float,
    reject_overlaps: bool,
    run_synthesis: bool,
    backend: str,
    text: str,
    language: str,
    consent_confirmed: bool,
) -> None:
    cfg = load_config()
    cfg["quality"]["min_duration_sec"] = min_duration_sec
    cfg["quality"]["max_duration_sec"] = max_duration_sec
    cfg["quality"]["max_overlap_sec"] = 0.0 if reject_overlaps else 999999.0
    prepared = output_dir / "work" / f"{source_audio.stem}_{sample_rate}.wav"
    raw_dir = output_dir / "raw_segments"
    clean_dir = output_dir / "clean_segments"
    rejected_dir = output_dir / "rejected_segments"
    manifests_dir = output_dir / "manifests"

    with st.status("Voice-Pipeline laeuft...", expanded=True) as status:
        st.write("Audio wird vorbereitet.")
        prepare_audio(source_audio, prepared, sample_rate=sample_rate, mono=True)

        st.write("Zielsprecher-Segmente werden geschnitten.")
        rows_for_prepared = _rewrite_diarization_source(diarization_path, prepared, output_dir / "work" / "prepared_diarization.json")
        all_segments = load_segments(rows_for_prepared)
        selected = filter_segments(all_segments, speaker, min_duration_sec, max_duration_sec)
        extracted = []
        for index, segment in enumerate(selected, start=1):
            dest = raw_dir / segment_filename(index, segment)
            cut_segment(segment.source_file, dest, segment.start, segment.end, padding_ms=padding_ms)
            overlap = segment_overlap(segment, all_segments)
            extracted.append(
                {
                    "file": str(dest),
                    "speaker": speaker,
                    "source_file": str(segment.source_file),
                    "start": segment.start,
                    "end": segment.end,
                    "duration_sec": segment.duration_sec,
                    **overlap,
                }
            )
        write_jsonl(manifests_dir / "raw_segments.jsonl", extracted)

        st.write("Segmente werden bewertet.")
        scored = [
            score_segment(
                Path(row["file"]),
                speaker=speaker,
                quality_config=cfg["quality"],
                overlap_sec=float(row.get("overlap_sec") or 0),
                overlap_speakers=list(row.get("overlap_speakers") or []),
            )
            for row in extracted
        ]
        write_jsonl(manifests_dir / "segments.jsonl", scored)

        st.write("Akzeptierte und abgelehnte Segmente werden getrennt.")
        accepted = [_copy_curated(row, clean_dir) for row in scored if row.get("accepted")]
        rejected = [_copy_curated(row, rejected_dir) for row in scored if not row.get("accepted")]
        write_jsonl(manifests_dir / "accepted.jsonl", accepted)
        write_jsonl(manifests_dir / "rejected.jsonl", rejected)
        _write_report(output_dir, accepted, rejected)

        st.write("Reference-Pack wird gebaut.")
        reference = build_reference_pack(clean_dir, manifests_dir / "accepted.jsonl", output_dir / "voice_refs", target_total_sec=target_total_sec)

        synth_metadata = None
        synthesis_error = None
        if run_synthesis:
            st.write("Synthetisches Sample wird erzeugt und geloggt.")
            validate_consent(consent_confirmed)
            output_wav = output_dir / "generated_samples" / f"sample_{backend}_001.wav"
            reference_files = sorted((output_dir / "voice_refs").glob("ref_*.wav"))
            if not reference_files:
                synthesis_error = "Keine Reference-Clips vorhanden; Synthese wurde nicht gestartet."
            else:
                try:
                    result = _backend(backend).synthesize(text, reference_files, output_wav, language=language)
                    synth_metadata = synthesis_metadata(speaker, backend, text, language, reference_files, result, consent_confirmed)
                    write_synthesis_log(manifests_dir / "synthesis_runs.jsonl", synth_metadata, result)
                except (RuntimeError, NotImplementedError, ValueError) as exc:
                    synthesis_error = str(exc)

        feedback_csv = build_voice_feedback_csv(
            speaker=speaker,
            source_audio=source_audio,
            diarization_path=diarization_path,
            settings={
                "sample_rate": sample_rate,
                "min_duration_sec": min_duration_sec,
                "max_duration_sec": max_duration_sec,
                "padding_ms": padding_ms,
                "target_total_sec": target_total_sec,
                "reject_overlaps": reject_overlaps,
                "backend": backend,
                "run_synthesis": run_synthesis,
            },
            raw_rows=extracted,
            scored_rows=scored,
            accepted_rows=accepted,
            rejected_rows=rejected,
            reference_metadata=reference,
            synthesis_metadata=synth_metadata,
            synthesis_error=synthesis_error,
        )
        (manifests_dir / "voice_feedback.csv").write_text(feedback_csv, encoding="utf-8")

        message = (
            f"Fertig: {len(extracted)} roh, {len(accepted)} akzeptiert, {len(rejected)} abgelehnt, "
            f"{len(reference.get('refs', []))} Referenzen."
        )
        if synthesis_error:
            status.update(label="Reference-Pack fertig, Synthese fehlgeschlagen", state="error")
            st.warning(message)
            st.error(f"Synthese fehlgeschlagen: {synthesis_error}")
            st.info(_backend_hint(backend))
        else:
            status.update(label="Voice-Pipeline fertig", state="complete")
            st.success(message)


def _rewrite_diarization_source(input_path: Path, prepared_audio: Path, output_path: Path) -> Path:
    rows = [
        {
            "speaker": segment.speaker,
            "start": segment.start,
            "end": segment.end,
            "source_file": str(prepared_audio),
        }
        for segment in load_segments(input_path)
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def _copy_curated(row: dict[str, Any], destination: Path) -> dict[str, Any]:
    source = Path(str(row["file"]))
    destination.mkdir(parents=True, exist_ok=True)
    dest = destination / source.name
    if source.resolve() != dest.resolve():
        shutil.copy2(source, dest)
    updated = dict(row)
    updated["file"] = str(dest)
    return updated


def _write_report(speaker_dir: Path, accepted: list[dict[str, Any]], rejected: list[dict[str, Any]]) -> None:
    all_rows = accepted + rejected
    durations = sorted(float(row.get("duration_sec") or 0) for row in all_rows)
    median = durations[len(durations) // 2] if durations else 0.0
    reasons = Counter(reason for row in rejected for reason in row.get("reject_reasons", []))
    overlap_count = sum(1 for row in all_rows if float(row.get("overlap_sec") or 0) > 0)
    lines = [
        f"# Speaker Report: {speaker_dir.name}",
        "## Summary",
        f"- Raw segments: {len(all_rows)}",
        f"- Accepted: {len(accepted)}",
        f"- Rejected: {len(rejected)}",
        f"- Accepted duration: {sum(float(row.get('duration_sec') or 0) for row in accepted):.1f} s",
        f"- Median segment duration: {median:.1f} s",
        f"- Overlapping segments: {overlap_count}",
        "## Rejection reasons",
    ]
    lines.extend(f"- {reason}: {count}" for reason, count in sorted(reasons.items()))
    speaker_dir.mkdir(parents=True, exist_ok=True)
    (speaker_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _render_existing_outputs(output_dir: Path) -> None:
    manifests = output_dir / "manifests"
    accepted_path = manifests / "accepted.jsonl"
    rejected_path = manifests / "rejected.jsonl"
    if not accepted_path.exists() and not rejected_path.exists():
        return
    st.divider()
    st.subheader("Aktueller Pipeline-Stand")
    accepted = read_jsonl(accepted_path) if accepted_path.exists() else []
    rejected = read_jsonl(rejected_path) if rejected_path.exists() else []
    cols = st.columns(4)
    cols[0].metric("Akzeptiert", len(accepted))
    cols[1].metric("Abgelehnt", len(rejected))
    cols[2].metric("Ref-Clips", len(list((output_dir / "voice_refs").glob("ref_*.wav"))))
    cols[3].metric("Samples", len(list((output_dir / "generated_samples").glob("*.wav"))))

    if accepted:
        st.dataframe(pd.DataFrame(accepted), use_container_width=True)
    feedback = manifests / "voice_feedback.csv"
    if feedback.exists():
        st.download_button(
            "Voice-Feedback-CSV herunterladen",
            data=feedback.read_bytes(),
            file_name=f"{output_dir.name}-voice-feedback.csv",
            mime="text/csv",
        )
    generated = sorted((output_dir / "generated_samples").glob("*.wav"))
    if generated:
        st.subheader("Generierte Samples")
        for wav in generated:
            st.markdown(f"**{wav.name}**")
            st.audio(wav.read_bytes(), format="audio/wav")
            st.download_button(
                "WAV herunterladen",
                data=wav.read_bytes(),
                file_name=wav.name,
                mime="audio/wav",
                key=f"download-{wav}",
            )
            sidecar = wav.with_suffix(wav.suffix + ".synthetic.json")
            if sidecar.exists():
                with st.expander(f"Metadaten: {sidecar.name}"):
                    st.json(json.loads(sidecar.read_text(encoding="utf-8")))
    report = output_dir / "report.md"
    if report.exists():
        with st.expander("Report"):
            st.markdown(report.read_text(encoding="utf-8"))


def _backend(name: str):
    if name == "mock":
        return MockBackend()
    if name == "xtts":
        return XTTSBackend()
    if name == "openvoice":
        return OpenVoiceBackend()
    raise ValueError(f"Unsupported backend: {name}")


def _backend_hint(name: str) -> str:
    if name == "xtts":
        return "XTTS laeuft bewusst in einer separaten Umgebung. Richte sie mit `bash scripts/setup_xtts_env.sh` ein oder setze `XTTS_PYTHON`."
    if name == "openvoice":
        return "OpenVoice benoetigt eine separate OpenVoice-V2 Installation plus konfigurierte Checkpoints."
    return "Mock erzeugt nur ein Test-WAV und benoetigt keine Modellinstallation."


def _safe_filename(name: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "-" for char in name).strip("-") or "upload"
