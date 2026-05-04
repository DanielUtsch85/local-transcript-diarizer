from __future__ import annotations

import json
import os
import signal
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

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


_PIPELINE_STEPS = [
    ("Audio", "WAV vorbereiten"),
    ("Schnitt", "Zielsprecher extrahieren"),
    ("Qualitaet", "VAD und Metriken"),
    ("Curation", "Akzeptiert/abgelehnt"),
    ("Referenzen", "Reference-Pack"),
    ("Synthese", "Sample erzeugen"),
]


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
        xtts_license_confirmed = False
        if backend == "xtts":
            xtts_license_confirmed = st.checkbox(
                "Coqui/XTTS Lizenzbedingungen sind fuer diesen Einsatz geprueft und akzeptiert",
                help=(
                    "XTTS-v2 fragt beim ersten Modell-Download nach einer Coqui-Lizenzbestaetigung. "
                    "Diese Checkbox setzt COQUI_TOS_AGREED=1 nur fuer den isolierten XTTS-Prozess."
                ),
            )
        text = st.text_area("Synthese-Text", value="Hallo, dies ist ein synthetisch erzeugter Testsatz.")
        consent_confirmed = st.checkbox("Explizite Einwilligung des Zielsprechers liegt vor")
        run_synthesis = st.checkbox("Nach Reference-Pack direkt Synthese erzeugen", value=False)
        with st.expander("Pipeline-Parameter"):
            sample_rate = st.selectbox("Sample Rate", [24000, 16000], index=0)
            min_duration = st.number_input("Min. Segmentlaenge Sekunden", min_value=0.1, max_value=30.0, value=2.0)
            max_duration = st.number_input("Max. Segmentlaenge Sekunden", min_value=0.5, max_value=60.0, value=15.0)
            padding_ms = st.number_input("Padding ms", min_value=0, max_value=1000, value=100, step=25)
            target_total_sec = st.number_input("Ziel-Laenge Reference-Pack Sekunden", min_value=1, max_value=600, value=60)
            synthesis_timeout_min = st.number_input(
                "Synthese automatisch abbrechen nach Minuten",
                min_value=1,
                max_value=180,
                value=30,
                help=(
                    "Beendet haengende Synthese-Prozesse hart. Die vorher berechneten Pipeline-Werte "
                    "und die Fehleranalyse werden trotzdem als Voice-Feedback-CSV geschrieben."
                ),
            )
            reject_overlaps = st.checkbox(
                "Ueberlappungen mit anderen Sprechern ablehnen",
                value=True,
                help="Markiert Segmente als ungeeignet, wenn sich im Diarization-Export zur gleichen Zeit ein anderer Sprecher ueberschneidet.",
            )
        eligible_count = sum(
            1
            for segment in segments
            if segment.speaker == speaker and float(min_duration) <= segment.duration_sec <= float(max_duration)
        )
        st.caption(
            f"Geplanter Lauf: {eligible_count} Segmente fuer {speaker}. "
            "Die Bewertung kann je nach VAD und Segmentanzahl mehrere Minuten dauern."
        )
        submitted = st.form_submit_button("Referenzdaten bauen", type="primary")

    output_dir = data_dir / "output" / speaker
    _render_abort_control(output_dir)
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
            synthesis_timeout_sec=int(synthesis_timeout_min) * 60,
            xtts_license_confirmed=xtts_license_confirmed,
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
    synthesis_timeout_sec: int,
    xtts_license_confirmed: bool,
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
        started_at = time.monotonic()
        stepper = st.empty()
        overall_progress = st.progress(0, text="Start")
        stage_detail = st.empty()

        _render_pipeline_stepper(stepper, active_index=0)
        status.update(label="Audio vorbereiten")
        stage_detail.write("Audio wird vorbereitet.")
        prepare_audio(source_audio, prepared, sample_rate=sample_rate, mono=True)
        overall_progress.progress(10, text="Audio vorbereitet")

        _render_pipeline_stepper(stepper, active_index=1)
        status.update(label="Zielsprecher-Segmente schneiden")
        stage_detail.write("Diarization wird auf die vorbereitete WAV-Datei umgeschrieben.")
        rows_for_prepared = _rewrite_diarization_source(diarization_path, prepared, output_dir / "work" / "prepared_diarization.json")
        all_segments = load_segments(rows_for_prepared)
        selected = filter_segments(all_segments, speaker, min_duration_sec, max_duration_sec)
        stage_detail.write(f"{len(selected)} Segmente fuer {speaker} werden geschnitten.")
        segment_progress = st.progress(0, text=f"Segmente schneiden: 0/{len(selected)}")
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
            if index == 1 or index == len(selected) or index % 5 == 0:
                segment_progress.progress(
                    int(index / max(1, len(selected)) * 100),
                    text=f"Segmente schneiden: {index}/{len(selected)} ({_elapsed(started_at)})",
                )
        write_jsonl(manifests_dir / "raw_segments.jsonl", extracted)
        overall_progress.progress(35, text="Segmente geschnitten")

        _render_pipeline_stepper(stepper, active_index=2)
        status.update(label="Segmente bewerten")
        stage_detail.write("Segmente werden bewertet. Dieser Schritt nutzt VAD und ist meist der laengste Teil der Curation.")
        score_progress = st.progress(0, text=f"Qualitaet bewerten: 0/{len(extracted)}")
        scored = []
        for index, row in enumerate(extracted, start=1):
            scored.append(
                score_segment(
                    Path(row["file"]),
                    speaker=speaker,
                    quality_config=cfg["quality"],
                    overlap_sec=float(row.get("overlap_sec") or 0),
                    overlap_speakers=list(row.get("overlap_speakers") or []),
                )
            )
            if index == 1 or index == len(extracted) or index % 5 == 0:
                score_progress.progress(
                    int(index / max(1, len(extracted)) * 100),
                    text=f"Qualitaet bewerten: {index}/{len(extracted)} ({_elapsed(started_at)})",
                )
        write_jsonl(manifests_dir / "segments.jsonl", scored)
        overall_progress.progress(70, text="Segmente bewertet")

        _render_pipeline_stepper(stepper, active_index=3)
        status.update(label="Segmente kuratieren")
        stage_detail.write("Akzeptierte und abgelehnte Segmente werden getrennt.")
        accepted = [_copy_curated(row, clean_dir) for row in scored if row.get("accepted")]
        rejected = [_copy_curated(row, rejected_dir) for row in scored if not row.get("accepted")]
        write_jsonl(manifests_dir / "accepted.jsonl", accepted)
        write_jsonl(manifests_dir / "rejected.jsonl", rejected)
        _write_report(output_dir, accepted, rejected)
        overall_progress.progress(82, text="Curation geschrieben")

        _render_pipeline_stepper(stepper, active_index=4)
        status.update(label="Reference-Pack bauen")
        stage_detail.write("Reference-Pack wird gebaut.")
        reference = build_reference_pack(clean_dir, manifests_dir / "accepted.jsonl", output_dir / "voice_refs", target_total_sec=target_total_sec)
        overall_progress.progress(90, text="Reference-Pack gebaut")

        synth_metadata = None
        synthesis_error = None
        if run_synthesis:
            _render_pipeline_stepper(stepper, active_index=5)
            status.update(label="Synthese erzeugen")
            stage_detail.write(
                "Synthetisches Sample wird erzeugt und geloggt. "
                f"Haengende Synthese wird nach {synthesis_timeout_sec // 60} Minuten abgebrochen."
            )
            synthesis_progress = st.progress(
                0,
                text=f"Synthese gestartet. Automatischer Abbruch in {_format_countdown(synthesis_timeout_sec)}.",
            )

            def update_synthesis_countdown(elapsed_sec: int, remaining_sec: int) -> None:
                percent = int(min(100, max(0, elapsed_sec / max(1, synthesis_timeout_sec) * 100)))
                synthesis_progress.progress(
                    percent,
                    text=(
                        "Synthese laeuft. "
                        f"Automatischer Abbruch in {_format_countdown(remaining_sec)}."
                    ),
                )

            validate_consent(consent_confirmed)
            output_wav = output_dir / "generated_samples" / f"sample_{backend}_001.wav"
            reference_files = sorted((output_dir / "voice_refs").glob("ref_*.wav"))
            if not reference_files:
                synthesis_error = "Keine Reference-Clips vorhanden; Synthese wurde nicht gestartet."
            else:
                try:
                    result = _backend(
                        backend,
                        timeout_sec=synthesis_timeout_sec,
                        pid_file=output_dir / "work" / "synthesis.pid",
                        xtts_license_confirmed=xtts_license_confirmed,
                        progress_callback=update_synthesis_countdown,
                    ).synthesize(
                        text,
                        reference_files,
                        output_wav,
                        language=language,
                    )
                    synth_metadata = synthesis_metadata(speaker, backend, text, language, reference_files, result, consent_confirmed)
                    write_synthesis_log(manifests_dir / "synthesis_runs.jsonl", synth_metadata, result)
                except (RuntimeError, NotImplementedError, ValueError) as exc:
                    synthesis_error = str(exc)
                finally:
                    if synthesis_error:
                        synthesis_progress.progress(100, text="Synthese abgebrochen oder fehlgeschlagen.")
        overall_progress.progress(96, text="Feedback-CSV schreiben")

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
                "synthesis_timeout_sec": synthesis_timeout_sec,
                "xtts_license_confirmed": xtts_license_confirmed,
            },
            raw_rows=extracted,
            scored_rows=scored,
            accepted_rows=accepted,
            rejected_rows=rejected,
            reference_metadata=reference,
            synthesis_metadata=synth_metadata,
            synthesis_error=synthesis_error,
        )
        feedback_path = manifests_dir / "voice_feedback.csv"
        feedback_path.write_text(feedback_csv, encoding="utf-8")
        overall_progress.progress(100, text=f"Fertig nach {_elapsed(started_at)}")

        message = (
            f"Fertig: {len(extracted)} roh, {len(accepted)} akzeptiert, {len(rejected)} abgelehnt, "
            f"{len(reference.get('refs', []))} Referenzen."
        )
        if synthesis_error:
            _render_pipeline_stepper(stepper, active_index=5, failed_index=5)
            status.update(label="Reference-Pack fertig, Synthese fehlgeschlagen", state="error")
            st.warning(message)
            st.error(f"Synthese fehlgeschlagen: {synthesis_error}")
            st.info(_backend_hint(backend))
            _render_feedback_download(
                feedback_csv,
                file_name=f"{speaker}-voice-feedback-error.csv",
                label="Fehleranalyse als Voice-Feedback-CSV herunterladen",
            )
        else:
            _render_pipeline_stepper(stepper, active_index=5, completed=True)
            status.update(label="Voice-Pipeline fertig", state="complete")
            st.success(message)


def _elapsed(started_at: float) -> str:
    elapsed = max(0, int(time.monotonic() - started_at))
    minutes, seconds = divmod(elapsed, 60)
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def _format_countdown(total_sec: int) -> str:
    total_sec = max(0, int(total_sec))
    minutes, seconds = divmod(total_sec, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def _render_pipeline_stepper(
    container: Any,
    *,
    active_index: int,
    failed_index: int | None = None,
    completed: bool = False,
) -> None:
    items = []
    for index, (title, detail) in enumerate(_PIPELINE_STEPS):
        if failed_index == index:
            state = "failed"
            state_label = "Fehler"
        elif completed or index < active_index:
            state = "done"
            state_label = "Fertig"
        elif index == active_index:
            state = "active"
            state_label = "Laeuft"
        else:
            state = "pending"
            state_label = "Wartet"
        items.append(
            f"""
            <div class="vp-step vp-step-{state}">
              <div class="vp-step-index">{index + 1}</div>
              <div class="vp-step-copy">
                <div class="vp-step-title">{title}</div>
                <div class="vp-step-detail">{detail}</div>
                <div class="vp-step-state">{state_label}</div>
              </div>
            </div>
            """
        )
    container.markdown(
        f"""
        <style>
          .vp-stepper {{
            display: grid;
            grid-template-columns: repeat(6, minmax(0, 1fr));
            gap: 8px;
            margin: 8px 0 14px;
          }}
          .vp-step {{
            border: 1px solid #d7dde7;
            border-radius: 8px;
            padding: 10px;
            min-height: 104px;
            background: #f8fafc;
          }}
          .vp-step-index {{
            width: 28px;
            height: 28px;
            border-radius: 999px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            margin-bottom: 8px;
            color: #475569;
            background: #e2e8f0;
          }}
          .vp-step-title {{
            font-weight: 700;
            color: #0f172a;
            line-height: 1.2;
          }}
          .vp-step-detail {{
            color: #64748b;
            font-size: 12px;
            line-height: 1.25;
            margin-top: 4px;
          }}
          .vp-step-state {{
            color: #475569;
            font-size: 11px;
            font-weight: 700;
            margin-top: 8px;
            text-transform: uppercase;
          }}
          .vp-step-done {{
            border-color: #9bd4b5;
            background: #f1fbf5;
          }}
          .vp-step-done .vp-step-index {{
            color: #065f46;
            background: #bbf7d0;
          }}
          .vp-step-active {{
            border-color: #60a5fa;
            background: #eff6ff;
            box-shadow: inset 0 0 0 1px #bfdbfe;
          }}
          .vp-step-active .vp-step-index {{
            color: #ffffff;
            background: #2563eb;
          }}
          .vp-step-failed {{
            border-color: #fca5a5;
            background: #fff1f2;
          }}
          .vp-step-failed .vp-step-index {{
            color: #ffffff;
            background: #dc2626;
          }}
          @media (max-width: 900px) {{
            .vp-stepper {{
              grid-template-columns: repeat(2, minmax(0, 1fr));
            }}
          }}
        </style>
        <div class="vp-stepper">{''.join(items)}</div>
        """,
        unsafe_allow_html=True,
    )


def _render_feedback_download(feedback_csv: str, *, file_name: str, label: str) -> None:
    st.download_button(
        label,
        data=feedback_csv.encode("utf-8"),
        file_name=file_name,
        mime="text/csv",
        key=f"download-{file_name}-{time.monotonic()}",
    )


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


def _render_abort_control(output_dir: Path) -> None:
    pid_file = output_dir / "work" / "synthesis.pid"
    if not pid_file.exists():
        return
    raw_pid = pid_file.read_text(encoding="utf-8").strip()
    if not raw_pid.isdigit():
        pid_file.unlink(missing_ok=True)
        return
    st.warning(f"XTTS-Synthese laeuft noch mit Prozess-ID {raw_pid}.")
    if st.button("XTTS-Synthese abbrechen", type="secondary"):
        try:
            os.kill(int(raw_pid), signal.SIGTERM)
            st.success("Abbruchsignal gesendet. Der Lauf schreibt danach die Fehleranalyse.")
        except ProcessLookupError:
            st.info("Der XTTS-Prozess laeuft nicht mehr.")
        finally:
            pid_file.unlink(missing_ok=True)


def _backend(
    name: str,
    timeout_sec: int = 30 * 60,
    pid_file: Path | None = None,
    xtts_license_confirmed: bool = False,
    progress_callback: Callable[[int, int], None] | None = None,
):
    if name == "mock":
        return MockBackend()
    if name == "xtts":
        return XTTSBackend(
            timeout_sec=timeout_sec,
            pid_file=pid_file,
            license_confirmed=xtts_license_confirmed,
            progress_callback=progress_callback,
        )
    if name == "openvoice":
        return OpenVoiceBackend()
    raise ValueError(f"Unsupported backend: {name}")


def _backend_hint(name: str) -> str:
    if name == "xtts":
        return (
            "XTTS laeuft bewusst in einer separaten Umgebung. Zusaetzlich muessen die Coqui/XTTS-Lizenzbedingungen "
            "fuer deinen Einsatzzweck geprueft und bestaetigt werden."
        )
    if name == "openvoice":
        return "OpenVoice benoetigt eine separate OpenVoice-V2 Installation plus konfigurierte Checkpoints."
    return "Mock erzeugt nur ein Test-WAV und benoetigt keine Modellinstallation."


def _safe_filename(name: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "-" for char in name).strip("-") or "upload"
