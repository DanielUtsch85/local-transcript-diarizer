from __future__ import annotations

import json
import os
import signal
import shutil
import time
from html import escape
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import streamlit as st

from transcript_mvp.models import SpeakerSegment, TranscriptSegment


_PIPELINE_STEPS = [
    ("Audio", "WAV vorbereiten"),
    ("Schnitt", "Zielsprecher extrahieren"),
    ("Qualitaet", "VAD und Metriken"),
    ("Curation", "Akzeptiert/abgelehnt"),
    ("Referenzen", "Reference-Pack"),
    ("Synthese", "Sample erzeugen"),
]

_dialog = getattr(st, "dialog", getattr(st, "experimental_dialog", None))
if _dialog is None:
    def _dialog(_title: str):
        def decorate(func):
            return func

        return decorate


def render_voice_pipeline_page(data_dir: Path) -> None:
    st.header("Voice Reference & Synthese")
    st.caption("Erstellt aus vorhandenen Sprechersegmenten kuratierte Referenzclips und synthetisch markierte Samples.")

    source_audio = _resolve_source_audio(data_dir)
    diarization_path = _resolve_diarization_source(data_dir, source_audio)

    if source_audio is None or diarization_path is None:
        st.info("Voraussetzung: Audio-Datei und Sprechersegmente. Nutze die Transkriptionsseite zuerst oder lade beides hier hoch.")
        return

    try:
        from voice_pipeline.segment_loader import load_segments

        segments = load_segments(diarization_path)
    except ValueError as exc:
        st.error(f"Diarization konnte nicht geladen werden: {exc}")
        return

    speakers = sorted({segment.speaker for segment in segments})
    if not speakers:
        st.warning("Keine Sprechersegmente gefunden.")
        return

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

    run_synthesis = st.checkbox("Nach Reference-Pack direkt Synthese erzeugen", value=False)

    with st.form("voice-pipeline-settings"):
        speaker = st.selectbox("Zielsprecher", speakers)
        language_options = ["en", "es", "fr", "zh", "jp", "kr"] if backend == "openvoice" else ["de", "en"]
        language = st.selectbox(
            "Sprache",
            language_options,
            index=0,
            help=(
                "OpenVoice V2 nutzt MeloTTS als Basisstimme und unterstuetzt hier EN/ES/FR/ZH/JP/KR. "
                "Deutsch ist fuer OpenVoice erst mit externer Basis-TTS-Verdrahtung sinnvoll."
                if backend == "openvoice"
                else "Sprache fuer das Synthese-Backend."
            ),
        )
        text = st.text_area(
            "Synthese-Text",
            value=(
                "Manchmal reicht ein einziger Moment, um alles zu verändern. "
                "Die Frage ist nicht ob, sondern wann – und wie wir dann reagieren. "
                "Ich glaube, dass wir alle nach denselben Dingen suchen: Verbindung, "
                "Bedeutung und ein bisschen mehr Leichtigkeit."
            ),
        )
        consent_confirmed = st.checkbox("Explizite Einwilligung des Zielsprechers liegt vor")
        with st.expander("Pipeline-Parameter"):
            sample_rate = st.selectbox(
                "Sample Rate",
                [24000, 16000],
                index=0,
                help=(
                    "Bestimmt die technische Aufloesung der Referenzclips. 24 kHz erhaelt mehr Hoehen "
                    "und ist fuer XTTS meist sinnvoller; 16 kHz ist kleiner und schneller, kann aber etwas weniger brillant klingen."
                ),
            )
            min_duration = st.number_input(
                "Min. Segmentlaenge Sekunden",
                min_value=0.1,
                max_value=30.0,
                value=2.0,
                help=(
                    "Sehr kurze Clips enthalten oft zu wenig Stimmeigenschaften. Hoehere Werte koennen die "
                    "Stimmstabilitaet verbessern, reduzieren aber die Anzahl nutzbarer Referenzen."
                ),
            )
            max_duration = st.number_input(
                "Max. Segmentlaenge Sekunden",
                min_value=0.5,
                max_value=60.0,
                value=15.0,
                help=(
                    "Sehr lange Clips enthalten haeufig Themenwechsel, Atmer, Nebengeraeusche oder mehrere Prosodiephasen. "
                    "Kuerzere Maximalwerte machen Referenzen homogener, koennen aber gutes Material ausschliessen."
                ),
            )
            padding_ms = st.number_input(
                "Padding ms",
                min_value=0,
                max_value=1000,
                value=100,
                step=25,
                help=(
                    "Fuegt vor und nach jedem Segment etwas Kontext hinzu. Etwas Padding verhindert abgeschnittene "
                    "Anlaute; zu viel Padding kann Stille, Fremdstimmen oder Raumgeraeusche in die Referenzen bringen."
                ),
            )
            target_total_sec = st.number_input(
                "Ziel-Laenge Reference-Pack Sekunden",
                min_value=1,
                max_value=600,
                value=60,
                help=(
                    "Legt fest, wie viel akzeptiertes Referenzmaterial fuer die Stimme gesammelt wird. Mehr Material "
                    "kann Aehnlichkeit und Robustheit verbessern, erhoeht aber Laufzeit und kann bei gemischter Qualitaet auch stoeren."
                ),
            )
            preferred_min_duration = st.number_input(
                "Bevorzugte Ref-Clip Mindestlaenge Sekunden",
                min_value=0.5,
                max_value=60.0,
                value=4.0,
                help=(
                    "Beeinflusst das Ranking fuer das Reference-Pack. Clips ab dieser Laenge werden bevorzugt, "
                    "weil sie meist genug Klangfarbe, Rhythmus und Artikulation enthalten."
                ),
            )
            preferred_max_duration = st.number_input(
                "Bevorzugte Ref-Clip Maximallaenge Sekunden",
                min_value=0.5,
                max_value=60.0,
                value=12.0,
                help=(
                    "Beeinflusst das Ranking fuer das Reference-Pack. Clips bis zu dieser Laenge werden bevorzugt, "
                    "weil sie meist fokussierter sind und weniger Nebengeräusche oder Sprecherwechsel enthalten."
                ),
            )
            min_speech_ratio = st.slider(
                "Min. Sprachanteil",
                min_value=0.0,
                max_value=1.0,
                value=0.55,
                step=0.01,
                help=(
                    "Mindestanteil erkannter Sprache im Segment. Hoehere Werte entfernen Pausen und Stille strenger; "
                    "das kann Referenzen sauberer machen, aber natuerliche Sprechpausen verlieren."
                ),
            )
            max_clipping_ratio = st.number_input(
                "Max. Clipping-Anteil",
                min_value=0.0,
                max_value=0.1,
                value=0.001,
                step=0.0005,
                format="%.4f",
                help=(
                    "Erlaubter Anteil uebersteuerter Samples. Niedrigere Werte schuetzen vor kratziger, verzerrter Synthese; "
                    "zu strenge Werte koennen laute, sonst brauchbare Segmente aussortieren."
                ),
            )
            min_rms_db = st.number_input(
                "Min. Lautheit RMS dB",
                min_value=-80.0,
                max_value=0.0,
                value=-35.0,
                step=1.0,
                help=(
                    "Sortiert sehr leise Segmente aus. Zu leise Referenzen enthalten oft mehr Rauschen als Stimme; "
                    "ein hoeherer Wert macht die Auswahl sauberer, kann aber leise gesprochene gute Passagen entfernen."
                ),
            )
            max_peak_db = st.number_input(
                "Max. Peak dB",
                min_value=-24.0,
                max_value=0.0,
                value=-0.5,
                step=0.1,
                help=(
                    "Obergrenze fuer Spitzenpegel. Werte nahe 0 dB lassen laute Clips zu; niedrigere Werte verlangen "
                    "mehr Headroom und koennen harte, ueberlaute Referenzen vermeiden."
                ),
            )
            denoise_references = st.checkbox(
                "Reference-Clips entrauschen",
                value=False,
                help=(
                    "Erzeugt vor dem Reference-Pack eine entrauschte Kopie der akzeptierten Clips. Das kann konstantes "
                    "Mikrofonknistern reduzieren, kann bei zu starker Einstellung aber Klangfarbe und Sprecheridentitaet verfälschen."
                ),
            )
            denoise_strength_db = st.slider(
                "Denoise-Staerke dB",
                min_value=0.0,
                max_value=30.0,
                value=12.0,
                step=1.0,
                help=(
                    "Staerke des ffmpeg-afftdn Filters. 8-12 dB ist ein konservativer Start; hoehere Werte entfernen mehr Rauschen, "
                    "koennen aber metallische Artefakte erzeugen."
                ),
                disabled=not denoise_references,
            )
            synthesis_timeout_min = st.number_input(
                "Synthese automatisch abbrechen nach Minuten",
                min_value=1,
                max_value=180,
                value=15 if backend == "openvoice" else 30,
                help=(
                    "Hat keinen Klangvorteil, schuetzt aber vor haengenden Modellprozessen. Wenn der Abbruch greift, "
                    "bleiben Referenzdaten und Fehleranalyse erhalten, aber es entsteht kein vollstaendiges Sample."
                ),
            )
            reject_overlaps = st.checkbox(
                "Ueberlappungen mit anderen Sprechern ablehnen",
                value=True,
                help=(
                    "Entfernt Passagen, in denen andere Sprecher gleichzeitig erkannt wurden. Das schuetzt die "
                    "Stimmidentitaet der synthetischen Stimme, kann aber bei ungenauer Diarization brauchbare Clips verlieren."
                ),
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
        if preferred_min_duration > preferred_max_duration:
            st.warning("Die bevorzugte Mindestlaenge sollte nicht groesser als die bevorzugte Maximallaenge sein.")
        submitted = st.form_submit_button("Referenzdaten bauen", type="primary")

    output_dir = data_dir / "output" / speaker
    _render_abort_control(output_dir)
    if submitted:
        st.session_state[_current_sample_key(output_dir)] = []
        if min_duration > max_duration:
            st.error("Min. Segmentlaenge darf nicht groesser als Max. Segmentlaenge sein.")
            return
        if preferred_min_duration > preferred_max_duration:
            st.error("Bevorzugte Ref-Clip Mindestlaenge darf nicht groesser als die Maximallaenge sein.")
            return
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
            preferred_min_duration_sec=float(preferred_min_duration),
            preferred_max_duration_sec=float(preferred_max_duration),
            min_speech_ratio=float(min_speech_ratio),
            max_clipping_ratio=float(max_clipping_ratio),
            min_rms_db=float(min_rms_db),
            max_peak_db=float(max_peak_db),
            denoise_references=denoise_references,
            denoise_strength_db=float(denoise_strength_db),
            reject_overlaps=reject_overlaps,
            run_synthesis=run_synthesis,
            backend=backend,
            synthesis_timeout_sec=int(synthesis_timeout_min) * 60,
            xtts_license_confirmed=xtts_license_confirmed,
            text=text,
            language=language,
            consent_confirmed=consent_confirmed,
        )

    _render_existing_outputs(
        output_dir,
        backend=backend,
        text=text,
        language=language,
        consent_confirmed=consent_confirmed,
        synthesis_timeout_sec=int(synthesis_timeout_min) * 60,
        xtts_license_confirmed=xtts_license_confirmed,
        denoise_references=denoise_references,
        denoise_strength_db=float(denoise_strength_db),
    )


def _resolve_source_audio(data_dir: Path) -> Path | None:
    current = st.session_state.get("last_audio_path")
    if current and Path(current).exists():
        st.success(f"Aktuelle Audio-Datei aus der Transkription: `{current}`")
        return Path(current)

    existing_audio = _recent_files(data_dir / "uploads", {".mp3", ".wav", ".m4a", ".mp4", ".flac"})
    if existing_audio:
        options = ["Neue Audio-Datei hochladen"] + [str(path) for path in existing_audio]
        selected = st.selectbox("Vorhandene Audio-Datei verwenden", options, index=1)
        if selected != options[0]:
            st.session_state.last_audio_path = selected
            st.success(f"Audio-Datei geladen: `{selected}`")
            return Path(selected)

    uploaded = st.file_uploader(
        "Audio-Datei fuer Voice-Pipeline laden",
        type=["mp3", "wav", "m4a", "mp4", "flac"],
        key="voice-audio-upload",
    )
    if uploaded is None:
        return None
    target = _save_uploaded_once(uploaded, data_dir / "uploads", "voice_audio")
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

    existing_diarization = _recent_files(data_dir / "work", {".json", ".jsonl"}, name_contains="diarization")
    if existing_diarization:
        options = ["Neue Diarization-Datei hochladen"] + [str(path) for path in existing_diarization]
        selected = st.selectbox("Vorhandene Diarization verwenden", options, index=1)
        if selected != options[0]:
            return Path(selected)

    uploaded = st.file_uploader(
        "Oder Diarization JSON/JSONL laden",
        type=["json", "jsonl"],
        key="voice-diarization-upload",
    )
    if uploaded is None:
        return None
    return _save_uploaded_once(uploaded, data_dir / "work", "voice_diarization")


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
    preferred_min_duration_sec: float,
    preferred_max_duration_sec: float,
    min_speech_ratio: float,
    max_clipping_ratio: float,
    min_rms_db: float,
    max_peak_db: float,
    denoise_references: bool,
    denoise_strength_db: float,
    reject_overlaps: bool,
    run_synthesis: bool,
    backend: str,
    synthesis_timeout_sec: int,
    xtts_license_confirmed: bool,
    text: str,
    language: str,
    consent_confirmed: bool,
) -> None:
    from voice_pipeline.audio_io import cut_segment, prepare_audio
    from voice_pipeline.config import load_config
    from voice_pipeline.feedback import build_voice_feedback_csv
    from voice_pipeline.logging_utils import write_json, write_jsonl
    from voice_pipeline.metadata import synthesis_metadata, validate_consent, write_synthesis_log
    from voice_pipeline.quality import relax_speech_ratio_rejections, score_segment, segment_overlap
    from voice_pipeline.reference_pack import build_reference_pack
    from voice_pipeline.segment_loader import filter_segments, load_segments, segment_filename

    cfg = load_config()
    cfg["quality"]["min_duration_sec"] = min_duration_sec
    cfg["quality"]["max_duration_sec"] = max_duration_sec
    cfg["quality"]["min_speech_ratio"] = min_speech_ratio
    cfg["quality"]["max_clipping_ratio"] = max_clipping_ratio
    cfg["quality"]["min_rms_db"] = min_rms_db
    cfg["quality"]["max_peak_db"] = max_peak_db
    cfg["quality"]["max_overlap_sec"] = 0.0 if reject_overlaps else 999999.0
    prepared = output_dir / "work" / f"{source_audio.stem}_{sample_rate}.wav"
    raw_dir = output_dir / "raw_segments"
    clean_dir = output_dir / "clean_segments"
    rejected_dir = output_dir / "rejected_segments"
    manifests_dir = output_dir / "manifests"

    started_at = time.monotonic()
    stepper = st.empty()
    current_step = {"index": None, "started_at": started_at}
    step_durations: dict[int, float] = {}

    def record_current_step(now: float) -> None:
        current_index = current_step["index"]
        if current_index is None or current_index in step_durations:
            return
        step_durations[int(current_index)] = max(0.0, now - float(current_step["started_at"]))

    def show_step(
        index: int | None,
        *,
        failed_index: int | None = None,
        completed: bool = False,
        skipped_indices: set[int] | None = None,
    ) -> None:
        now = time.monotonic()
        if index is not None and current_step["index"] != index:
            record_current_step(now)
            current_step["index"] = index
            current_step["started_at"] = now
        if completed or failed_index is not None:
            record_current_step(now)
        _render_pipeline_stepper(
            stepper,
            active_index=index,
            failed_index=failed_index,
            completed=completed,
            overall_started_at=started_at,
            step_started_at=current_step["started_at"] if index is not None else None,
            step_durations=step_durations,
            skipped_indices=skipped_indices,
            run_synthesis=run_synthesis,
        )

    show_step(0)

    with st.status("Voice-Pipeline laeuft...", expanded=True) as status:
        overall_progress = st.progress(0, text="Start")
        stage_detail = st.empty()

        status.update(label="Audio vorbereiten")
        stage_detail.write("Audio wird vorbereitet.")
        prepare_audio(source_audio, prepared, sample_rate=sample_rate, mono=True)
        overall_progress.progress(10, text="Audio vorbereitet")

        show_step(1)
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
                show_step(1)
                segment_progress.progress(
                    int(index / max(1, len(selected)) * 100),
                    text=f"Segmente schneiden: {index}/{len(selected)} ({_elapsed(started_at)})",
                )
        write_jsonl(manifests_dir / "raw_segments.jsonl", extracted)
        overall_progress.progress(35, text="Segmente geschnitten")

        show_step(2)
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
                show_step(2)
                score_progress.progress(
                    int(index / max(1, len(extracted)) * 100),
                    text=f"Qualitaet bewerten: {index}/{len(extracted)} ({_elapsed(started_at)})",
                )
        scored = relax_speech_ratio_rejections(
            scored,
            fallback_min_speech_ratio=float(cfg["quality"].get("fallback_min_speech_ratio", 0.45)),
            configured_min_speech_ratio=float(cfg["quality"].get("min_speech_ratio", min_speech_ratio)),
        )
        write_jsonl(manifests_dir / "segments.jsonl", scored)
        overall_progress.progress(70, text="Segmente bewertet")

        show_step(3)
        status.update(label="Segmente kuratieren")
        stage_detail.write("Akzeptierte und abgelehnte Segmente werden getrennt.")
        accepted = [_copy_curated(row, clean_dir) for row in scored if row.get("accepted")]
        rejected = [_copy_curated(row, rejected_dir) for row in scored if not row.get("accepted")]
        write_jsonl(manifests_dir / "accepted.jsonl", accepted)
        write_jsonl(manifests_dir / "rejected.jsonl", rejected)
        _write_report(output_dir, accepted, rejected)
        overall_progress.progress(82, text="Curation geschrieben")

        show_step(4)
        status.update(label="Reference-Pack bauen")
        stage_detail.write("Reference-Pack wird gebaut.")
        reference_rows = _prepare_reference_rows(
            accepted,
            output_dir=output_dir,
            denoise_references=denoise_references,
            denoise_strength_db=denoise_strength_db,
        )
        reference = build_reference_pack(
            output_dir / ("denoised_segments" if denoise_references else "clean_segments"),
            _write_reference_source_manifest(manifests_dir / "reference_source.jsonl", reference_rows),
            output_dir / "voice_refs",
            target_total_sec=target_total_sec,
            preferred_min_duration_sec=preferred_min_duration_sec,
            preferred_max_duration_sec=preferred_max_duration_sec,
        )
        reference["denoise_enabled"] = denoise_references
        reference["denoise_strength_db"] = denoise_strength_db if denoise_references else 0.0
        write_json(output_dir / "voice_refs" / "reference_pack.json", reference)
        overall_progress.progress(90, text="Reference-Pack gebaut")

        synth_metadata = None
        synthesis_error = None
        if run_synthesis:
            show_step(5)
            status.update(label="Synthese erzeugen")
            stage_detail.write(
                "Synthetisches Sample wird erzeugt und geloggt. "
                f"Haengende Synthese wird nach {synthesis_timeout_sec // 60} Minuten abgebrochen."
            )
            synthesis_progress = st.progress(
                0,
                text=f"Synthese gestartet. Automatischer Abbruch in {_format_countdown(synthesis_timeout_sec)}.",
            )
            synthesis_log_box = st.empty()
            append_synthesis_log = _make_synthesis_log_callback(synthesis_log_box)

            def update_synthesis_countdown(elapsed_sec: int, remaining_sec: int) -> None:
                show_step(5)
                percent = int(min(100, max(0, elapsed_sec / max(1, synthesis_timeout_sec) * 100)))
                synthesis_progress.progress(
                    percent,
                    text=(
                        "Synthese laeuft. "
                        f"Automatischer Abbruch in {_format_countdown(remaining_sec)}."
                    ),
                )

            validate_consent(consent_confirmed)
            output_wav = _next_synthesis_output(output_dir, backend)
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
                        log_callback=append_synthesis_log,
                    ).synthesize(
                        text,
                        reference_files,
                        output_wav,
                        language=language,
                    )
                    synth_metadata = synthesis_metadata(speaker, backend, text, language, reference_files, result, consent_confirmed)
                    write_synthesis_log(manifests_dir / "synthesis_runs.jsonl", synth_metadata, result)
                    current_samples = st.session_state.setdefault(_current_sample_key(output_dir), [])
                    if str(result) not in current_samples:
                        current_samples.append(str(result))
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
                "preferred_min_duration_sec": preferred_min_duration_sec,
                "preferred_max_duration_sec": preferred_max_duration_sec,
                "min_speech_ratio": min_speech_ratio,
                "max_clipping_ratio": max_clipping_ratio,
                "min_rms_db": min_rms_db,
                "max_peak_db": max_peak_db,
                "denoise_references": denoise_references,
                "denoise_strength_db": denoise_strength_db if denoise_references else 0.0,
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
            show_step(5, failed_index=5)
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
            show_step(5 if run_synthesis else 4, completed=True, skipped_indices=set() if run_synthesis else {5})
            status.update(label="Voice-Pipeline fertig", state="complete")
            st.success(message)


def _elapsed(started_at: float) -> str:
    return _format_duration(time.monotonic() - started_at)


def _format_duration(duration_sec: float) -> str:
    elapsed = max(0, int(duration_sec))
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


def _make_synthesis_log_callback(container: Any, max_lines: int = 80) -> Callable[[str], None]:
    lines: list[str] = []

    def append(line: str) -> None:
        lines.append(line)
        container.code("\n".join(lines[-max_lines:]), language="text")

    return append


def _render_pipeline_stepper(
    container: Any,
    *,
    active_index: int | None,
    failed_index: int | None = None,
    completed: bool = False,
    overall_started_at: float | None = None,
    step_started_at: float | None = None,
    step_durations: dict[int, float] | None = None,
    skipped_indices: set[int] | None = None,
    run_synthesis: bool = True,
) -> None:
    skipped_indices = skipped_indices or set()
    step_durations = step_durations or {}
    if not run_synthesis and not completed:
        skipped_indices = set(skipped_indices) | {5}
    overall_label = _elapsed(overall_started_at) if overall_started_at is not None else "bereit"
    step_label = _elapsed(step_started_at) if step_started_at is not None else "noch nicht gestartet"
    items: list[str] = []
    for index, (title, detail) in enumerate(_PIPELINE_STEPS):
        if index in skipped_indices:
            state = "skipped"
            state_label = "Uebersprungen"
            timing = ""
        elif failed_index == index:
            state = "failed"
            state_label = "Fehler"
            timing = (
                f"Dauer {_format_duration(step_durations[index])}"
                if index in step_durations
                else (_elapsed(step_started_at) if step_started_at is not None else "")
            )
        elif completed or (active_index is not None and index < active_index):
            state = "done"
            state_label = "Fertig"
        elif index == active_index:
            state = "active"
            state_label = "Laeuft"
        else:
            state = "pending"
            state_label = "Wartet"
            timing = ""
        if state == "active" and index == active_index and step_started_at is not None:
            timing = f"Schrittzeit {_elapsed(step_started_at)}"
        elif state == "done":
            timing = (
                f"Dauer {_format_duration(step_durations[index])}"
                if index in step_durations
                else "abgeschlossen"
            )
        elif state == "pending":
            timing = "ausstehend"
        items.append(
            '<div class="vp-process-item">'
            f'<div class="vp-process-card vp-step-{state}">'
            f'<div class="vp-step-index">{index + 1}</div>'
            '<div class="vp-step-copy">'
            f'<div class="vp-step-title">{escape(title)}</div>'
            f'<div class="vp-step-detail">{escape(detail)}</div>'
            '<div class="vp-step-meta">'
            f'<span>{escape(state_label)}</span>'
            f'<span>{escape(timing)}</span>'
            "</div>"
            "</div>"
            "</div>"
            "</div>"
        )
    style = (
        "<style>"
        ".vp-process-shell{border:1px solid #dbe3ee;border-radius:8px;padding:14px;margin:14px 0 16px;background:#fff;}"
        ".vp-process-head{display:flex;align-items:baseline;justify-content:space-between;gap:12px;margin-bottom:14px;}"
        ".vp-process-title{color:#0f172a;font-size:15px;font-weight:700;}"
        ".vp-process-time{display:flex;gap:10px;flex-wrap:wrap;color:#475569;font-size:12px;}"
        ".vp-process-time span{border:1px solid #e2e8f0;border-radius:999px;padding:4px 9px;background:#f8fafc;}"
        ".vp-stepper{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:0;align-items:stretch;margin:0;}"
        ".vp-process-item{position:relative;padding:0 6px;}"
        ".vp-process-item:not(:last-child)::after{content:'';position:absolute;top:23px;right:-10px;width:20px;height:2px;background:#cbd5e1;z-index:1;}"
        ".vp-process-card{position:relative;z-index:2;border:1px solid #d7dde7;border-radius:8px;padding:10px;min-height:128px;background:#f8fafc;}"
        ".vp-step-index{width:28px;height:28px;border-radius:999px;display:flex;align-items:center;justify-content:center;font-weight:700;margin-bottom:8px;color:#475569;background:#e2e8f0;}"
        ".vp-step-title{font-weight:700;color:#0f172a;line-height:1.2;}"
        ".vp-step-detail{color:#64748b;font-size:12px;line-height:1.25;margin-top:4px;}"
        ".vp-step-meta{display:flex;flex-direction:column;gap:2px;color:#475569;font-size:11px;font-weight:700;margin-top:8px;text-transform:uppercase;}"
        ".vp-step-done{border-color:#9bd4b5;background:#f1fbf5;}"
        ".vp-step-done .vp-step-index{color:#065f46;background:#bbf7d0;}"
        ".vp-step-active{border-color:#60a5fa;background:#eff6ff;box-shadow:inset 0 0 0 1px #bfdbfe;}"
        ".vp-step-active .vp-step-index{color:#fff;background:#2563eb;}"
        ".vp-step-failed{border-color:#fca5a5;background:#fff1f2;}"
        ".vp-step-failed .vp-step-index{color:#fff;background:#dc2626;}"
        ".vp-step-skipped{border-color:#e2e8f0;background:#f8fafc;}"
        ".vp-step-skipped .vp-step-index{color:#64748b;background:#e2e8f0;}"
        "@media (max-width:900px){.vp-stepper{grid-template-columns:repeat(2,minmax(0,1fr));gap:8px;}.vp-process-item:not(:last-child)::after{display:none;}.vp-process-head{align-items:flex-start;flex-direction:column;}}"
        "</style>"
    )
    html = (
        f"{style}<div class=\"vp-process-shell\"><div class=\"vp-process-head\">"
        "<div class=\"vp-process-title\">Voice-Pipeline Prozess</div>"
        f"<div class=\"vp-process-time\"><span>Gesamtzeit {escape(overall_label)}</span>"
        f"<span>Aktueller Schritt {escape(step_label)}</span></div></div>"
        f"<div class=\"vp-stepper\">{''.join(items)}</div></div>"
    )
    container.markdown(html, unsafe_allow_html=True)


def _render_feedback_download(feedback_csv: str, *, file_name: str, label: str) -> None:
    st.download_button(
        label,
        data=feedback_csv.encode("utf-8"),
        file_name=file_name,
        mime="text/csv",
        key=f"download-{file_name}-{time.monotonic()}",
    )


def _rewrite_diarization_source(input_path: Path, prepared_audio: Path, output_path: Path) -> Path:
    from voice_pipeline.segment_loader import load_segments

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


def _render_existing_outputs(
    output_dir: Path,
    *,
    backend: str,
    text: str,
    language: str,
    consent_confirmed: bool,
    synthesis_timeout_sec: int,
    xtts_license_confirmed: bool,
    denoise_references: bool,
    denoise_strength_db: float,
) -> None:
    manifests = output_dir / "manifests"
    accepted_path = manifests / "accepted.jsonl"
    rejected_path = manifests / "rejected.jsonl"
    if not accepted_path.exists() and not rejected_path.exists():
        return
    from voice_pipeline.logging_utils import read_jsonl

    generated = _current_session_generated(output_dir)
    st.divider()
    st.subheader("Aktueller Pipeline-Stand")
    accepted = read_jsonl(accepted_path) if accepted_path.exists() else []
    rejected = read_jsonl(rejected_path) if rejected_path.exists() else []
    cols = st.columns(4)
    cols[0].metric("Akzeptiert", len(accepted))
    cols[1].metric("Abgelehnt", len(rejected))
    cols[2].metric("Ref-Clips", len(list((output_dir / "voice_refs").glob("ref_*.wav"))))
    cols[3].metric("Samples", len(generated))

    feedback = manifests / "voice_feedback.csv"
    action_cols = st.columns([1, 1, 2])
    if feedback.exists():
        action_cols[0].download_button(
            "Voice-Feedback-CSV",
            data=feedback.read_bytes(),
            file_name=f"{output_dir.name}-voice-feedback.csv",
            mime="text/csv",
        )
    if accepted:
        with st.expander("Akzeptierte Segmente anzeigen"):
            import pandas as pd

            st.dataframe(pd.DataFrame(accepted), use_container_width=True, height=320)
        _render_manual_reference_builder(
            output_dir,
            accepted,
            denoise_references=denoise_references,
            denoise_strength_db=denoise_strength_db,
        )
    _render_reference_synthesis_controls(
        output_dir,
        backend=backend,
        text=text,
        language=language,
        consent_confirmed=consent_confirmed,
        synthesis_timeout_sec=synthesis_timeout_sec,
        xtts_license_confirmed=xtts_license_confirmed,
    )
    if generated:
        st.subheader("Generierte Samples")
        for wav in generated:
            with st.container(border=True):
                st.markdown(f"**{wav.name}**")
                st.audio(wav.read_bytes(), format="audio/wav")
                sample_cols = st.columns([1, 1, 3])
                sample_cols[0].download_button(
                    "WAV herunterladen",
                    data=wav.read_bytes(),
                    file_name=wav.name,
                    mime="audio/wav",
                    key=f"download-{wav}",
                )
                with sample_cols[1]:
                    _render_synthesis_review(output_dir, wav, feedback)
                sidecar = wav.with_suffix(wav.suffix + ".synthetic.json")
                if sidecar.exists():
                    with st.expander(f"Metadaten: {sidecar.name}"):
                        st.json(json.loads(sidecar.read_text(encoding="utf-8")))
    report = output_dir / "report.md"
    if report.exists():
        with st.expander("Report"):
            st.markdown(report.read_text(encoding="utf-8"))


def _current_sample_key(output_dir: Path) -> str:
    return f"voice-pipeline-current-samples:{output_dir.resolve()}"


def _current_session_generated(output_dir: Path) -> list[Path]:
    current = st.session_state.get(_current_sample_key(output_dir), [])
    generated: list[Path] = []
    for value in current:
        path = Path(str(value))
        if path.exists() and path.suffix.lower() == ".wav":
            generated.append(path)
    return sorted(generated)


def _manual_reference_key(output_dir: Path) -> str:
    return f"voice-pipeline-manual-refs:{output_dir.resolve()}"


def _render_manual_reference_builder(
    output_dir: Path,
    accepted: list[dict[str, Any]],
    *,
    denoise_references: bool,
    denoise_strength_db: float,
) -> None:
    candidates = sorted(accepted, key=_reference_candidate_score, reverse=True)
    if not candidates:
        return
    selection_key = _manual_reference_key(output_dir)
    if selection_key not in st.session_state:
        st.session_state[selection_key] = _default_reference_selection(candidates, target_total_sec=60.0)

    with st.expander("Reference-Pack manuell kuratieren"):
        st.caption(
            "Hoere die besten akzeptierten Clips an und markiere nur eindeutige, saubere Zielsprecher-Passagen. "
            "Ein kleiner, sauberer Pack ist fuer XTTS oft besser als viele gemischte Clips."
        )
        max_candidates = min(len(candidates), 60)
        shown = st.slider(
            "Kandidaten anzeigen",
            min_value=min(5, max_candidates),
            max_value=max_candidates,
            value=min(20, max_candidates),
            step=1,
            help="Begrenzt nur die Anzeige in der UI. Bereits markierte Clips bleiben in der Auswahl, auch wenn sie gerade nicht sichtbar sind.",
            key=f"{output_dir}-manual-ref-candidate-count",
        )
        selected_files = set(st.session_state.get(selection_key, []))
        visible_rows = candidates[:shown]
        for index, row in enumerate(visible_rows, start=1):
            file_path = Path(str(row["file"]))
            if not file_path.exists():
                continue
            file_id = str(file_path)
            with st.container(border=True):
                cols = st.columns([1, 3])
                with cols[0]:
                    selected = st.checkbox(
                        f"Ref {index:02d}",
                        value=file_id in selected_files,
                        key=f"{output_dir}-manual-ref-{file_path.name}",
                    )
                    if selected:
                        selected_files.add(file_id)
                    else:
                        selected_files.discard(file_id)
                    st.caption(_reference_metric_line(row))
                with cols[1]:
                    st.audio(file_path.read_bytes(), format="audio/wav")

        st.session_state[selection_key] = sorted(selected_files)
        selected_rows = _selected_reference_rows(candidates, selected_files)
        selected_total = sum(float(row.get("duration_sec") or 0.0) for row in selected_rows)
        metric_cols = st.columns(3)
        metric_cols[0].metric("Ausgewaehlt", len(selected_rows))
        metric_cols[1].metric("Dauer", f"{selected_total:.1f}s")
        metric_cols[2].metric("Kandidaten", len(candidates))
        if st.button("Auswahl als Reference-Pack verwenden", disabled=not selected_rows, type="secondary"):
            from voice_pipeline.logging_utils import write_json, write_jsonl
            from voice_pipeline.reference_pack import build_reference_pack_from_rows

            reference_rows = _prepare_reference_rows(
                selected_rows,
                output_dir=output_dir,
                denoise_references=denoise_references,
                denoise_strength_db=denoise_strength_db,
            )
            metadata = build_reference_pack_from_rows(
                output_dir / ("denoised_segments" if denoise_references else "clean_segments"),
                reference_rows,
                output_dir / "voice_refs",
                target_total_sec=round(selected_total, 3),
                selection_mode="manual",
            )
            metadata["denoise_enabled"] = denoise_references
            metadata["denoise_strength_db"] = denoise_strength_db if denoise_references else 0.0
            write_json(output_dir / "voice_refs" / "reference_pack.json", metadata)
            write_jsonl(output_dir / "manifests" / "manual_reference_selection.jsonl", reference_rows)
            st.session_state[_current_sample_key(output_dir)] = []
            st.success(
                f"Manuelles Reference-Pack gespeichert: {len(metadata.get('refs', []))} Clips, "
                f"{metadata.get('actual_total_sec', 0)} Sekunden."
            )


def _render_reference_synthesis_controls(
    output_dir: Path,
    *,
    backend: str,
    text: str,
    language: str,
    consent_confirmed: bool,
    synthesis_timeout_sec: int,
    xtts_license_confirmed: bool,
) -> None:
    reference_files = sorted((output_dir / "voice_refs").glob("ref_*.wav"))
    if not reference_files:
        return
    with st.expander("Synthese mit aktuellem Reference-Pack erzeugen"):
        st.caption(
            f"Aktueller Pack: {len(reference_files)} Referenzclips. "
            "Nutze diesen Schritt nach manueller Auswahl, ohne die ganze Pipeline erneut zu starten."
        )
        if st.button("Synthese jetzt erzeugen", type="primary"):
            _synthesize_reference_pack_from_ui(
                output_dir=output_dir,
                backend=backend,
                text=text,
                language=language,
                consent_confirmed=consent_confirmed,
                synthesis_timeout_sec=synthesis_timeout_sec,
                xtts_license_confirmed=xtts_license_confirmed,
            )


def _synthesize_reference_pack_from_ui(
    *,
    output_dir: Path,
    backend: str,
    text: str,
    language: str,
    consent_confirmed: bool,
    synthesis_timeout_sec: int,
    xtts_license_confirmed: bool,
) -> None:
    from voice_pipeline.metadata import synthesis_metadata, validate_consent, write_synthesis_log

    reference_files = sorted((output_dir / "voice_refs").glob("ref_*.wav"))
    if not reference_files:
        st.error("Keine Reference-Clips vorhanden.")
        return
    try:
        validate_consent(consent_confirmed)
    except PermissionError as exc:
        st.error(str(exc))
        return
    progress = st.progress(0, text=f"Synthese gestartet. Automatischer Abbruch in {_format_countdown(synthesis_timeout_sec)}.")
    synthesis_log_box = st.empty()
    append_synthesis_log = _make_synthesis_log_callback(synthesis_log_box)

    def update_synthesis_countdown(elapsed_sec: int, remaining_sec: int) -> None:
        percent = int(min(100, max(0, elapsed_sec / max(1, synthesis_timeout_sec) * 100)))
        progress.progress(percent, text=f"Synthese laeuft. Automatischer Abbruch in {_format_countdown(remaining_sec)}.")

    output_wav = _next_synthesis_output(output_dir, backend)
    try:
        result = _backend(
            backend,
            timeout_sec=synthesis_timeout_sec,
            pid_file=output_dir / "work" / "synthesis.pid",
            xtts_license_confirmed=xtts_license_confirmed,
            progress_callback=update_synthesis_countdown,
            log_callback=append_synthesis_log,
        ).synthesize(text, reference_files, output_wav, language=language)
    except (RuntimeError, NotImplementedError, ValueError) as exc:
        progress.progress(100, text="Synthese abgebrochen oder fehlgeschlagen.")
        st.error(f"Synthese fehlgeschlagen: {exc}")
        return

    metadata = synthesis_metadata(output_dir.name, backend, text, language, reference_files, result, consent_confirmed)
    write_synthesis_log(output_dir / "manifests" / "synthesis_runs.jsonl", metadata, result)
    current_samples = st.session_state.setdefault(_current_sample_key(output_dir), [])
    if str(result) not in current_samples:
        current_samples.append(str(result))
    progress.progress(100, text="Synthese fertig.")
    st.success(f"Synthetisches Sample erzeugt: {result.name}")
    st.audio(result.read_bytes(), format="audio/wav")


def _next_synthesis_output(output_dir: Path, backend: str) -> Path:
    out_dir = output_dir / "generated_samples"
    out_dir.mkdir(parents=True, exist_ok=True)
    for index in range(1, 1000):
        candidate = out_dir / f"sample_{backend}_{index:03d}.wav"
        if not candidate.exists():
            return candidate
    return out_dir / f"sample_{backend}_{int(time.time())}.wav"


def _default_reference_selection(rows: list[dict[str, Any]], target_total_sec: float) -> list[str]:
    selected: list[str] = []
    total = 0.0
    for row in rows:
        file_path = Path(str(row.get("file", "")))
        if not file_path.exists():
            continue
        selected.append(str(file_path))
        total += float(row.get("duration_sec") or 0.0)
        if total >= target_total_sec and selected:
            break
    return selected


def _selected_reference_rows(rows: list[dict[str, Any]], selected_files: set[str]) -> list[dict[str, Any]]:
    return [row for row in rows if str(Path(str(row.get("file", "")))) in selected_files]


def _prepare_reference_rows(
    rows: list[dict[str, Any]],
    *,
    output_dir: Path,
    denoise_references: bool,
    denoise_strength_db: float,
) -> list[dict[str, Any]]:
    if not denoise_references:
        return [dict(row, denoise_enabled=False, denoise_strength_db=0.0) for row in rows]
    from voice_pipeline.audio_io import denoise_wav

    denoised_dir = output_dir / "denoised_segments"
    prepared: list[dict[str, Any]] = []
    for row in rows:
        source = Path(str(row["file"]))
        if not source.exists():
            continue
        target = denoised_dir / source.name
        denoise_wav(source, target, noise_reduction_db=denoise_strength_db)
        updated = dict(row)
        updated["file"] = str(target)
        updated["denoise_enabled"] = True
        updated["denoise_strength_db"] = denoise_strength_db
        updated["denoise_source_file"] = str(source)
        prepared.append(updated)
    return prepared


def _write_reference_source_manifest(path: Path, rows: list[dict[str, Any]]) -> Path:
    from voice_pipeline.logging_utils import write_jsonl

    write_jsonl(path, rows)
    return path


def _reference_candidate_score(row: dict[str, Any]) -> float:
    duration = float(row.get("duration_sec") or 0.0)
    speech = float(row.get("speech_ratio") or 0.0)
    clipping = float(row.get("clipping_ratio") or 0.0)
    overlap = float(row.get("overlap_sec") or 0.0)
    rms = float(row.get("rms_db") or -120.0)
    duration_penalty = 0.0 if 5.0 <= duration <= 11.0 else min(abs(duration - 5.0), abs(duration - 11.0))
    rms_penalty = abs(rms + 22.0)
    return speech - clipping * 10.0 - overlap * 0.25 - duration_penalty * 0.08 - rms_penalty * 0.01


def _reference_metric_line(row: dict[str, Any]) -> str:
    return (
        f"{float(row.get('duration_sec') or 0):.1f}s | "
        f"Speech {float(row.get('speech_ratio') or 0):.2f} | "
        f"RMS {float(row.get('rms_db') or 0):.1f} dB | "
        f"Overlap {float(row.get('overlap_sec') or 0):.1f}s"
    )


def _render_synthesis_review(output_dir: Path, wav: Path, technical_feedback_path: Path) -> None:
    if not technical_feedback_path.exists():
        return
    dialog_key = f"review-dialog-open-{wav}"
    if st.button("Stimme bewerten", key=f"{wav}-open-review"):
        st.session_state[dialog_key] = True
    if st.session_state.get(dialog_key):
        _render_synthesis_review_dialog(output_dir, wav, technical_feedback_path, dialog_key)


@_dialog("Stimme qualitativ bewerten")
def _render_synthesis_review_dialog(output_dir: Path, wav: Path, technical_feedback_path: Path, dialog_key: str) -> None:
    from voice_pipeline.feedback import build_synthesis_review_csv

    st.audio(wav.read_bytes(), format="audio/wav")
    st.markdown(
        "**Bewertungsskala**  \n"
        "*1 = schwach oder problematisch, 3 = brauchbar mit sichtbaren Grenzen, 5 = sehr gut. "
        "Bei Artefakten bedeutet 1 kaum Stoerungen und 5 stark stoerende Artefakte.*"
    )
    cols = st.columns(2)
    with cols[0]:
        overall_quality = _review_slider(
            "Gesamtqualitaet",
            "Gesamteindruck aus Klang, Stabilitaet, Verstaendlichkeit und Nutzbarkeit. 1 = nicht brauchbar, 5 = ohne groessere Einschraenkung nutzbar.",
            key=f"{wav}-overall-quality",
        )
        voice_similarity = _review_slider(
            "Stimm-Aehnlichkeit",
            "Wie nah Klangfarbe, Alterseindruck, Sprechlage und Timbre an der Zielstimme liegen. 1 = andere Stimme, 5 = sehr nah an der Referenz.",
            key=f"{wav}-voice-similarity",
        )
        speaker_recognition = _review_slider(
            "Treffergenauigkeit Zielsprecher",
            "Ob die synthetische Stimme eindeutig dem gewuenschten Sprecher zugeordnet werden kann. 1 = Zielsprecher kaum erkennbar, 5 = klar erkennbar.",
            key=f"{wav}-speaker-recognition",
        )
    with cols[1]:
        intelligibility = _review_slider(
            "Verstaendlichkeit",
            "Wie gut Woerter, Satzmelodie und Aussprache verstanden werden. 1 = schwer verstaendlich, 5 = klar und sauber.",
            key=f"{wav}-intelligibility",
        )
        naturalness = _review_slider(
            "Natuerlichkeit",
            "Wie menschlich und fluessig die Stimme wirkt, inklusive Rhythmus, Betonung und Pausen. 1 = deutlich kuenstlich, 5 = sehr natuerlich.",
            key=f"{wav}-naturalness",
        )
        artifact_level = _review_slider(
            "Artefakte/Stoerungen",
            "Hoerbare Fehler wie Rauschen, metallischer Klang, Knacken, Hall, Pumpen oder instabile Phoneme. 1 = kaum Stoerungen, 5 = stark stoerend.",
            key=f"{wav}-artifact-level",
        )
    st.markdown("**Optimierungsbedarf**  \n*Markiere, welche Stellschrauben beim naechsten Lauf wahrscheinlich helfen wuerden.*")
    needs = st.multiselect(
        "Was sollte optimiert werden?",
        [
            "mehr Referenzmaterial",
            "weniger Referenzmaterial",
            "strengere Segmentfilter",
            "weniger Overlap-Ausschluss",
            "anderer Synthese-Text",
            "andere Sprache/Phonetik",
            "Rauschen reduzieren",
            "Lautstaerke/Normalisierung",
            "anderes Backend",
        ],
        key=f"{wav}-needs",
    )
    notes = st.text_area("Notizen zur Stimme und Treffergenauigkeit", key=f"{wav}-review-notes")
    review = {
        "overall_quality_1_to_5": overall_quality,
        "voice_similarity_1_to_5": voice_similarity,
        "target_speaker_accuracy_1_to_5": speaker_recognition,
        "intelligibility_1_to_5": intelligibility,
        "naturalness_1_to_5": naturalness,
        "artifact_level_1_to_5": artifact_level,
        "optimization_needs": "; ".join(needs),
        "notes": notes,
    }
    csv_text = build_synthesis_review_csv(
        speaker=output_dir.name,
        sample_wav=wav,
        technical_feedback_csv=technical_feedback_path.read_text(encoding="utf-8"),
        review=review,
        sample_synthesis_metadata=_read_sample_sidecar(wav),
    )
    review_path = output_dir / "manifests" / f"{wav.stem}-quality-review.csv"
    review_path.write_text(csv_text, encoding="utf-8")
    if st.download_button(
        "Bewertungs-CSV herunterladen",
        data=csv_text.encode("utf-8"),
        file_name=f"{output_dir.name}-{wav.stem}-quality-review.csv",
        mime="text/csv",
        key=f"{wav}-download-review",
    ):
        st.session_state[dialog_key] = False
        st.rerun()
    if st.button("Schliessen", key=f"{wav}-close-review"):
        st.session_state[dialog_key] = False
        st.rerun()


def _review_slider(label: str, description: str, *, key: str) -> int:
    st.markdown(f"**{label}**  \n*{description}*")
    return int(st.slider(label, 1, 5, 3, key=key, label_visibility="collapsed"))


def _read_sample_sidecar(wav: Path) -> dict[str, Any] | None:
    sidecar = wav.with_suffix(wav.suffix + ".synthetic.json")
    if not sidecar.exists():
        return None
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _render_abort_control(output_dir: Path) -> None:
    pid_file = output_dir / "work" / "synthesis.pid"
    if not pid_file.exists():
        return
    raw_pid = pid_file.read_text(encoding="utf-8").strip()
    if not raw_pid.isdigit():
        pid_file.unlink(missing_ok=True)
        return
    st.warning(f"Synthese laeuft noch mit Prozess-ID {raw_pid}.")
    if st.button("Synthese abbrechen", type="secondary"):
        try:
            os.kill(int(raw_pid), signal.SIGTERM)
            st.success("Abbruchsignal gesendet. Der Lauf schreibt danach die Fehleranalyse.")
        except ProcessLookupError:
            st.info("Der Synthese-Prozess laeuft nicht mehr.")
        finally:
            pid_file.unlink(missing_ok=True)


def _backend(
    name: str,
    timeout_sec: int = 30 * 60,
    pid_file: Path | None = None,
    xtts_license_confirmed: bool = False,
    progress_callback: Callable[[int, int], None] | None = None,
    log_callback: Callable[[str], None] | None = None,
):
    if name == "mock":
        from voice_pipeline.synthesis.mock_backend import MockBackend

        return MockBackend()
    if name == "xtts":
        from voice_pipeline.synthesis.xtts_backend import XTTSBackend

        return XTTSBackend(
            timeout_sec=timeout_sec,
            pid_file=pid_file,
            license_confirmed=xtts_license_confirmed,
            progress_callback=progress_callback,
        )
    if name == "openvoice":
        from voice_pipeline.synthesis.openvoice_backend import OpenVoiceBackend

        return OpenVoiceBackend(
            timeout_sec=timeout_sec,
            pid_file=pid_file,
            progress_callback=progress_callback,
            log_callback=log_callback,
        )
    raise ValueError(f"Unsupported backend: {name}")


def _backend_hint(name: str) -> str:
    if name == "xtts":
        return (
            "XTTS laeuft bewusst in einer separaten Umgebung. Zusaetzlich muessen die Coqui/XTTS-Lizenzbedingungen "
            "fuer deinen Einsatzzweck geprueft und bestaetigt werden."
        )
    if name == "openvoice":
        return (
            "OpenVoice benoetigt eine separate OpenVoice-V2 Umgebung plus checkpoints_v2. "
            "Richte sie mit bash scripts/setup_openvoice_env.sh ein und setze bei Bedarf OPENVOICE_CHECKPOINT_DIR."
        )
    return "Mock erzeugt nur ein Test-WAV und benoetigt keine Modellinstallation."


def _recent_files(directory: Path, suffixes: set[str], name_contains: str | None = None, limit: int = 8) -> list[Path]:
    if not directory.exists():
        return []
    files: list[Path] = []
    for path in directory.iterdir():
        if not path.is_file():
            continue
        if path.suffix.lower() not in suffixes:
            continue
        if name_contains and name_contains.lower() not in path.name.lower():
            continue
        files.append(path)
    return sorted(files, key=lambda path: path.stat().st_mtime, reverse=True)[:limit]


def _save_uploaded_once(uploaded: Any, directory: Path, state_prefix: str) -> Path:
    signature = f"{uploaded.name}:{getattr(uploaded, 'size', 'unknown')}"
    signature_key = f"{state_prefix}_upload_signature"
    path_key = f"{state_prefix}_upload_path"
    previous_path = st.session_state.get(path_key)
    if st.session_state.get(signature_key) == signature and previous_path and Path(previous_path).exists():
        return Path(previous_path)

    target = directory / _safe_filename(uploaded.name)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(uploaded.getbuffer())
    st.session_state[signature_key] = signature
    st.session_state[path_key] = str(target)
    return target


def _safe_filename(name: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "-" for char in name).strip("-") or "upload"
