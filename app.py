from collections import deque
from pathlib import Path
import sys
import time
from typing import Any

import streamlit as st

APP_ROOT = Path(__file__).resolve().parent
SRC_ROOT = APP_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from transcript_mvp.estimates import estimate_processing_seconds
from transcript_mvp.kpis import quality_notes, resource_kpis, transcript_kpis
from transcript_mvp.local_diarization import run_local_diarize
from transcript_mvp.models import SpeakerMapping
from transcript_mvp.pipeline import (
    assign_speakers_by_overlap,
    create_run_dir,
    extract_speakers,
    get_audio_duration,
    load_transcript_json,
    render_segments,
    save_upload,
    run_whisperx,
)
from transcript_mvp.progress import ProgressReporter
from transcript_mvp.resources import ResourceSnapshot, collect_resource_snapshot

DATA_DIR = APP_ROOT / "data"
ResourceHistory = deque[dict[str, float | int]]

_PERF_DEFAULTS: dict[bool, dict[str, int | str]] = {
    True: {
        "batch_size": 1,
        "chunk_size": 20,
        "threads": 4,
        "batch_help": (
            "Anzahl der Audio-Stuecke, die WhisperX gleichzeitig verarbeitet. "
            "Kleinere Werte brauchen weniger RAM, sind aber langsamer."
        ),
        "chunk_help": (
            "Laenge der Audio-Bloecke fuer die Transkription. Kleinere Chunks senken Speicherverbrauch, "
            "koennen aber etwas langsamer und weniger stabil im Kontext sein."
        ),
        "threads_help": (
            "Anzahl CPU-Threads fuer WhisperX. 0 ueberlaesst die Wahl dem System; mehr Threads koennen "
            "schneller sein, belasten aber den Rechner staerker."
        ),
    },
    False: {
        "batch_size": 4,
        "chunk_size": 30,
        "threads": 0,
        "batch_help": (
            "Anzahl der Audio-Stuecke, die WhisperX gleichzeitig verarbeitet. "
            "Groessere Werte koennen schneller sein, brauchen aber deutlich mehr RAM."
        ),
        "chunk_help": (
            "Laenge der Audio-Bloecke fuer die Transkription. Groessere Chunks geben dem Modell mehr Kontext, "
            "brauchen aber mehr Speicher."
        ),
        "threads_help": (
            "Anzahl CPU-Threads fuer WhisperX. 0 ueberlaesst die Wahl dem System; feste Werte koennen "
            "Last und Laufzeit planbarer machen."
        ),
    },
}


def _init_session_state() -> None:
    defaults = {
        "segments": [],
        "source_name": "transkript",
        "feedback_csv": None,
        "feedback_filename": "feedback.csv",
        "speaker_segments": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "unbekannt"
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours} Std. {minutes:02d} Min."
    if minutes:
        return f"{minutes} Min. {secs:02d} Sek."
    return f"{secs} Sek."


def format_percent(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.0f}%"


def format_gb(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.1f} GB"


def memory_pressure_help(label: str) -> str:
    if label == "kritisch":
        return "Kritisch: Transkription besser abbrechen oder kleineres Modell nutzen."
    if label == "hoch":
        return "Hoch: Speichermodus, kleineres Modell oder Sprechererkennung ohne Testlauf vermeiden."
    if label == "mittel":
        return "Mittel: Beobachten, aber normalerweise noch nutzbar."
    if label == "niedrig":
        return "Niedrig: Es ist ausreichend Speicher verfuegbar."
    return "Nicht verfuegbar."


def append_resource_history(
    history: ResourceHistory,
    elapsed: float,
    process_pid: int,
    snapshot: ResourceSnapshot | None = None,
) -> None:
    snapshot = snapshot or collect_resource_snapshot(process_pid)
    if not snapshot.available:
        return
    process_cpu_percent = snapshot.process_cpu_percent or 0.0
    if snapshot.process_cpu_time_seconds is not None and history:
        previous = history[-1]
        elapsed_delta = max(0.001, elapsed - float(previous.get("Zeit exakt", previous.get("Zeit", 0))))
        cpu_delta = max(0.0, snapshot.process_cpu_time_seconds - float(previous.get("WhisperX CPU Sekunden", 0.0)))
        process_cpu_percent = (cpu_delta / elapsed_delta) * 100.0
    history.append(
        {
            "Zeit": int(elapsed),
            "Zeit exakt": elapsed,
            "CPU gesamt %": snapshot.system_cpu_percent or 0.0,
            "WhisperX CPU %": process_cpu_percent,
            "WhisperX CPU Sekunden": snapshot.process_cpu_time_seconds or 0.0,
            "RAM gesamt %": snapshot.system_memory_percent or 0.0,
            "Speicherdruck %": snapshot.memory_pressure_percent or 0.0,
            "WhisperX RAM GB": snapshot.process_memory_gb or 0.0,
            "RAM genutzt GB": snapshot.system_memory_used_gb or 0.0,
        }
    )


def render_resource_history(history: ResourceHistory | list[dict[str, float | int]]) -> None:
    if len(history) < 2:
        return
    import altair as alt
    import pandas as pd

    frame = pd.DataFrame(list(history)).drop_duplicates(subset=["Zeit"], keep="last")
    st.markdown("**Verlauf**")
    system_frame = frame[["Zeit", "CPU gesamt %", "RAM gesamt %", "Speicherdruck %"]].melt(
        "Zeit",
        var_name="Kennzahl",
        value_name="Wert",
    )
    st.altair_chart(
        alt.Chart(system_frame)
        .mark_line()
        .encode(
            x=alt.X("Zeit:Q", title="Sekunden"),
            y=alt.Y("Wert:Q", title="Prozent", scale=alt.Scale(domain=[0, 100])),
            color="Kennzahl:N",
        )
        .properties(height=220),
        use_container_width=True,
    )

    process_cpu_frame = frame[["Zeit", "WhisperX CPU %"]]
    st.altair_chart(
        alt.Chart(process_cpu_frame)
        .mark_line(color="#ff8f8f")
        .encode(
            x=alt.X("Zeit:Q", title="Sekunden"),
            y=alt.Y("WhisperX CPU %:Q", title="WhisperX CPU %"),
        )
        .properties(height=160),
        use_container_width=True,
    )

    ram_frame = frame[["Zeit", "WhisperX RAM GB", "RAM genutzt GB"]].melt(
        "Zeit",
        var_name="Kennzahl",
        value_name="Wert",
    )
    st.altair_chart(
        alt.Chart(ram_frame)
        .mark_line()
        .encode(
            x=alt.X("Zeit:Q", title="Sekunden"),
            y=alt.Y("Wert:Q", title="GB"),
            color="Kennzahl:N",
        )
        .properties(height=160),
        use_container_width=True,
    )


def build_streamlit_progress_reporter(
    *,
    overall_progress: Any,
    transcription_progress: Any,
    elapsed_box: Any,
    resource_box: Any,
    log_box: Any,
    resource_history: ResourceHistory,
    estimated_seconds: float | None,
    use_local_diarize: bool,
) -> ProgressReporter:
    log_lines: list[str] = []
    progress_state = {"transcription": 0.0}

    def update_overall_progress(value: float, text: str) -> None:
        overall_progress.progress(min(1.0, max(0.0, value)), text=text)

    def update_transcription_progress(value: float) -> None:
        progress_state["transcription"] = value
        transcription_progress.progress(
            value,
            text=f"Transkription: {value * 100:.1f}%",
        )
        update_overall_progress(
            value * (0.85 if use_local_diarize else 0.95),
            f"Gesamtfortschritt: Transkription {value * 100:.1f}%",
        )

    def update_elapsed(elapsed: float, process_pid: int) -> None:
        if estimated_seconds:
            remaining = max(0.0, estimated_seconds - elapsed)
            if progress_state["transcription"] == 0:
                update_overall_progress(
                    min(0.2, elapsed / estimated_seconds),
                    "Gesamtfortschritt: startet...",
                )
            transcription_progress.progress(
                progress_state["transcription"],
                text=(
                    f"Transkription: {progress_state['transcription'] * 100:.1f}% "
                    f"- laeuft seit {format_duration(elapsed)}, "
                    f"grobe Restzeit {format_duration(remaining)}"
                ),
            )
        else:
            transcription_progress.progress(
                progress_state["transcription"],
                text=(
                    f"Transkription: {progress_state['transcription'] * 100:.1f}% "
                    f"- laeuft seit {format_duration(elapsed)}"
                ),
            )
        elapsed_box.caption(
            "Die Anzeige ist eine Schaetzung, weil WhisperX keinen exakten Gesamtfortschritt liefert."
        )
        snapshot = collect_resource_snapshot(process_pid)
        append_resource_history(resource_history, elapsed, process_pid, snapshot)
        with resource_box.container():
            st.markdown("**System & Speicher**")
            if not snapshot.available:
                st.warning(snapshot.message)
            else:
                metric_columns = st.columns(4)
                metric_columns[0].metric("CPU gesamt", format_percent(snapshot.system_cpu_percent))
                metric_columns[1].metric("WhisperX CPU", format_percent(snapshot.process_cpu_percent))
                metric_columns[2].metric(
                    "RAM gesamt",
                    f"{format_percent(snapshot.system_memory_percent)}",
                    help=(
                        f"{format_gb(snapshot.system_memory_used_gb)} von "
                        f"{format_gb(snapshot.system_memory_total_gb)}"
                    ),
                )
                metric_columns[3].metric("WhisperX RAM", format_gb(snapshot.process_memory_gb))
                pressure = snapshot.memory_pressure_percent or 0.0
                st.progress(
                    min(1.0, pressure / 100.0),
                    text=(
                        f"Speicherdruck: {snapshot.memory_pressure_label} "
                        f"({format_percent(snapshot.memory_pressure_percent)})"
                    ),
                )
                st.caption(memory_pressure_help(snapshot.memory_pressure_label))
                render_resource_history(resource_history)

    def append_log(line: str) -> None:
        log_lines.append(line)
        visible_lines = log_lines[-30:]
        log_box.code("\n".join(visible_lines), language="text")

    return ProgressReporter(
        on_overall=update_overall_progress,
        on_transcription=update_transcription_progress,
        on_tick=update_elapsed,
        on_log=append_log,
    )


def render_finished_kpis(audio_duration: float | None, processing_seconds: float, resource_history: list[dict]) -> None:
    transcript_stats = transcript_kpis(st.session_state.segments, audio_duration)
    resource_stats = resource_kpis(resource_history)
    st.subheader("Auswertung")

    tech_columns = st.columns(4)
    tech_columns[0].metric("Laufzeit", format_duration(processing_seconds))
    tech_columns[1].metric("Faktor", f"{processing_seconds / audio_duration:.2f}x" if audio_duration else "-")
    tech_columns[2].metric("Peak RAM", format_gb(resource_stats.peak_process_ram_gb))
    tech_columns[3].metric("Peak Speicherdruck", format_percent(resource_stats.peak_memory_pressure_percent))

    quality_columns = st.columns(4)
    quality_columns[0].metric("Abschnitte", str(transcript_stats.segment_count))
    quality_columns[1].metric("Sprecher", str(transcript_stats.speaker_count))
    quality_columns[2].metric("Ø Abschnitt", format_duration(transcript_stats.avg_segment_duration))
    quality_columns[3].metric("Längster Abschnitt", format_duration(transcript_stats.max_segment_duration))

    detail_columns = st.columns(3)
    detail_columns[0].metric("Wörter", str(transcript_stats.word_count))
    detail_columns[1].metric(
        "Abschnitte/Stunde",
        f"{transcript_stats.segments_per_hour:.1f}" if transcript_stats.segments_per_hour else "-",
    )
    detail_columns[2].metric(
        "Dominanter Sprecher",
        f"{transcript_stats.dominant_speaker_share * 100:.0f}%" if transcript_stats.dominant_speaker_share else "-",
    )

    for note in quality_notes(transcript_stats):
        st.info(note)


st.set_page_config(page_title="Lokale Transkription", layout="wide")
_init_session_state()

st.title("Lokale Transkription")
st.caption("MP3 rein, Sprecherlabels pruefen, Word-Datei raus.")

if st.session_state.pop("open_voice_pipeline", False):
    st.session_state.active_page = "Voice-Pipeline"

page = st.sidebar.radio(
    "Bereich",
    ["Transkription", "Voice-Pipeline"],
    key="active_page",
)

if page == "Voice-Pipeline":
    from transcript_mvp.voice_pipeline_ui import render_voice_pipeline_page

    render_voice_pipeline_page(DATA_DIR)
    st.stop()

with st.sidebar:
    st.header("Einstellungen")
    memory_mode = st.toggle(
        "Speichermodus",
        value=True,
        help="Deutlich weniger RAM-Verbrauch. Dafuer langsamer und etwas weniger genau.",
    )
    default_model_index = 1
    model = st.selectbox("Whisper-Modell", ["small", "medium", "large-v3"], index=default_model_index)
    language = st.selectbox("Sprache", ["de", "en", "auto"], index=0)
    diarize_choice = st.radio(
        "Sprecher",
        ["Sprecher erkennen", "Nur transkribieren"],
        index=0,
        help="Sprecher erkennen trennt die Unterhaltung in Sprecherlabels. Nur transkribieren ist schneller.",
    )
    diarize = diarize_choice == "Sprecher erkennen"
    min_speakers = st.number_input(
        "Min. Sprecher",
        min_value=0,
        max_value=20,
        value=1,
        help="Untere Grenze fuer die Sprechererkennung. 1 ist sinnvoll, wenn mindestens eine Stimme sicher vorhanden ist.",
    )
    max_speakers = st.number_input(
        "Max. Sprecher",
        min_value=0,
        max_value=20,
        value=2,
        help="Obere Grenze fuer die Sprechererkennung. 2 passt gut fuer Interviews; hoeher setzen, wenn mehr Personen sprechen.",
    )
    with st.expander("Leistung & Speicher"):
        perf_defaults = _PERF_DEFAULTS[memory_mode]
        batch_size = st.number_input(
            "Batch-Groesse",
            min_value=1,
            max_value=16,
            value=int(perf_defaults["batch_size"]),
            help=str(perf_defaults["batch_help"]),
        )
        chunk_size = st.number_input(
            "Chunk-Groesse Sekunden",
            min_value=5,
            max_value=60,
            value=int(perf_defaults["chunk_size"]),
            help=str(perf_defaults["chunk_help"]),
        )
        no_align = st.toggle(
            "Wortgenaue Ausrichtung sparen",
            value=False,
            help="Ueberspringt die genaue Wort-Zeit-Ausrichtung. Das spart Zeit und Speicher, kann aber weniger genaue Zeitmarken liefern.",
        )
        threads = st.number_input(
            "CPU-Threads",
            min_value=0,
            max_value=16,
            value=int(perf_defaults["threads"]),
            help=str(perf_defaults["threads_help"]),
        )
        vad_method = st.selectbox(
            "Spracherkennung vor Transkription",
            ["pyannote", "silero"],
            index=0,
            help="pyannote laeuft bei dir stabil lokal. Silero kann GitHub-Zugriff brauchen, falls es nicht im Cache ist.",
        )
    with st.expander("Feedback-CSV"):
        include_feedback_text_samples = st.toggle(
            "Kurze Textbeispiele aufnehmen",
            value=False,
            help="Aus Datenschutzgruenden aus. Aktivieren, wenn die CSV kleine Textproben zur Plausibilitaetspruefung enthalten soll.",
        )
    st.caption("Speichermodus nutzt `int8`, kleine Batches und kleinere Audio-Chunks.")

uploaded = st.file_uploader("MP3-Datei auswaehlen", type=["mp3", "wav", "m4a", "mp4"])

left, right = st.columns([2, 1])

with left:
    if uploaded is not None:
        st.session_state.source_name = Path(uploaded.name).stem
        st.write(f"Ausgewaehlt: `{uploaded.name}`")

        if st.button("Transkription starten", type="primary"):
            use_local_diarize = diarize
            if min_speakers and max_speakers and min_speakers > max_speakers:
                st.error("Min. Sprecher darf nicht groesser als Max. Sprecher sein.")
                st.stop()

            run_dir = create_run_dir(DATA_DIR / "runs", st.session_state.source_name)
            audio_path = save_upload(uploaded, DATA_DIR / "uploads")
            st.session_state.last_audio_path = str(audio_path)
            st.session_state.last_run_dir = str(run_dir)
            st.session_state.speaker_segments = []
            audio_duration = get_audio_duration(audio_path)
            estimate = estimate_processing_seconds(
                audio_seconds=audio_duration,
                model=model,
                speaker_backend="diarize" if diarize else "disabled",
                memory_mode=memory_mode,
                no_align=no_align,
                batch_size=batch_size,
                chunk_size=chunk_size,
                threads=threads,
                vad_method=vad_method,
                history_dir=DATA_DIR / "runs",
            )
            estimated_seconds = estimate.seconds
            run_started_at = time.monotonic()
            speaker_segments_count = None
            with st.status("WhisperX verarbeitet die Datei lokal...", expanded=True) as status:
                st.write(f"Audiodauer: {format_duration(audio_duration)}")
                st.write(f"Grobe Schaetzung: {format_duration(estimated_seconds)}")
                if estimate.ratio is not None:
                    st.caption(
                        f"Schaetzung: {estimate.source}, {estimate.samples} Vergleichslaeufe, "
                        f"Faktor {estimate.ratio:.2f}x Audiodauer."
                    )
                overall_progress = st.progress(0, text="Startet...")
                transcription_progress = st.progress(0, text="Transkription wartet...")
                elapsed_box = st.empty()
                resource_box = st.empty()
                log_box = st.empty()
                resource_history: ResourceHistory = deque(maxlen=3600)
                reporter = build_streamlit_progress_reporter(
                    overall_progress=overall_progress,
                    transcription_progress=transcription_progress,
                    elapsed_box=elapsed_box,
                    resource_box=resource_box,
                    log_box=log_box,
                    resource_history=resource_history,
                    estimated_seconds=estimated_seconds,
                    use_local_diarize=use_local_diarize,
                )

                try:
                    output_json = run_whisperx(
                        audio_path=audio_path,
                        output_dir=run_dir,
                        model=model,
                        language=None if language == "auto" else language,
                        min_speakers=min_speakers or None,
                        max_speakers=max_speakers or None,
                        batch_size=batch_size,
                        chunk_size=chunk_size,
                        threads=threads,
                        no_align=no_align,
                        vad_method=vad_method,
                        reporter=reporter,
                    )
                except RuntimeError as exc:
                    status.update(label="Fehlgeschlagen", state="error")
                    st.error(str(exc))
                    partial_jsons = sorted(run_dir.glob("*.json"))
                    if partial_jsons:
                        st.warning(
                            f"Partielles Ergebnis gefunden: `{partial_jsons[-1].name}`. "
                            "Du kannst es unten als JSON laden."
                        )
                    st.stop()
                transcription_progress.progress(1.0, text="Transkription: 100.0%")
                reporter.overall(
                    0.9 if use_local_diarize else 0.98,
                    "Transkription abgeschlossen. Ergebnis wird geladen.",
                )
                st.write("Ergebnis wird geladen.")
                transcript = load_transcript_json(output_json)
                st.session_state.segments = render_segments(
                    transcript,
                    merge_adjacent=not use_local_diarize,
                )
                if use_local_diarize:
                    reporter.overall(0.9, "Lokale Sprechererkennung laeuft ohne Token.")
                    st.write("Lokale Sprechererkennung ohne Token wird ausgefuehrt.")
                    try:
                        speaker_segments = run_local_diarize(
                            audio_path=audio_path,
                            min_speakers=min_speakers or None,
                            max_speakers=max_speakers or None,
                        )
                    except RuntimeError as exc:
                        st.warning(str(exc))
                        st.info("Das Transkript bleibt ohne Sprecherzuordnung erhalten und kann exportiert werden.")
                        st.session_state.speaker_segments = []
                    else:
                        st.session_state.speaker_segments = speaker_segments
                        speaker_segments_count = len(speaker_segments)
                        reporter.overall(0.97, "Sprecher werden dem Transkript zugeordnet.")
                        st.session_state.segments = assign_speakers_by_overlap(
                            st.session_state.segments,
                            speaker_segments,
                        )
                        st.write(f"{len(speaker_segments)} Sprecher-Zeitbereiche gefunden.")
                reporter.overall(1.0, "Fertig.")
                processing_seconds = time.monotonic() - run_started_at
                resource_rows = list(resource_history)
                from transcript_mvp.feedback import build_feedback_csv

                feedback_csv = build_feedback_csv(
                    source_name=st.session_state.source_name,
                    settings={
                        "memory_mode": memory_mode,
                        "model": model,
                        "language": language,
                        "speaker_mode": diarize_choice,
                        "speaker_backend": "diarize" if diarize else "disabled",
                        "min_speakers": min_speakers,
                        "max_speakers": max_speakers,
                        "batch_size": batch_size,
                        "chunk_size": chunk_size,
                        "no_align": no_align,
                        "threads": threads,
                        "vad_method": vad_method,
                    },
                    audio_duration_seconds=audio_duration,
                    processing_seconds=processing_seconds,
                    segments=st.session_state.segments,
                    resource_history=resource_rows,
                    speaker_segments_count=speaker_segments_count,
                    output_json_path=str(output_json),
                    include_text_samples=include_feedback_text_samples,
                )
                feedback_path = run_dir / "feedback.csv"
                feedback_path.write_text(feedback_csv, encoding="utf-8")
                st.session_state.feedback_csv = feedback_csv
                st.session_state.feedback_filename = f"{st.session_state.source_name}-feedback.csv"
                st.write(f"Feedback-CSV erstellt: `{feedback_path}`")
                render_finished_kpis(audio_duration, processing_seconds, resource_rows)
                status.update(label="Fertig", state="complete")

    uploaded_json = st.file_uploader(
        "Oder vorhandenes WhisperX-JSON laden",
        type=["json"],
        help="Praktisch zum Nachbearbeiten ohne erneute Transkription.",
    )
    if uploaded_json is not None and st.button("JSON laden"):
        import json

        transcript = json.loads(uploaded_json.getvalue().decode("utf-8"))
        st.session_state.source_name = Path(uploaded_json.name).stem
        st.session_state.segments = render_segments(transcript)
        st.session_state.speaker_segments = []

with right:
    st.subheader("Export")
    if not st.session_state.segments:
        st.info("Nach der Transkription erscheinen hier die Exporte.")
    else:
        from transcript_mvp.exports import build_diarization_json, build_diarization_jsonl, build_docx, build_html

        speakers = extract_speakers(st.session_state.segments)
        mapping_values = {}
        for speaker in speakers:
            mapping_values[speaker] = st.text_input(speaker, value=speaker, key=f"speaker-{speaker}")

        mapping = SpeakerMapping(mapping_values)
        docx_bytes = build_docx(st.session_state.segments, mapping, title=st.session_state.source_name)
        html = build_html(st.session_state.segments, mapping, title=st.session_state.source_name)

        st.download_button(
            "Word-Datei herunterladen",
            data=docx_bytes,
            file_name=f"{st.session_state.source_name}.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
        st.download_button(
            "HTML herunterladen",
            data=html.encode("utf-8"),
            file_name=f"{st.session_state.source_name}.html",
            mime="text/html",
        )
        if st.session_state.feedback_csv:
            st.download_button(
                "Feedback-CSV herunterladen",
                data=st.session_state.feedback_csv.encode("utf-8"),
                file_name=st.session_state.feedback_filename,
                mime="text/csv",
            )
        last_audio_path = st.session_state.get("last_audio_path")
        if last_audio_path:
            diarization_source = st.session_state.speaker_segments or st.session_state.segments
            diarization_json = build_diarization_json(diarization_source, last_audio_path, mapping)
            diarization_jsonl = build_diarization_jsonl(diarization_source, last_audio_path, mapping)
            st.download_button(
                "Diarization JSON herunterladen",
                data=diarization_json.encode("utf-8"),
                file_name=f"{st.session_state.source_name}-diarization.json",
                mime="application/json",
            )
            st.download_button(
                "Diarization JSONL herunterladen",
                data=diarization_jsonl.encode("utf-8"),
                file_name=f"{st.session_state.source_name}-diarization.jsonl",
                mime="application/x-ndjson",
            )
        else:
            st.caption("Diarization-Export erscheint, sobald die zugehoerige Audio-Datei in dieser Sitzung bekannt ist.")
        if st.button("Voice-Pipeline oeffnen"):
            st.session_state.open_voice_pipeline = True
            st.rerun()

if st.session_state.segments:
    st.divider()
    st.subheader("Transkript")
    speakers = extract_speakers(st.session_state.segments)
    current_mapping = SpeakerMapping(
        {speaker: st.session_state.get(f"speaker-{speaker}", speaker) for speaker in speakers}
    )
    for segment in st.session_state.segments:
        label = current_mapping.label_for(segment.speaker)
        st.markdown(f"**{segment.timestamp} - {label}**")
        st.write(segment.text)
