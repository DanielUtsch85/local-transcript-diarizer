# Problem Description: Streamlit appears stuck after WhisperX

## Incident Summary

During a real run with a large MP3, the Streamlit UI appeared stuck in processing and the user could not adjust speaker labels in the right-side export panel.

Observed state during diagnosis:

- Run directory: `data/runs/20090601_Interview_Joachim-U.1-1779530877`
- `whisperx.log` ended with `Progress: 100.00%...`
- WhisperX JSON existed and was valid:
  - `segments`: 1148
  - `language`: `de`
  - last segment end: `4673.624` seconds, about 78 minutes
- No `feedback.csv` existed in the run directory.
- No active `whisperx` process was visible.
- Streamlit server was still running.

## Working Interpretation

WhisperX itself almost certainly finished successfully. The app did not complete the Streamlit run after WhisperX, because `feedback.csv` was never written and the UI never reached the final export/speaker-edit state.

The failure window is therefore:

```text
run_whisperx() returned output_json
-> load_transcript_json(output_json)
-> render_segments(transcript)
-> optional run_local_diarize(audio_path)
-> assign_speakers_by_overlap(...)
-> build_feedback_csv(...)
-> write feedback.csv
-> Streamlit rerenders right-side speaker controls
```

The two highest-probability branches are:

1. `diarize()` was still running slowly after WhisperX.
2. `diarize()` or local diarization code raised an unexpected non-`RuntimeError` exception, aborting the Streamlit script before `feedback.csv`.

## Important Streamlit Constraint

Streamlit executes the script top-to-bottom for each interaction. The right-side speaker rename controls are only rendered after the current button run reaches the right-column block:

```python
with right:
    if not st.session_state.segments:
        st.info("Nach der Transkription erscheinen hier die Exporte.")
    else:
        speakers = extract_speakers(st.session_state.segments)
        for speaker in speakers:
            mapping_values[speaker] = st.text_input(...)
```

`st.session_state.segments` is already assigned immediately after `render_segments()`, before local diarization starts. The core problem is not missing transcript state. The problem is that Streamlit does not render the right-column controls until the whole button run finishes. If local diarization blocks or crashes before the script completes, the user cannot use the already available transcript through the UI.

## Diagnostic Distinction: Slow Run vs. Script Crash

"Streamlit is still running" does not prove that the active script run is still working. Streamlit keeps the server alive after an unhandled script exception and may show a red error page or an apparently stale UI.

To distinguish the two most likely causes during the next incident:

```text
WhisperX JSON exists + no feedback.csv
|
+-- Streamlit/Python CPU is high or a diarization worker is active
|   -> H1.2: local diarization is still running or stuck
|
+-- Streamlit/Python CPU is near 0%, browser still responds or shows an error
    -> H1.3: Streamlit script likely crashed after WhisperX
```

This distinction matters because:

- H1.2 needs job isolation, timeout/cancel behavior, or a two-stage UI.
- H1.3 needs better exception boundaries around optional diarization.

---

# Decisions From Claude Review

## Accepted

- Raise H1.3 priority: unexpected exceptions from optional diarization should be treated as the first fix.
- Explicitly document the ambiguity between "Streamlit server alive" and "script run still active".
- Downgrade speaker assignment performance as a root cause for this incident.
- Add a disk/temp-dir hypothesis for future completeness.
- Keep `run_state.json` stage markers as a high-value diagnostic improvement.
- Keep feedback CSV write hardening as a small reliability patch.

## Modified

- Transcript recovery is not "missing state"; the state is already set. The real issue is Streamlit's synchronous button-run model.
- Diarization timeout should not be implemented with a simple `ThreadPoolExecutor` context. Python threads cannot be cancelled safely, and the temporary audio file may be deleted while a timed-out thread still uses it. A real timeout should use a cancellable subprocess/job boundary or be deferred until diarization is isolated.

## Dropped For Now

- Sweep-line optimization for `assign_speakers_by_overlap`: current scale does not justify it.
- Browser reload auto-recovery: useful later, too much state-management work for the immediate incident.
- Memory-pressure heuristics as a first patch: hard to calibrate and less directly supported by evidence.

---

# Hypothesis Tree

## H1.3: Local diarization raised an unexpected exception after WhisperX

### Path

```text
run_whisperx() succeeds
load_transcript_json() succeeds
render_segments() succeeds
run_local_diarize(...) raises ValueError / OSError / AttributeError / AssertionError / other Exception
Streamlit script aborts before feedback.csv and before final UI render
```

### Evidence For

- WhisperX output exists and is valid.
- No `feedback.csv` exists.
- Current app catches `RuntimeError` around local diarization, but not unexpected exception classes.
- Third-party ML/audio packages commonly raise non-`RuntimeError` exceptions.
- This fully explains why the transcript was generated but the UI never reached speaker editing.

### Evidence Against

- We did not capture a Streamlit traceback for the incident.
- If the UI was truly still updating, H1.2 may be more likely.

### Likelihood

High.

### Fix

Keep the existing `except RuntimeError` path for expected, user-formatted diarization errors. Add a second, narrow `except Exception` only around optional local diarization:

```python
except RuntimeError as exc:
    local_diarization_error = str(exc)
    st.warning(str(exc))
    st.info("Das Transkript bleibt ohne Sprecherzuordnung erhalten und kann exportiert werden.")
    st.session_state.speaker_segments = []
except Exception as exc:
    local_diarization_error = (
        f"Unerwarteter Fehler in der lokalen Sprechererkennung: "
        f"{type(exc).__name__}: {exc}"
    )
    st.warning(local_diarization_error)
    st.info("Das Transkript bleibt ohne Sprecherzuordnung erhalten und kann exportiert werden.")
    st.session_state.speaker_segments = []
```

Do not catch `BaseException`; `KeyboardInterrupt` and `SystemExit` should not be swallowed.

### Tests

- Mock `run_local_diarize` to raise `ValueError("boom")`.
- Assert transcript segments remain available.
- Assert `speaker_segments` is empty.
- Assert feedback CSV can still be produced with `run.status = partial`.
- If direct Streamlit app testing is too heavy, factor the local-diarization handling into a small helper and test that helper.

---

## H1.2: Local diarization is still running or stuck after WhisperX

### Path

```text
run_whisperx() succeeds
load_transcript_json() succeeds
render_segments() succeeds
run_local_diarize(audio_path) keeps running
Streamlit button run does not finish
right-side speaker UI is not rendered
```

### Evidence For

- 78-minute audio can make diarization slow on CPU.
- `run_local_diarize()` emits no progress updates while `diarize()` runs.
- Streamlit is synchronous; the UI can appear inactive while a Python call is still running.
- No `feedback.csv` exists because feedback is written only after diarization and speaker assignment.

### Evidence Against

- At diagnosis time, no active WhisperX process was visible, but we did not conclusively identify a running diarization worker.
- Streamlit CPU was low in the later process snapshot, which may point more toward H1.3 if that was also true during the incident.

### Likelihood

High, tied with H1.3 until CPU/activity evidence distinguishes them.

### Fixes

Immediate low-risk fixes:

- Add a UI note before diarization:

```python
st.info(
    "Bei langen Dateien kann die Sprechererkennung mehrere Minuten dauern. "
    "Die Seite wirkt moeglicherweise inaktiv; das ist waehrend dieses Schritts normal."
)
```

- Write `run_state.json` before and after local diarization so the next incident has proof of the current stage.

Architectural fix:

- Split the app into a two-stage flow:
  1. Transcribe and render/export transcript.
  2. Start speaker diarization as a separate optional step.

Timeout fix:

- Use a real cancellable job boundary, preferably a subprocess, for local diarization.
- Avoid a naive `ThreadPoolExecutor` timeout around `diarize()` in the current `TemporaryDirectory` context. Timed-out Python threads keep running, and the temp WAV may be cleaned up while the thread still needs it.

### Tests

- Mock `run_local_diarize` to block or sleep and verify the app's stage marker remains at `local_diarization_started`.
- If job isolation is implemented, test timeout and cancellation at the subprocess level.
- Manual test with a long audio file and verify visible stage text during diarization.

---

## H0: WhisperX itself is still running or stuck

### Evidence For

- The UI text referred to WhisperX/local processing.
- Large MP3 files can take a long time on CPU.

### Evidence Against

- `whisperx.log` reached `Progress: 100.00%...`.
- Valid JSON was written.
- No active `whisperx` process was visible.

### Likelihood

Low for this incident.

### Fixes

- Add explicit stage marker `whisperx_finished` after JSON is loaded.
- Add stale-progress detection if WhisperX log/progress does not change for N minutes.

### Tests

- Mock `run_whisperx` success and assert stage changes to `whisperx_finished`.
- Mock a long-running WhisperX process with no progress and assert stale warning behavior if implemented.

---

## H3: Feedback CSV generation/write failed after successful post-processing

### Path

```text
diarization succeeds
speaker assignment succeeds
build_feedback_csv(...) succeeds or fails
feedback_path.write_text(...) raises OSError
script aborts before final UI render
```

### Evidence For

- No `feedback.csv` exists.
- Feedback writing is currently near the end of the run.

### Evidence Against

- JSON/log writing to the same run directory succeeded.
- This does not explain long "WhisperX processing" unless the UI was stale or the error page was missed.

### Likelihood

Medium-low.

### Fix

Wrap only the disk write. Keep the in-memory CSV so the download button can still work:

```python
try:
    feedback_path.write_text(feedback_csv, encoding="utf-8")
except OSError as exc:
    st.warning(f"Feedback-CSV konnte nicht geschrieben werden: {exc}")
st.session_state.feedback_csv = feedback_csv
```

### Tests

- Mock `Path.write_text` for `feedback.csv` to raise `OSError`.
- Assert no abort.
- Assert `st.session_state.feedback_csv` remains set.

---

## H4: Streamlit frontend/browser state is stale

### Path

```text
server has completed or crashed
browser websocket/front-end does not clearly reflect state
user sees stale processing UI
```

### Evidence For

- Long Streamlit runs can feel stale.
- The user did not have access to speaker controls, which may happen if a script run crashed before final render.

### Evidence Against

- No `feedback.csv` indicates the backend did not complete normally.
- H1.2/H1.3 explain the evidence more directly.

### Likelihood

Medium as a presentation layer, not as the root cause.

### Fixes

- Add stage markers on disk.
- Add clear stage text in the UI.
- Later: add a "Load latest completed WhisperX JSON" recovery affordance.

### Tests

- Manual browser test with a forced exception after WhisperX.
- Verify the error/warning is visible and the transcript can be recovered.

---

## H5: Memory pressure or CPU starvation after WhisperX

### Path

```text
WhisperX completes
diarization/model loading creates heavy CPU/RAM/swap pressure
Streamlit appears frozen or very slow
```

### Evidence For

- Long audio and ML models can create high memory pressure.

### Evidence Against

- Later process snapshot did not show a clearly busy Streamlit process.
- No direct memory-pressure evidence was captured.

### Likelihood

Low-medium.

### Fixes

- Capture stage timestamps and maybe process/resource snapshots in `run_state.json`.
- Do not add calibrated memory warnings until real data supports thresholds.

### Tests

- Not a first-line test target.

---

## H6: Disk or temporary-directory failure during diarization preparation

### Path

```text
prepare_audio_for_local_diarize()
-> tempfile.TemporaryDirectory(...)
-> ffmpeg writes converted WAV
-> disk/tempdir OSError or partial file issue
-> diarize() receives bad input or cleanup fails
```

### Evidence For

- Long files create larger temporary WAVs.
- Disk/temp-dir errors can surface as `OSError`, not always as `subprocess.CalledProcessError`.

### Evidence Against

- Likelihood is low unless disk space was actually constrained.
- ffmpeg non-zero exits are already caught and formatted.

### Likelihood

Low for this incident, relevant as a future robustness edge case.

### Fixes

- Catch `OSError` inside `run_local_diarize` and format it as local diarization failure.
- Optionally show free disk space in setup diagnostics later.

### Tests

- Mock `tempfile.TemporaryDirectory` or `subprocess.run` path to raise `OSError`.
- Assert a user-readable `RuntimeError` is produced.

---

## H2: Speaker assignment is computationally expensive

### Path

```text
assign_speakers_by_overlap(transcript_segments, speaker_segments)
```

### Evidence For

- The algorithm is O(transcript_segments * speaker_segments).
- Current run had 1148 transcript segments.

### Evidence Against

- At realistic speaker segment counts, this should be far below "stuck" territory.
- Claude measured approximate performance:
  - 1148 x 200 speaker segments: about 108 ms
  - 1148 x 1000 speaker segments: about 513 ms
  - 1148 x 10000 speaker segments: about 5 seconds
- Even pathological inputs are unlikely to explain a long perceived hang.

### Likelihood

Very low. Can be excluded for this incident unless real `speaker_segments_count` is unexpectedly enormous.

### Fixes

- Do not optimize now.
- Optionally log `speaker_segments_count` and assignment duration for diagnostics.

### Tests

- Existing correctness tests are enough.
- No sweep-line/performance optimization test is currently worth adding.

---

# Recommended Implementation Plan

## Step 1: Catch unexpected local diarization exceptions

- Priority: P1
- Goal: Preserve completed transcript even if `diarize()` throws an unexpected exception.
- Files: `app.py`
- Test idea: mock local diarization to raise `ValueError` and assert partial-success behavior.
- Risk: Low if scoped only to optional diarization and if `RuntimeError` remains a separate expected path.

## Step 2: Add `run_state.json` stage markers

- Priority: P2
- Goal: Make future incidents diagnosable from disk without relying on UI state.
- Files: likely `app.py`, optional helper in `src/transcript_mvp/pipeline.py` or a small new diagnostics module.
- Stages to record:
  - `run_created`
  - `whisperx_started`
  - `whisperx_finished`
  - `json_loaded`
  - `transcript_rendered`
  - `local_diarization_started`
  - `local_diarization_failed`
  - `local_diarization_finished`
  - `speaker_assignment_finished`
  - `feedback_written`
  - `complete`
- Test idea: unit test the stage writer; mocked app-flow test if practical.
- Risk: Low. Catch `OSError` for marker writes.

## Step 3: Harden feedback CSV write

- Priority: P2
- Goal: Feedback write failure should not block transcript export or speaker UI.
- Files: `app.py`
- Test idea: mock feedback path write to raise `OSError`; assert in-memory CSV remains available.
- Risk: Low.

## Step 4: Add a visible diarization-stage message

- Priority: P3
- Goal: Reduce user confusion during long local diarization.
- Files: `app.py`
- Test idea: no unit test needed; manual UI check is enough.
- Risk: Low.

## Step 5: Defer true diarization timeout until job isolation exists

- Priority: Later architecture work
- Goal: Provide real cancellation for stuck diarization.
- Recommended approach: subprocess/job boundary, not a simple thread timeout.
- Files: likely `src/transcript_mvp/local_diarization.py`, maybe a worker script.
- Test idea: worker timeout kills subprocess and leaves transcript recoverable.
- Risk: Medium. Requires careful process cleanup and temp-file lifetime management.

---

# Test Plan

## Must-Have For Next Patch

1. Unexpected diarization exception:
   - `run_local_diarize` raises `ValueError`.
   - transcript remains available.
   - diarization error is recorded as partial failure.

2. Stage marker writer:
   - writes valid JSON with stage, timestamp, and selected metadata.
   - ignores/captures `OSError` without crashing the app flow.

3. Feedback write failure:
   - `feedback.csv` disk write raises `OSError`.
   - download state remains set.
   - user warning is shown.

## Nice-To-Have

- `OSError` inside local diarization preparation becomes readable `RuntimeError`.
- Manual long-file check confirms visible diarization-stage message.
- Assignment duration logging, if diagnostics are added.

## Not Worth Testing Now

- Browser reload recovery.
- Sweep-line speaker assignment optimization.
- Large artificial 50,000-speaker-segment performance test.
- Memory-pressure threshold warnings.

---

# Current-Run Recovery Guidance

The completed WhisperX result from the incident is usable:

```text
data/runs/20090601_Interview_Joachim-U.1-1779530877/20090601_Interview_Joachim-U.1.json
```

Manual recovery:

1. Refresh the Streamlit browser tab.
2. Use "Oder vorhandenes WhisperX-JSON laden".
3. Load the JSON above.
4. Export transcript or rerun with `Nur transkribieren`.

Speaker labels may remain generic or missing because local diarization likely did not complete.

