# Edge Silence Trim Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in setting that trims only the beginning and ending silence beyond 5 seconds, using the same silence/activity standard as the existing long-silence auto-stop feature.

**Architecture:** Keep capture unchanged: `RawRecorder` writes complete temporary WAV files. When trim is enabled, `AudioRecorder` analyzes those raw temporary WAV files with the existing `SourceActivityDetector` and the same mic/loopback detector configs used by auto-stop; single-source mode trims from that source activity timeline, while `both` mode combines `mic_active OR loopback_active` before choosing one shared trim range. After trimming the temp source files, the existing normalize, mix, and final WAV/FLAC/MP3 export flow continues.

**Tech Stack:** Python, PyQt6, `numpy`, `soundfile`, `lameenc`, `unittest`, PyInstaller.

---

## Product Decisions

- Setting key: `trim_silence`.
- Default: `False`, because this mutates saved audio and should be opt-in.
- Silence standard: exactly the same detector logic as auto-stop. Do not add independent trim dB thresholds, independent spike filtering, or a separate calibration path.
- Mic source: use `SourceActivityDetector(**MIC_ACTIVITY_DETECTOR_CONFIG)`.
- Loopback source: use `SourceActivityDetector(**LOOPBACK_ACTIVITY_DETECTOR_CONFIG)`.
- Single-source mode: trim according to that one source's detector activity.
- Both mode: detect mic and loopback separately, then treat the recording as active when `mic_active OR loopback_active`; trim only where both sources are silent.
- Edge retention: preserve up to 5.0 seconds of silence at each edge. Example: if the detector sees first activity at 7.2s, remove audio before 2.2s.
- Scope: trim only beginning and ending silence. Do not remove silence in the middle.
- Processing order: detect and trim raw temp WAV files first, then normalize if enabled, mix if needed, and encode final output.
- All-silent recordings: if no source is active, keep the first 5 seconds as a safety clip so the output is never empty.
- Dependencies: no new packages. Use existing `numpy` and `soundfile`.
- PRD: no repo PRD file was found by `rg --files | rg -i "prd|requirements|spec|roadmap|plan"`, so update `README.md` only.
- Current workspace note: preserve the existing uncommitted settings, tray icon, screenshot, and single-instance changes.

## File Structure

- Modify `audio_recorder.py`
  - Add `RAW_RECORDER_CHUNK_FRAMES = 2048` and use it in `RawRecorder`.
  - Add `TRIM_SILENCE_DEFAULT_ENABLED` and `TRIM_EDGE_SILENCE_SECONDS`.
  - Add `normalize_trim_silence_enabled()`.
  - Add a shared `build_activity_detector(source_name)` helper and make `AudioRecorder._build_activity_detector()` call it.
  - Add trim helpers that build activity timelines from `SourceActivityDetector`, combine timelines for `both`, calculate trim bounds, and apply bounds to source temp WAV files.
  - Extend `AudioRecorder(trim_silence=False)` and trim metadata.
  - Apply trim at the start of `_prepare_source_wav()` before normalize and mix.
- Create `tests/test_audio_trim.py`
  - Unit tests prove trim uses `SourceActivityDetector` behavior and preserves 5 seconds around detector activity.
  - Integration tests prove single-source and both-source temp WAV trimming.
- Modify `gui.py`
  - Add a checkbox under `Post-Processing & Clipboard`: `Trim start/end silence over 5s`.
  - Load/save `trim_silence`.
  - Pass `trim_silence` into `AudioRecorder`.
- Modify `tests/test_gui_hotkeys.py`
  - Test default setting, load behavior, group placement, and `AudioRecorder` argument passing.
- Modify `README.md`
  - Document the opt-in trim setting and that it uses the same silence detection rules as auto-stop.

### Task 1: Add Detector-Based Trim Tests

**Files:**
- Create: `tests/test_audio_trim.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_audio_trim.py` with this content:

```python
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from audio_recorder import (
    AudioRecorder,
    SourceActivityDetector,
    calculate_trim_bounds_for_sources,
    trim_edge_silence_data,
)


SAMPLE_RATE = 1000
CHUNK_FRAMES = 100


def silence(seconds, channels=1):
    return np.zeros((int(seconds * SAMPLE_RATE), channels), dtype=np.float32)


def tone(seconds, channels=1, level=0.2):
    return np.full((int(seconds * SAMPLE_RATE), channels), level, dtype=np.float32)


class DetectorBasedTrimTests(unittest.TestCase):
    def test_trims_using_source_activity_detector_window(self):
        data = np.concatenate([silence(7), tone(2), silence(8)], axis=0)

        trimmed, stats = trim_edge_silence_data(
            data,
            SAMPLE_RATE,
            source_name="mic",
            chunk_frames=CHUNK_FRAMES,
        )

        self.assertEqual(len(trimmed), 12600)
        self.assertTrue(stats["applied"])
        self.assertAlmostEqual(stats["start_removed_seconds"], 2.2)
        self.assertAlmostEqual(stats["end_removed_seconds"], 2.2)
        self.assertEqual(stats["source_names"], ["mic"])

    def test_short_spike_is_ignored_by_same_active_window_rule(self):
        data = np.concatenate(
            [silence(7), tone(0.1), silence(2), tone(2), silence(7)],
            axis=0,
        )

        trimmed, stats = trim_edge_silence_data(
            data,
            SAMPLE_RATE,
            source_name="mic",
            chunk_frames=CHUNK_FRAMES,
        )

        self.assertEqual(len(trimmed), 12600)
        self.assertTrue(stats["applied"])
        self.assertAlmostEqual(stats["start_removed_seconds"], 4.3)
        self.assertAlmostEqual(stats["end_removed_seconds"], 1.2)

    def test_uses_requested_source_detector(self):
        data = np.concatenate([silence(7), tone(2), silence(8)], axis=0)

        with patch("audio_recorder.build_activity_detector") as build_detector:
            build_detector.return_value = SourceActivityDetector(
                margin_db=12.0,
                min_threshold_db=-60.0,
                max_threshold_db=-28.0,
            )
            trim_edge_silence_data(
                data,
                SAMPLE_RATE,
                source_name="loopback",
                chunk_frames=CHUNK_FRAMES,
            )

        build_detector.assert_called_once_with("loopback")

    def test_combines_both_sources_with_or_activity(self):
        mic = silence(17)
        loopback = np.concatenate([silence(7), tone(2), silence(8)], axis=0)

        start_frame, end_frame, stats = calculate_trim_bounds_for_sources(
            [("mic", mic), ("loopback", loopback)],
            SAMPLE_RATE,
            chunk_frames=CHUNK_FRAMES,
        )

        self.assertEqual(start_frame, 2200)
        self.assertEqual(end_frame, 14800)
        self.assertEqual(end_frame - start_frame, 12600)
        self.assertEqual(stats["source_names"], ["mic", "loopback"])

    def test_all_silent_recording_keeps_five_second_safety_clip(self):
        data = silence(12)

        trimmed, stats = trim_edge_silence_data(
            data,
            SAMPLE_RATE,
            source_name="mic",
            chunk_frames=CHUNK_FRAMES,
        )

        self.assertEqual(len(trimmed), SAMPLE_RATE * 5)
        self.assertTrue(stats["applied"])
        self.assertAlmostEqual(stats["start_removed_seconds"], 0.0)
        self.assertAlmostEqual(stats["end_removed_seconds"], 7.0)


class AudioRecorderTrimIntegrationTests(unittest.TestCase):
    def test_prepare_source_wav_applies_trim_before_single_source_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source_wav = os.path.join(temp_dir, "source.wav")
            data = np.concatenate([silence(7), tone(2), silence(8)], axis=0)
            sf.write(source_wav, data, SAMPLE_RATE, format="WAV", subtype="FLOAT")

            recorder = AudioRecorder(
                mic_id="mic1",
                source_mode="mic",
                output_folder=temp_dir,
                output_format="wav",
                quality="balanced",
                stereo=False,
                normalize=False,
                trim_silence=True,
            )
            recorder.temp_files = [source_wav]

            prepared_wav = recorder._prepare_source_wav("FLOAT")
            trimmed, sr = sf.read(prepared_wav, always_2d=True)

            self.assertEqual(sr, SAMPLE_RATE)
            self.assertEqual(len(trimmed), 12600)
            self.assertTrue(recorder.trim_silence_applied)
            self.assertAlmostEqual(recorder.trim_silence_removed_seconds, 4.4)

    def test_prepare_source_wav_applies_shared_bounds_before_both_mix(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            mic_wav = os.path.join(temp_dir, "mic.wav")
            loopback_wav = os.path.join(temp_dir, "loopback.wav")
            mic = silence(17)
            loopback = np.concatenate([silence(7), tone(2), silence(8)], axis=0)
            sf.write(mic_wav, mic, SAMPLE_RATE, format="WAV", subtype="FLOAT")
            sf.write(loopback_wav, loopback, SAMPLE_RATE, format="WAV", subtype="FLOAT")

            recorder = AudioRecorder(
                mic_id="mic1",
                source_mode="both",
                output_folder=temp_dir,
                output_format="wav",
                quality="balanced",
                stereo=False,
                normalize=False,
                trim_silence=True,
            )
            recorder.temp_files = [mic_wav, loopback_wav]

            mixed_wav = recorder._prepare_source_wav("FLOAT")
            mixed, sr = sf.read(mixed_wav, always_2d=True)

            self.assertEqual(sr, SAMPLE_RATE)
            self.assertEqual(len(mixed), 12600)
            self.assertTrue(recorder.trim_silence_applied)
            self.assertAlmostEqual(recorder.trim_silence_removed_seconds, 4.4)

    def test_prepare_source_wav_skips_trim_when_disabled(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source_wav = os.path.join(temp_dir, "source.wav")
            data = np.concatenate([silence(7), tone(2), silence(8)], axis=0)
            sf.write(source_wav, data, SAMPLE_RATE, format="WAV", subtype="FLOAT")

            recorder = AudioRecorder(
                mic_id="mic1",
                source_mode="mic",
                output_folder=temp_dir,
                output_format="wav",
                quality="balanced",
                stereo=False,
                normalize=False,
                trim_silence=False,
            )
            recorder.temp_files = [source_wav]

            prepared_wav = recorder._prepare_source_wav("FLOAT")
            untrimmed, sr = sf.read(prepared_wav, always_2d=True)

            self.assertEqual(sr, SAMPLE_RATE)
            self.assertEqual(len(untrimmed), len(data))
            self.assertFalse(recorder.trim_silence_applied)
            self.assertAlmostEqual(recorder.trim_silence_removed_seconds, 0.0)


class AudioRecorderTrimMetadataTests(unittest.TestCase):
    def test_finish_metadata_records_trim_state(self):
        recorder = AudioRecorder(
            mic_id="mic1",
            source_mode="mic",
            output_folder=".",
            trim_silence=True,
        )
        recorder.trim_silence_applied = True
        recorder.trim_silence_removed_seconds = 4.4

        metadata = recorder.build_finish_metadata()

        self.assertEqual(metadata["trim_silence_enabled"], True)
        self.assertEqual(metadata["trim_silence_keep_seconds"], 5.0)
        self.assertEqual(metadata["trim_silence_applied"], True)
        self.assertEqual(metadata["trim_silence_removed_seconds"], 4.4)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run focused tests and verify failure**

Run:

```bash
python -m unittest tests.test_audio_trim
```

Expected: FAIL with import errors for `calculate_trim_bounds_for_sources` and `trim_edge_silence_data`, or `TypeError` for missing `trim_silence`.

### Task 2: Implement Detector-Based Trim Core

**Files:**
- Modify: `audio_recorder.py`
- Test: `tests/test_audio_trim.py`

- [ ] **Step 1: Add shared constants**

In `audio_recorder.py`, after `NORMALIZE_LIMIT = 0.98`, add:

```python
RAW_RECORDER_CHUNK_FRAMES = 2048
```

After `AUTO_STOP_OPTIONS`, add:

```python
TRIM_SILENCE_DEFAULT_ENABLED = False
TRIM_EDGE_SILENCE_SECONDS = 5.0
```

In `RawRecorder.run()`, replace:

```python
                        data = mic.record(numframes=2048)
```

with:

```python
                        data = mic.record(numframes=RAW_RECORDER_CHUNK_FRAMES)
```

- [ ] **Step 2: Add setting normalization**

After `normalize_auto_stop_silence_seconds()`, add:

```python
def normalize_trim_silence_enabled(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return TRIM_SILENCE_DEFAULT_ENABLED
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("true", "1", "yes", "on"):
            return True
        if normalized in ("false", "0", "no", "off", ""):
            return False
    return bool(value)
```

- [ ] **Step 3: Add shared detector builder**

After `SourceActivityDetector`, add:

```python
def build_activity_detector(source_name):
    config = (
        LOOPBACK_ACTIVITY_DETECTOR_CONFIG
        if source_name == "loopback"
        else MIC_ACTIVITY_DETECTOR_CONFIG
    )
    return SourceActivityDetector(**config)
```

Replace `AudioRecorder._build_activity_detector()` with:

```python
    def _build_activity_detector(self, source_name):
        return build_activity_detector(source_name)
```

- [ ] **Step 4: Add detector timeline and trim helpers**

Add this code after `build_activity_detector()`:

```python
def _as_2d_audio(data):
    audio = np.asarray(data)
    if audio.ndim == 1:
        audio = audio.reshape(-1, 1)
    return audio


def build_source_activity_timeline(
    data,
    samplerate,
    source_name,
    chunk_frames=RAW_RECORDER_CHUNK_FRAMES,
):
    audio = _as_2d_audio(data).astype(np.float32, copy=False)
    detector = build_activity_detector(source_name)
    timeline = []

    for start_frame in range(0, len(audio), int(chunk_frames)):
        end_frame = min(len(audio), start_frame + int(chunk_frames))
        now = start_frame / float(samplerate)
        active = detector.update(audio[start_frame:end_frame], samplerate, now=now)
        timeline.append((start_frame, end_frame, bool(active)))

    return timeline


def _combine_activity_timelines(source_timelines, total_frames, chunk_frames):
    combined = []
    chunk_count = int(np.ceil(total_frames / float(chunk_frames))) if total_frames else 0

    for chunk_index in range(chunk_count):
        start_frame = chunk_index * int(chunk_frames)
        end_frame = min(total_frames, start_frame + int(chunk_frames))
        active = any(
            chunk_index < len(timeline) and bool(timeline[chunk_index][2])
            for timeline in source_timelines
        )
        combined.append((start_frame, end_frame, active))

    return combined


def _bounds_from_activity_timeline(
    timeline,
    total_frames,
    samplerate,
    keep_silence_seconds=TRIM_EDGE_SILENCE_SECONDS,
):
    keep_frames = max(1, int(round(float(samplerate) * keep_silence_seconds)))
    active_entries = [entry for entry in timeline if entry[2]]

    if total_frames <= 0:
        return 0, 0, {
            "applied": False,
            "start_removed_seconds": 0.0,
            "end_removed_seconds": 0.0,
        }

    if not active_entries:
        end_frame = min(total_frames, keep_frames)
        return 0, end_frame, {
            "applied": end_frame < total_frames,
            "start_removed_seconds": 0.0,
            "end_removed_seconds": (total_frames - end_frame) / float(samplerate),
        }

    first_active_frame = active_entries[0][0]
    last_active_frame = active_entries[-1][1]
    start_frame = max(0, first_active_frame - keep_frames)
    end_frame = min(total_frames, last_active_frame + keep_frames)

    return start_frame, end_frame, {
        "applied": start_frame > 0 or end_frame < total_frames,
        "start_removed_seconds": start_frame / float(samplerate),
        "end_removed_seconds": (total_frames - end_frame) / float(samplerate),
    }


def calculate_trim_bounds_for_sources(
    sources,
    samplerate,
    keep_silence_seconds=TRIM_EDGE_SILENCE_SECONDS,
    chunk_frames=RAW_RECORDER_CHUNK_FRAMES,
):
    normalized_sources = [
        (source_name, _as_2d_audio(data))
        for source_name, data in sources
    ]
    total_frames = max((len(data) for _source_name, data in normalized_sources), default=0)
    source_timelines = [
        build_source_activity_timeline(
            data,
            samplerate,
            source_name,
            chunk_frames=chunk_frames,
        )
        for source_name, data in normalized_sources
    ]
    combined_timeline = _combine_activity_timelines(
        source_timelines,
        total_frames,
        chunk_frames,
    )
    start_frame, end_frame, stats = _bounds_from_activity_timeline(
        combined_timeline,
        total_frames,
        samplerate,
        keep_silence_seconds=keep_silence_seconds,
    )
    stats["source_names"] = [source_name for source_name, _data in normalized_sources]
    return start_frame, end_frame, stats


def trim_edge_silence_data(
    data,
    samplerate,
    source_name="mic",
    keep_silence_seconds=TRIM_EDGE_SILENCE_SECONDS,
    chunk_frames=RAW_RECORDER_CHUNK_FRAMES,
):
    audio = _as_2d_audio(data)
    start_frame, end_frame, stats = calculate_trim_bounds_for_sources(
        [(source_name, audio)],
        samplerate,
        keep_silence_seconds=keep_silence_seconds,
        chunk_frames=chunk_frames,
    )
    return audio[start_frame:end_frame].copy(), stats
```

- [ ] **Step 5: Run focused tests and verify partial failure**

Run:

```bash
python -m unittest tests.test_audio_trim
```

Expected: detector helper tests pass, while `AudioRecorder` tests still fail because `AudioRecorder.__init__()` does not accept `trim_silence` yet.

### Task 3: Integrate Trim Into AudioRecorder

**Files:**
- Modify: `audio_recorder.py`
- Test: `tests/test_audio_trim.py`

- [ ] **Step 1: Extend `AudioRecorder.__init__()`**

Change the constructor parameter list from:

```python
        normalize=False,
        auto_stop_silence_seconds=AUTO_STOP_DEFAULT_SECONDS,
        on_finish_callback=None,
```

to:

```python
        normalize=False,
        trim_silence=TRIM_SILENCE_DEFAULT_ENABLED,
        auto_stop_silence_seconds=AUTO_STOP_DEFAULT_SECONDS,
        on_finish_callback=None,
```

After `self.normalize = normalize`, add:

```python
        self.trim_silence = normalize_trim_silence_enabled(trim_silence)
```

After `self.auto_stop_reason = None`, add:

```python
        self.trim_silence_applied = False
        self.trim_silence_removed_seconds = 0.0
```

- [ ] **Step 2: Add trim metadata to `build_finish_metadata()`**

Change `build_finish_metadata()` to return:

```python
    def build_finish_metadata(self):
        return {
            "auto_stop_enabled": self.auto_stop_silence_seconds is not None,
            "auto_stop_silence_seconds": self.auto_stop_silence_seconds,
            "auto_stop_triggered": self.auto_stop_triggered,
            "auto_stop_reason": self.auto_stop_reason,
            "trim_silence_enabled": self.trim_silence,
            "trim_silence_keep_seconds": TRIM_EDGE_SILENCE_SECONDS,
            "trim_silence_applied": self.trim_silence_applied,
            "trim_silence_removed_seconds": round(self.trim_silence_removed_seconds, 3),
        }
```

- [ ] **Step 3: Add source-file trim methods to `AudioRecorder`**

Add these methods before `_prepare_source_wav()`:

```python
    def _maybe_trim_temp_sources(self):
        if not self.trim_silence:
            self.trim_silence_applied = False
            self.trim_silence_removed_seconds = 0.0
            return

        source_names = self._active_source_names()
        source_paths = self.temp_files[:len(source_names)]
        if not source_paths:
            return

        try:
            loaded_sources = []
            file_infos = []
            samplerate = None

            for source_name, path in zip(source_names, source_paths):
                info = sf.info(path)
                data, sr = sf.read(path, always_2d=True)
                if samplerate is None:
                    samplerate = sr
                elif sr != samplerate:
                    raise ValueError("Cannot trim audio with different sample rates.")
                loaded_sources.append((source_name, data))
                file_infos.append((path, info, data))

            start_frame, end_frame, stats = calculate_trim_bounds_for_sources(
                loaded_sources,
                samplerate,
            )

            if stats["applied"]:
                for path, info, data in file_infos:
                    clipped_start = min(start_frame, len(data))
                    clipped_end = min(end_frame, len(data))
                    trimmed = data[clipped_start:clipped_end]
                    if len(trimmed) == 0:
                        safety_frames = min(
                            len(data),
                            int(round(samplerate * TRIM_EDGE_SILENCE_SECONDS)),
                        )
                        trimmed = data[:safety_frames]
                    sf.write(path, trimmed, samplerate, format=info.format, subtype=info.subtype)

            self.trim_silence_applied = bool(stats["applied"])
            self.trim_silence_removed_seconds = (
                float(stats["start_removed_seconds"]) + float(stats["end_removed_seconds"])
            )
        except Exception as e:
            print(f"Silence trim failed: {e}")
            self.trim_silence_applied = False
            self.trim_silence_removed_seconds = 0.0
```

- [ ] **Step 4: Call trim before normalize and mix**

At the top of `_prepare_source_wav()`, before:

```python
        if len(self.temp_files) == 2:
```

add:

```python
        self._maybe_trim_temp_sources()
```

Do not add any trim call after mix. Both-mode trim must happen on separate mic and loopback temp files before mixing.

- [ ] **Step 5: Run focused trim tests**

Run:

```bash
python -m unittest tests.test_audio_trim
```

Expected: PASS.

- [ ] **Step 6: Run audio output profile tests**

Run:

```bash
python -m unittest tests.test_audio_output_profile
```

Expected: PASS.

### Task 4: Add Settings UI and Recorder Wiring

**Files:**
- Modify: `gui.py`
- Modify: `tests/test_gui_hotkeys.py`

- [ ] **Step 1: Add GUI tests for the new setting**

Add these tests inside `SettingsWindowRecordingIndicatorTests` after the existing auto-stop tests:

```python
    def test_trim_silence_setting_defaults_off(self):
        window = self.make_window({})

        settings = window.get_settings()

        self.assertIs(settings["trim_silence"], False)

    def test_trim_silence_setting_can_be_enabled(self):
        window = self.make_window({"trim_silence": True})

        settings = window.get_settings()

        self.assertIs(settings["trim_silence"], True)

    def test_trim_silence_setting_is_in_post_processing_group(self):
        window = self.make_window({})

        post_group = window.findChild(QGroupBox, "postProcessingSettingsGroup")

        self.assertIsNotNone(post_group)
        self.assertIs(window.chk_trim_silence.parentWidget(), post_group)
```

In `TrayApplicationRecordingIndicatorTests.make_subject()`, add this key to the fake settings dict:

```python
                    "trim_silence": True,
```

Add this test after `test_start_recording_passes_auto_stop_setting`:

```python
    def test_start_recording_passes_trim_silence_setting(self):
        subject, _indicator = self.make_subject(show_indicator=True)

        with patch("gui.QIcon"), patch("gui.AudioRecorder") as AudioRecorder:
            TrayApplication.start_recording(subject, "both")

        self.assertEqual(AudioRecorder.call_args.kwargs["trim_silence"], True)
```

- [ ] **Step 2: Run GUI tests and verify failure**

Run:

```bash
python -m unittest tests.test_gui_hotkeys
```

Expected: FAIL with `AttributeError: 'SettingsWindow' object has no attribute 'chk_trim_silence'` or missing `trim_silence` key.

- [ ] **Step 3: Add the checkbox in `SettingsWindow.init_ui()`**

In `gui.py`, in the Post-Processing section, set an object name on the group:

```python
        group_post = QGroupBox("Post-Processing & Clipboard")
        group_post.setObjectName("postProcessingSettingsGroup")
```

After `self.chk_normalize = QCheckBox("Normalize Audio (Apply first)")`, add:

```python
        self.chk_trim_silence = QCheckBox("Trim start/end silence over 5s")
        self.chk_trim_silence.setChecked(False)
```

After `layout_post.addWidget(self.chk_normalize)`, add:

```python
        layout_post.addWidget(self.chk_trim_silence)
```

- [ ] **Step 4: Load and save `trim_silence`**

In `SettingsWindow.load_settings()`, after:

```python
        self.chk_normalize.setChecked(data.get("normalize", False))
```

add:

```python
        self.chk_trim_silence.setChecked(
            self._parse_bool_setting(data.get("trim_silence"))
        )
```

In `SettingsWindow.get_settings()`, after:

```python
            "normalize": self.chk_normalize.isChecked(),
```

add:

```python
            "trim_silence": self.chk_trim_silence.isChecked(),
```

- [ ] **Step 5: Pass the setting into `AudioRecorder`**

In `TrayApplication.start_recording()`, add this keyword argument after `normalize=settings['normalize'],`:

```python
            trim_silence=settings.get("trim_silence", False),
```

- [ ] **Step 6: Run GUI tests**

Run:

```bash
python -m unittest tests.test_gui_hotkeys
```

Expected: PASS.

### Task 5: Documentation, Verification, and Packaging

**Files:**
- Modify: `README.md`
- Test: full suite and PyInstaller packaging

- [ ] **Step 1: Update README**

In `README.md`, add this bullet near the existing normalize/post-processing text:

```markdown
- **Silence trim**: optional post-processing that trims only the start and end silence beyond 5 seconds, using the same mic/loopback activity detection rules as silence auto-stop.
```

In the Usage section, update the output/post-processing step so it mentions:

```markdown
5. In **Post-Processing & Clipboard**, enable normalization, edge-silence trim, clipboard copy, or delete-after-copy as needed.
```

- [ ] **Step 2: Run all tests**

Run:

```bash
python -m unittest discover -s tests
```

Expected: all tests pass.

- [ ] **Step 3: Build the EXE**

Run:

```bash
pyinstaller --clean --noconfirm MeetRec.spec
```

Expected: build exits with code 0 and produces `dist/MeetRec.exe`. Existing non-blocking `pycparser.lextab` and `pycparser.yacctab` warnings can remain if the EXE is produced.

- [ ] **Step 4: Verify the artifact exists**

Run:

```powershell
Get-Item -LiteralPath 'dist\MeetRec.exe' | Select-Object FullName,Length,LastWriteTime
```

Expected: output includes `D:\Git\quickaudiorecorder\dist\MeetRec.exe` with a fresh `LastWriteTime`.

- [ ] **Step 5: Commit the completed feature if requested**

Run:

```bash
git add audio_recorder.py gui.py README.md tests/test_audio_trim.py tests/test_gui_hotkeys.py
git commit -m "Add detector-based edge silence trim"
```

Expected: commit succeeds. Do not stage unrelated untracked assets unless the user explicitly wants all current work committed.

## Self-Review

- Spec coverage: the plan implements automatic trim of beginning and ending silence over 5 seconds, adds a setting switch, and explicitly reuses auto-stop's mic/loopback detector logic.
- Placeholder scan: no task depends on unspecified functions; every new test and helper has concrete code.
- Type consistency: setting key is consistently `trim_silence`; detector builder is consistently `build_activity_detector`; recorder argument is consistently `trim_silence`.
- Risk control: default is off, both mode trims only where mic and loopback are both silent, and all-silent recordings keep a 5-second safety clip.
