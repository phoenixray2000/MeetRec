# Silence Auto-Stop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop an active recording automatically after the selected duration of continuous silence.

**Architecture:** Keep the current recording architecture: `RawRecorder` threads capture chunks and write WAV data, while `AudioRecorder` owns lifecycle and finalization. Add per-source rolling activity detectors and a single auto-stop controller in `audio_recorder.py`; `RawRecorder` only reports audio chunks, and `AudioRecorder` sets its own stop event when silence timeout is reached. Add one minimal settings combo in `gui.py` with Off / 5 / 10 / 20 minutes, defaulting to 10 minutes.

**Tech Stack:** Python 3.12, `soundcard`, `soundfile`, `numpy`, `PyQt6`, `unittest`. No new runtime packages.

---

## Final Design Decisions

- Scope is only long-silence auto-stop. Do not implement silence trimming or remove silent sections from saved audio.
- Use existing `numpy` math for RMS/dB detection. Do not add `webrtcvad`, `pydub`, `librosa`, `scipy`, or `ffmpeg`.
- Maintain one `SourceActivityDetector` per active input source.
- Single-source modes stop when that source is silent for the configured duration.
- `both` mode stops only when mic and loopback are both silent for the configured duration.
- Detection uses a rolling noise floor, not a fixed global threshold.
- Do not run an initial calibration period. Meetings may begin with speech, so the detector continuously estimates noise from recent low-percentile dB values.
- `RawRecorder` must not touch UI and must not join/stop recorder threads. It only calls back with `source_name` and `data`.
- `AudioRecorder.request_auto_stop()` sets `stop_event`; the main `AudioRecorder.run()` path then stops and joins all `RawRecorder` threads normally.
- Store auto-stop metadata in memory on `AudioRecorder.finish_metadata`. Do not create sidecar JSON files in v1 because clipboard/delete-after-copy behavior currently assumes one output artifact.

## File Structure

- Modify `audio_recorder.py`
  - Add `AUTO_STOP_OPTIONS`, `AUTO_STOP_DEFAULT_SECONDS`, source detector configs, `normalize_auto_stop_silence_seconds()`.
  - Add `SourceActivityDetector`.
  - Add `AutoStopController`.
  - Extend `RawRecorder` with `source_name` and `on_audio_data`.
  - Extend `AudioRecorder` with auto-stop setup, activity reporting, and in-memory finish metadata.
- Modify `gui.py`
  - Add an "Auto-stop after silence" combo.
  - Persist `auto_stop_silence_seconds`.
  - Pass the setting into `AudioRecorder`.
- Add `tests/test_silence_auto_stop.py`
  - Unit tests for detector, controller, and `AudioRecorder.report_activity()`.
- Modify `tests/test_gui_hotkeys.py`
  - Settings persistence tests and `AudioRecorder` constructor argument coverage.
- Modify `README.md`
  - Document the new setting and explicitly state there are no new dependencies.

---

### Task 1: Add Detector and Controller Tests

**Files:**
- Create: `tests/test_silence_auto_stop.py`
- Modify later: `audio_recorder.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_silence_auto_stop.py` with this complete content:

```python
import unittest

import numpy as np

from audio_recorder import (
    AUTO_STOP_DEFAULT_SECONDS,
    AutoStopController,
    AudioRecorder,
    SourceActivityDetector,
    normalize_auto_stop_silence_seconds,
)


def chunk(level, frames=2048, channels=1):
    return np.full((frames, channels), level, dtype=np.float32)


class SourceActivityDetectorTests(unittest.TestCase):
    def test_silence_below_dynamic_threshold_is_inactive(self):
        detector = SourceActivityDetector(
            margin_db=9.0,
            min_threshold_db=-55.0,
            max_threshold_db=-30.0,
            history_seconds=30.0,
            active_window_seconds=1.0,
            active_ratio=0.2,
        )

        active = detector.update(chunk(0.00001), samplerate=2048, now=0.0)

        self.assertFalse(active)
        self.assertEqual(detector.noise_floor_db, -100.0)
        self.assertEqual(detector.threshold_db, -55.0)

    def test_signal_above_dynamic_threshold_becomes_active(self):
        detector = SourceActivityDetector(
            margin_db=9.0,
            min_threshold_db=-55.0,
            max_threshold_db=-30.0,
            history_seconds=30.0,
            active_window_seconds=1.0,
            active_ratio=0.2,
        )

        detector.update(chunk(0.00001), samplerate=2048, now=0.0)
        active = detector.update(chunk(0.05), samplerate=2048, now=1.0)

        self.assertTrue(active)
        self.assertGreater(detector.current_db, detector.threshold_db)

    def test_isolated_spike_does_not_make_window_active(self):
        detector = SourceActivityDetector(
            margin_db=9.0,
            min_threshold_db=-55.0,
            max_threshold_db=-30.0,
            history_seconds=30.0,
            active_window_seconds=1.0,
            active_ratio=0.2,
        )

        now = 0.0
        for _ in range(9):
            active = detector.update(chunk(0.00001, frames=1600), samplerate=16000, now=now)
            now += 0.1
            self.assertFalse(active)

        active = detector.update(chunk(0.2, frames=1600), samplerate=16000, now=now)

        self.assertFalse(active)


class AutoStopControllerTests(unittest.TestCase):
    def test_normalizes_supported_values(self):
        self.assertIsNone(normalize_auto_stop_silence_seconds(None))
        self.assertIsNone(normalize_auto_stop_silence_seconds("off"))
        self.assertEqual(normalize_auto_stop_silence_seconds("300"), 300)
        self.assertEqual(normalize_auto_stop_silence_seconds(600), 600)
        self.assertEqual(normalize_auto_stop_silence_seconds(1200), 1200)
        self.assertEqual(normalize_auto_stop_silence_seconds("invalid"), AUTO_STOP_DEFAULT_SECONDS)

    def test_disabled_controller_never_stops(self):
        controller = AutoStopController(
            silence_seconds=None,
            min_record_seconds=60.0,
            start_ts=0.0,
        )

        self.assertFalse(controller.update({"mic": False}, now=10000.0))

    def test_respects_min_record_seconds(self):
        controller = AutoStopController(
            silence_seconds=300,
            min_record_seconds=60.0,
            start_ts=0.0,
        )

        self.assertFalse(controller.update({"mic": False}, now=59.0))

    def test_stops_after_single_source_silence_timeout(self):
        controller = AutoStopController(
            silence_seconds=300,
            min_record_seconds=60.0,
            start_ts=0.0,
        )

        self.assertFalse(controller.update({"mic": False}, now=299.0))
        self.assertTrue(controller.update({"mic": False}, now=300.0))
        self.assertEqual(controller.reason(), "silence_timeout_5min")

    def test_any_active_source_resets_silence_timer(self):
        controller = AutoStopController(
            silence_seconds=300,
            min_record_seconds=60.0,
            start_ts=0.0,
        )

        self.assertFalse(controller.update({"mic": True}, now=200.0))
        self.assertFalse(controller.update({"mic": False}, now=499.0))
        self.assertTrue(controller.update({"mic": False}, now=500.0))

    def test_both_mode_only_stops_when_all_sources_are_silent(self):
        controller = AutoStopController(
            silence_seconds=300,
            min_record_seconds=60.0,
            start_ts=0.0,
        )

        self.assertFalse(controller.update({"mic": False, "loopback": True}, now=200.0))
        self.assertFalse(controller.update({"mic": False, "loopback": False}, now=499.0))
        self.assertTrue(controller.update({"mic": False, "loopback": False}, now=500.0))


class AudioRecorderAutoStopTests(unittest.TestCase):
    def test_single_source_activity_can_request_auto_stop(self):
        recorder = AudioRecorder(
            mic_id="mic1",
            source_mode="mic",
            output_folder=".",
            auto_stop_silence_seconds=300,
        )
        recorder._setup_auto_stop()
        recorder.auto_stop_controller = AutoStopController(
            silence_seconds=300,
            min_record_seconds=60.0,
            start_ts=0.0,
        )

        recorder.report_activity("mic", chunk(0.00001), now=300.0)

        self.assertTrue(recorder.stop_event.is_set())
        self.assertTrue(recorder.auto_stop_triggered)
        self.assertEqual(recorder.auto_stop_reason, "silence_timeout_5min")

    def test_both_mode_activity_from_one_source_prevents_auto_stop(self):
        recorder = AudioRecorder(
            mic_id="mic1",
            source_mode="both",
            output_folder=".",
            auto_stop_silence_seconds=300,
        )
        recorder._setup_auto_stop()
        recorder.auto_stop_controller = AutoStopController(
            silence_seconds=300,
            min_record_seconds=60.0,
            start_ts=0.0,
        )

        recorder.source_active["loopback"] = True
        recorder.report_activity("mic", chunk(0.00001), now=300.0)

        self.assertFalse(recorder.stop_event.is_set())
        self.assertFalse(recorder.auto_stop_triggered)

    def test_finish_metadata_records_auto_stop_state(self):
        recorder = AudioRecorder(
            mic_id="mic1",
            source_mode="mic",
            output_folder=".",
            auto_stop_silence_seconds=600,
        )
        recorder.auto_stop_triggered = True
        recorder.auto_stop_reason = "silence_timeout_10min"

        metadata = recorder.build_finish_metadata()

        self.assertEqual(
            metadata,
            {
                "auto_stop_enabled": True,
                "auto_stop_silence_seconds": 600,
                "auto_stop_triggered": True,
                "auto_stop_reason": "silence_timeout_10min",
            },
        )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```powershell
python -m unittest tests.test_silence_auto_stop
```

Expected: FAIL with import errors for `SourceActivityDetector`, `AutoStopController`, and `normalize_auto_stop_silence_seconds`.

- [ ] **Step 3: Commit the failing tests only if your workflow uses red commits**

Most local work here should not commit red tests. Keep this as a checkpoint:

```powershell
git diff -- tests/test_silence_auto_stop.py
```

Expected: the new test file is the only change.

---

### Task 2: Implement Detector and Controller in `audio_recorder.py`

**Files:**
- Modify: `audio_recorder.py`
- Test: `tests/test_silence_auto_stop.py`

- [ ] **Step 1: Add imports and constants**

Modify the imports near the top of `audio_recorder.py`:

```python
from collections import deque
```

Add these constants after the normalize constants:

```python
AUTO_STOP_OFF = None
AUTO_STOP_5_MINUTES = 300
AUTO_STOP_10_MINUTES = 600
AUTO_STOP_20_MINUTES = 1200
AUTO_STOP_DEFAULT_SECONDS = AUTO_STOP_10_MINUTES
AUTO_STOP_MIN_RECORD_SECONDS = 60.0

AUTO_STOP_OPTIONS = [
    (AUTO_STOP_OFF, "Off"),
    (AUTO_STOP_5_MINUTES, "5 minutes"),
    (AUTO_STOP_10_MINUTES, "10 minutes"),
    (AUTO_STOP_20_MINUTES, "20 minutes"),
]

MIC_ACTIVITY_DETECTOR_CONFIG = {
    "margin_db": 9.0,
    "min_threshold_db": -55.0,
    "max_threshold_db": -30.0,
}

LOOPBACK_ACTIVITY_DETECTOR_CONFIG = {
    "margin_db": 12.0,
    "min_threshold_db": -60.0,
    "max_threshold_db": -28.0,
}
```

- [ ] **Step 2: Add value normalization**

Add this helper after `describe_output_profile()`:

```python
def normalize_auto_stop_silence_seconds(value):
    if value is None or value is False:
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("", "off", "none", "false", "0"):
            return None
        try:
            value = int(normalized)
        except ValueError:
            return AUTO_STOP_DEFAULT_SECONDS

    try:
        seconds = int(value)
    except (TypeError, ValueError):
        return AUTO_STOP_DEFAULT_SECONDS

    supported = {seconds for seconds, _label in AUTO_STOP_OPTIONS if seconds is not None}
    if seconds in supported:
        return seconds
    return AUTO_STOP_DEFAULT_SECONDS
```

- [ ] **Step 3: Add `SourceActivityDetector`**

Add this class before `RawRecorder`:

```python
class SourceActivityDetector:
    def __init__(
        self,
        margin_db=10.0,
        min_threshold_db=-55.0,
        max_threshold_db=-28.0,
        history_seconds=30.0,
        active_window_seconds=1.0,
        active_ratio=0.2,
    ):
        self.margin_db = float(margin_db)
        self.min_threshold_db = float(min_threshold_db)
        self.max_threshold_db = float(max_threshold_db)
        self.history_seconds = float(history_seconds)
        self.active_window_seconds = float(active_window_seconds)
        self.active_ratio = float(active_ratio)
        self.recent_db_values = deque()
        self.recent_activity = deque()
        self.current_db = -120.0
        self.noise_floor_db = -120.0
        self.threshold_db = self.min_threshold_db
        self.active = False

    def update(self, data, samplerate, now=None):
        if now is None:
            now = time.monotonic()

        audio = np.asarray(data, dtype=np.float32)
        if audio.size == 0:
            return self.active

        rms = float(np.sqrt(np.mean(audio ** 2) + 1e-12))
        self.current_db = max(20.0 * np.log10(rms + 1e-12), -100.0)

        self.recent_db_values.append((now, self.current_db))
        history_cutoff = now - self.history_seconds
        while self.recent_db_values and self.recent_db_values[0][0] < history_cutoff:
            self.recent_db_values.popleft()

        db_values = np.array([value for _timestamp, value in self.recent_db_values], dtype=np.float32)
        self.noise_floor_db = float(np.percentile(db_values, 20))
        dynamic_threshold = self.noise_floor_db + self.margin_db
        self.threshold_db = min(
            max(dynamic_threshold, self.min_threshold_db),
            self.max_threshold_db,
        )

        block_active = self.current_db >= self.threshold_db
        self.recent_activity.append((now, block_active))
        active_cutoff = now - self.active_window_seconds
        while self.recent_activity and self.recent_activity[0][0] < active_cutoff:
            self.recent_activity.popleft()

        total_blocks = len(self.recent_activity)
        active_blocks = sum(1 for _timestamp, is_active in self.recent_activity if is_active)
        self.active = total_blocks > 0 and (active_blocks / total_blocks) >= self.active_ratio
        return self.active
```

- [ ] **Step 4: Add `AutoStopController`**

Add this class after `SourceActivityDetector`:

```python
class AutoStopController:
    def __init__(
        self,
        silence_seconds=None,
        min_record_seconds=AUTO_STOP_MIN_RECORD_SECONDS,
        start_ts=None,
    ):
        self.silence_seconds = normalize_auto_stop_silence_seconds(silence_seconds)
        self.min_record_seconds = float(min_record_seconds)
        self.start_ts = time.monotonic() if start_ts is None else float(start_ts)
        self.last_active_ts = self.start_ts

    def update(self, source_active, now=None):
        if now is None:
            now = time.monotonic()

        if self.silence_seconds is None:
            return False

        any_active = any(bool(active) for active in source_active.values())
        if any_active:
            self.last_active_ts = now
            return False

        if now - self.start_ts < self.min_record_seconds:
            return False

        return now - self.last_active_ts >= self.silence_seconds

    def reason(self):
        if self.silence_seconds is None:
            return None
        minutes = int(self.silence_seconds // 60)
        return f"silence_timeout_{minutes}min"
```

- [ ] **Step 5: Run detector/controller tests**

Run:

```powershell
python -m unittest tests.test_silence_auto_stop.SourceActivityDetectorTests tests.test_silence_auto_stop.AutoStopControllerTests
```

Expected: PASS.

---

### Task 3: Wire Auto-Stop Into Recording Lifecycle

**Files:**
- Modify: `audio_recorder.py`
- Test: `tests/test_silence_auto_stop.py`

- [ ] **Step 1: Extend `RawRecorder.__init__`**

Replace the existing `RawRecorder.__init__` signature and body with:

```python
def __init__(
    self,
    device,
    filepath,
    samplerate=44100,
    channels=2,
    subtype="PCM_16",
    source_name=None,
    on_audio_data=None,
):
    super().__init__()
    self.device = device
    self.filepath = filepath
    self.samplerate = samplerate
    self.channels = channels
    self.subtype = subtype
    self.source_name = source_name
    self.on_audio_data = on_audio_data
    self.stop_event = threading.Event()
    self.error = None
```

- [ ] **Step 2: Report audio chunks from `RawRecorder.run()`**

Replace the loop body inside `RawRecorder.run()` with:

```python
while not self.stop_event.is_set():
    data = mic.record(numframes=2048)
    if self.on_audio_data and self.source_name:
        self.on_audio_data(self.source_name, data)
    f_wav.write(data)
```

- [ ] **Step 3: Extend `AudioRecorder.__init__`**

Add this parameter after `normalize=False`:

```python
auto_stop_silence_seconds=AUTO_STOP_DEFAULT_SECONDS,
```

Add these fields after `self.normalize = normalize`:

```python
self.auto_stop_silence_seconds = normalize_auto_stop_silence_seconds(auto_stop_silence_seconds)
```

Add these fields after `self.recorders = []`:

```python
self.activity_lock = threading.Lock()
self.activity_detectors = {}
self.source_active = {}
self.auto_stop_controller = None
self.auto_stop_triggered = False
self.auto_stop_reason = None
self.finish_metadata = self.build_finish_metadata()
```

- [ ] **Step 4: Add auto-stop helper methods to `AudioRecorder`**

Add these methods before `_get_device()`:

```python
def _active_source_names(self):
    if self.source_mode == "both":
        return ["mic", "loopback"]
    if self.source_mode == "loopback":
        return ["loopback"]
    return ["mic"]

def _build_activity_detector(self, source_name):
    config = (
        LOOPBACK_ACTIVITY_DETECTOR_CONFIG
        if source_name == "loopback"
        else MIC_ACTIVITY_DETECTOR_CONFIG
    )
    return SourceActivityDetector(**config)

def _setup_auto_stop(self):
    source_names = self._active_source_names()
    self.activity_detectors = {
        source_name: self._build_activity_detector(source_name)
        for source_name in source_names
    }
    self.source_active = {source_name: False for source_name in source_names}
    self.auto_stop_controller = AutoStopController(
        silence_seconds=self.auto_stop_silence_seconds,
        min_record_seconds=AUTO_STOP_MIN_RECORD_SECONDS,
    )
    self.auto_stop_triggered = False
    self.auto_stop_reason = None

def report_activity(self, source_name, data, now=None):
    with self.activity_lock:
        detector = self.activity_detectors.get(source_name)
        if detector is None:
            return

        self.source_active[source_name] = detector.update(
            data,
            self.profile["sample_rate"],
            now=now,
        )

        if self.auto_stop_controller and self.auto_stop_controller.update(self.source_active, now=now):
            self.request_auto_stop(self.auto_stop_controller.reason())

def request_auto_stop(self, reason):
    if self.auto_stop_triggered:
        return
    self.auto_stop_triggered = True
    self.auto_stop_reason = reason
    self.stop_event.set()

def build_finish_metadata(self):
    return {
        "auto_stop_enabled": self.auto_stop_silence_seconds is not None,
        "auto_stop_silence_seconds": self.auto_stop_silence_seconds,
        "auto_stop_triggered": self.auto_stop_triggered,
        "auto_stop_reason": self.auto_stop_reason,
    }
```

- [ ] **Step 5: Initialize auto-stop in `AudioRecorder.run()`**

In `AudioRecorder.run()`, after `samplerate`, `channels`, and `subtype` are assigned, add:

```python
self._setup_auto_stop()
```

- [ ] **Step 6: Pass source names and callbacks into `RawRecorder`**

Replace the `both` mode recorder creation with:

```python
self.recorders.append(
    RawRecorder(
        dev_mic,
        t1,
        samplerate=samplerate,
        channels=channels,
        subtype=subtype,
        source_name="mic",
        on_audio_data=self.report_activity,
    )
)
self.recorders.append(
    RawRecorder(
        dev_loop,
        t2,
        samplerate=samplerate,
        channels=channels,
        subtype=subtype,
        source_name="loopback",
        on_audio_data=self.report_activity,
    )
)
```

Replace the `loopback` recorder creation with:

```python
self.recorders.append(
    RawRecorder(
        dev,
        t1,
        samplerate=samplerate,
        channels=channels,
        subtype=subtype,
        source_name="loopback",
        on_audio_data=self.report_activity,
    )
)
```

Replace the `mic` recorder creation with:

```python
self.recorders.append(
    RawRecorder(
        dev,
        t1,
        samplerate=samplerate,
        channels=channels,
        subtype=subtype,
        source_name="mic",
        on_audio_data=self.report_activity,
    )
)
```

- [ ] **Step 7: Update finish metadata in `finally`**

In `AudioRecorder.run()` `finally`, before the callback, add:

```python
self.finish_metadata = self.build_finish_metadata()
```

- [ ] **Step 8: Run recording lifecycle tests**

Run:

```powershell
python -m unittest tests.test_silence_auto_stop
```

Expected: PASS.

- [ ] **Step 9: Commit audio core**

```powershell
git add audio_recorder.py tests/test_silence_auto_stop.py
git commit -m "Add silence auto-stop detection core"
```

Expected: commit succeeds.

---

### Task 4: Add Minimal UI Setting and Persistence

**Files:**
- Modify: `gui.py`
- Modify: `tests/test_gui_hotkeys.py`

- [ ] **Step 1: Update imports in `gui.py`**

Add these names to the existing `from audio_recorder import (...)` list:

```python
AUTO_STOP_DEFAULT_SECONDS,
AUTO_STOP_OPTIONS,
normalize_auto_stop_silence_seconds,
```

- [ ] **Step 2: Add the settings combo**

In `SettingsWindow.init_ui()`, inside the "Tray Icon Behavior" group after `self.combo_left_click`, add:

```python
self.combo_auto_stop = QComboBox()
for seconds, label in AUTO_STOP_OPTIONS:
    self.combo_auto_stop.addItem(label, seconds)
```

Then add the row after the left-click row:

```python
layout_tray.addRow("Auto-stop after silence:", self.combo_auto_stop)
```

- [ ] **Step 3: Add a combo setter helper**

Add this method after `_set_combo_by_data()`:

```python
def _set_auto_stop_combo(self, value):
    seconds = normalize_auto_stop_silence_seconds(value)
    for idx in range(self.combo_auto_stop.count()):
        if self.combo_auto_stop.itemData(idx) == seconds:
            self.combo_auto_stop.setCurrentIndex(idx)
            return

    default_idx = self.combo_auto_stop.findData(AUTO_STOP_DEFAULT_SECONDS)
    if default_idx >= 0:
        self.combo_auto_stop.setCurrentIndex(default_idx)
```

- [ ] **Step 4: Load persisted setting with 10-minute default**

In `SettingsWindow.load_settings()`, after tray click mode loading, add:

```python
self._set_auto_stop_combo(
    data.get("auto_stop_silence_seconds", AUTO_STOP_DEFAULT_SECONDS)
)
```

- [ ] **Step 5: Save the setting**

In `SettingsWindow.get_settings()`, add:

```python
"auto_stop_silence_seconds": self.combo_auto_stop.currentData(),
```

Place it near `tray_click_mode` and `show_recording_indicator`.

- [ ] **Step 6: Pass the setting into `AudioRecorder`**

In `TrayApplication.start_recording()`, add this constructor argument:

```python
auto_stop_silence_seconds=settings.get("auto_stop_silence_seconds"),
```

Place it after `normalize=settings['normalize']`.

- [ ] **Step 7: Add UI persistence tests**

In `tests/test_gui_hotkeys.py`, add these tests to `SettingsWindowRecordingIndicatorTests`:

```python
def test_auto_stop_setting_defaults_to_ten_minutes(self):
    window = self.make_window({})

    settings = window.get_settings()

    self.assertEqual(settings["auto_stop_silence_seconds"], 600)

def test_auto_stop_setting_can_be_disabled(self):
    window = self.make_window({"auto_stop_silence_seconds": None})

    settings = window.get_settings()

    self.assertIsNone(settings["auto_stop_silence_seconds"])

def test_auto_stop_setting_loads_five_minutes(self):
    window = self.make_window({"auto_stop_silence_seconds": 300})

    settings = window.get_settings()

    self.assertEqual(settings["auto_stop_silence_seconds"], 300)
```

- [ ] **Step 8: Update fake settings**

In `TrayApplicationRecordingIndicatorTests.make_subject()`, add this setting:

```python
"auto_stop_silence_seconds": 600,
```

- [ ] **Step 9: Assert `AudioRecorder` receives the setting**

Add this test to `TrayApplicationRecordingIndicatorTests`:

```python
def test_start_recording_passes_auto_stop_setting(self):
    subject, _indicator = self.make_subject(show_indicator=True)

    with patch("gui.QIcon"), patch("gui.AudioRecorder") as AudioRecorder:
        TrayApplication.start_recording(subject, "both")

    self.assertEqual(AudioRecorder.call_args.kwargs["auto_stop_silence_seconds"], 600)
```

- [ ] **Step 10: Run GUI tests**

Run:

```powershell
python -m unittest tests.test_gui_hotkeys
```

Expected: PASS.

- [ ] **Step 11: Commit UI setting**

```powershell
git add gui.py tests/test_gui_hotkeys.py
git commit -m "Add silence auto-stop setting"
```

Expected: commit succeeds.

---

### Task 5: Documentation and Full Verification

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update README feature list**

In the `Control` feature list, add:

```markdown
- Auto-stop: End recording automatically after 5, 10, or 20 minutes of silence.
```

- [ ] **Step 2: Update README usage steps**

After the settings step that describes selecting microphone/output settings, add:

```markdown
3. Choose **Auto-stop after silence** if you want MeetRec to end long silent recordings automatically. The default is 10 minutes; choose Off, 5 minutes, 10 minutes, or 20 minutes.
```

Renumber the following usage steps.

- [ ] **Step 3: Keep dependency list unchanged**

Verify this line still has no new package:

```markdown
- `pip install PyQt6 soundcard soundfile numpy lameenc keyboard`
```

- [ ] **Step 4: Run all tests**

Run:

```powershell
python -m unittest discover -s tests
```

Expected: all tests pass.

- [ ] **Step 5: Build the Windows package**

Run:

```powershell
pyinstaller --noconfirm MeetRec.spec
```

Expected: `dist\MeetRec.exe` is created.

- [ ] **Step 6: Manual smoke test**

Run `dist\MeetRec.exe` and verify:

- Settings opens.
- Auto-stop combo shows Off / 5 minutes / 10 minutes / 20 minutes.
- Default is 10 minutes for a fresh `settings.json`.
- Starting mic recording still works.
- Stopping manually still saves the file.
- For a short local test, temporarily set `auto_stop_silence_seconds` to `65` in code or test-only branch, stay silent after the 60-second minimum, and confirm auto-stop fires without UI thread errors. Do not ship unsupported `65` as a UI option.

- [ ] **Step 7: Commit docs**

```powershell
git add README.md
git commit -m "Document silence auto-stop"
```

Expected: commit succeeds.

---

## Self-Review

**Spec coverage**

- Per-source rolling baselines: Task 2 adds `SourceActivityDetector` per source.
- `both` mode AND silence rule: Task 1 and Task 3 cover `mic OR loopback` active and all-silent timeout.
- No initial calibration: detector updates rolling history continuously from first chunk.
- 1-second active window and 20 percent active ratio: Task 2 implements `active_window_seconds=1.0` and `active_ratio=0.2`.
- UI options Off / 5 / 10 / 20 minutes: Task 4 adds combo from `AUTO_STOP_OPTIONS`.
- Default 10 minutes: Task 4 tests default `600`.
- Metadata: Task 3 stores in-memory `finish_metadata`.
- No new packages: Task 5 verifies README dependency list remains unchanged.

**Placeholder scan**

- No placeholder markers or fill-later notes.
- All tests and implementation snippets name concrete functions, classes, and settings keys.

**Type consistency**

- Setting key is consistently `auto_stop_silence_seconds`.
- Internal state fields are consistently `auto_stop_triggered`, `auto_stop_reason`, and `finish_metadata`.
- Source names are consistently `"mic"` and `"loopback"`.
