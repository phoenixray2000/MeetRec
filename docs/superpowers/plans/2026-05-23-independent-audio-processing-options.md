# Independent Audio Processing Options Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Normalize, AEC, RNNoise denoise, and Ducking independent recording options, default only Normalize/AEC/Ducking on, make RNNoise conservative at 35% dry/wet, and fix single-source Normalize so it uses the same source-leveling path and metadata as Both mode.

**Architecture:** Keep the existing recorder pipeline shape: microphone processing happens before source leveling and mixing, loopback processing happens before final mix, and final trim still acts after mix. Add a small reusable ducking function to `audio_processing.py`; keep RNNoise delay compensation inside `denoise.py`; keep user settings in `gui.py` and runtime state in `audio_recorder.py`.

**Tech Stack:** Python 3.12, PyQt6 settings UI, soundfile/numpy/scipy audio processing, RNNoise via `ctypes`, unittest.

---

## Requirements And Evidence

- User decision: Normalize, AEC, RNNoise denoise, and Ducking are independent options.
- Default options: Normalize on, AEC on, Ducking on, RNNoise off.
- RNNoise strength when enabled: 35% dry/wet mix, not full-wet.
- Ducking should be stronger than the previous offline `-6 dB` experiment; use `-9 dB` reduction with fast attack and smooth release.
- Fix Normalize: single-source recordings must use source leveling with stats, not the old opaque `_normalize_audio()` path.
- No repo PRD file was found with `rg --encoding utf-8 -n "PRD|prd|product|requirement|需求|产品" . -g "*.md"`. Update `README.md` instead because this changes user-visible behavior.

## File Structure

- Modify `audio_processing.py`
  - Add `DuckingConfig`.
  - Add `duck_reference_audio(reference, trigger, samplerate, config)`.
  - Keep source leveling and echo suppression unchanged.
- Modify `denoise.py`
  - Add `latency_ms` to `NoiseReductionConfig`.
  - Align RNNoise wet output before dry/wet mix.
  - Return latency metadata in `stats`.
- Modify `audio_recorder.py`
  - Add Normalize default constant and Ducking constants/default.
  - Set `NOISE_REDUCTION_MIX = 0.35`.
  - Add `ducking` constructor option and metadata fields.
  - Use source leveling for single-source normalize.
  - Apply ducking to leveled loopback in Both mode only when enabled.
- Modify `gui.py`
  - Add Ducking checkbox under Post-Processing.
  - Change default Normalize/AEC/Ducking to on, RNNoise off.
  - Save/load/pass `ducking`.
- Modify `tests/test_denoise.py`
  - Cover RNNoise latency alignment and 35% dry/wet behavior.
- Modify `tests/test_audio_processing.py`
  - Cover ducking behavior.
- Modify `tests/test_audio_output_profile.py`
  - Cover single-source source-leveling stats, RNNoise mix, ducking integration, and metadata.
- Modify `tests/test_gui_hotkeys.py`
  - Cover defaults, Ducking UI, and `AudioRecorder` argument passing.
- Modify `README.md`
  - Document independent options and new defaults.

---

### Task 1: Update UI Defaults And Setting Plumbing

**Files:**
- Modify: `gui.py`
- Test: `tests/test_gui_hotkeys.py`

- [ ] **Step 1: Write failing GUI default and Ducking tests**

Add these tests near the existing post-processing setting tests in `tests/test_gui_hotkeys.py`:

```python
    def test_post_processing_defaults_enable_normalize_echo_and_ducking(self):
        window = self.make_window({})

        settings = window.get_settings()

        self.assertIs(settings["normalize"], True)
        self.assertIs(settings["echo_suppression"], True)
        self.assertIs(settings["ducking"], True)
        self.assertIs(settings["noise_reduction"], False)

    def test_ducking_setting_can_be_disabled(self):
        window = self.make_window({"ducking": False})

        settings = window.get_settings()

        self.assertIs(settings["ducking"], False)

    def test_ducking_setting_is_in_post_processing_group(self):
        window = self.make_window({})

        post_group = window.findChild(QGroupBox, "postProcessingSettingsGroup")

        self.assertIsNotNone(post_group)
        self.assertIs(window.chk_ducking.parentWidget(), post_group)
```

Update `TrayApplicationRecordingIndicatorTests.make_subject()` fake settings to include Ducking:

```python
                    "ducking": True,
```

Update `test_start_recording_passes_new_audio_settings`:

```python
        self.assertEqual(AudioRecorder.call_args.kwargs["echo_suppression"], True)
        self.assertEqual(AudioRecorder.call_args.kwargs["noise_reduction"], True)
        self.assertEqual(AudioRecorder.call_args.kwargs["ducking"], True)
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```powershell
python -m unittest tests.test_gui_hotkeys
```

Expected: FAIL because `chk_ducking` and the `ducking` setting do not exist, and current Normalize/AEC defaults are `False`.

- [ ] **Step 3: Add GUI constants and Ducking checkbox**

In `gui.py`, add defaults near `CONFIG_FILE`:

```python
NORMALIZE_DEFAULT_ENABLED = True
ECHO_SUPPRESSION_UI_DEFAULT_ENABLED = True
NOISE_REDUCTION_UI_DEFAULT_ENABLED = False
DUCKING_DEFAULT_ENABLED = True
```

In `SettingsWindow.init_ui()`, after `self.chk_noise_reduction` setup and before debug pipeline setup, add:

```python
        self.chk_ducking = QCheckBox("Lower system audio while microphone is active")
        self.chk_ducking.setToolTip(
            "In Both mode, automatically lowers the loopback track while the "
            "microphone track is active so local speech stays intelligible. "
            "This does not change microphone recordings or loopback-only recordings."
        )
```

Add the widget to the post-processing layout immediately after noise reduction:

```python
        layout_post.addWidget(self.chk_ducking)
```

- [ ] **Step 4: Load, save, and pass the Ducking setting**

In `SettingsWindow.load_settings()`, replace the three existing post-processing defaults with explicit defaults and add Ducking:

```python
        self.chk_normalize.setChecked(
            self._parse_bool_setting(data.get("normalize", NORMALIZE_DEFAULT_ENABLED))
        )
        self.chk_echo_suppression.setChecked(
            self._parse_bool_setting(
                data.get("echo_suppression", ECHO_SUPPRESSION_UI_DEFAULT_ENABLED)
            )
        )
        self.chk_noise_reduction.setChecked(
            self._parse_bool_setting(
                data.get("noise_reduction", NOISE_REDUCTION_UI_DEFAULT_ENABLED)
            )
        )
        self.chk_ducking.setChecked(
            self._parse_bool_setting(data.get("ducking", DUCKING_DEFAULT_ENABLED))
        )
```

In `SettingsWindow.get_settings()`, add:

```python
            "ducking": self.chk_ducking.isChecked(),
```

In `TrayApplication.start_recording()`, pass the setting to `AudioRecorder` next to `noise_reduction`:

```python
            ducking=settings.get("ducking", True),
```

- [ ] **Step 5: Run tests and verify pass**

Run:

```powershell
python -m unittest tests.test_gui_hotkeys
```

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add gui.py tests/test_gui_hotkeys.py
git commit -m "Add independent ducking setting defaults"
```

---

### Task 2: Make RNNoise Conservative And Latency-Aligned

**Files:**
- Modify: `denoise.py`
- Modify: `audio_recorder.py`
- Test: `tests/test_denoise.py`
- Test: `tests/test_audio_output_profile.py`

- [ ] **Step 1: Write failing RNNoise latency and mix tests**

Replace `_FakeBackend` in `tests/test_denoise.py` with a configurable fake:

```python
class _FakeBackend:
    samplerate = 48000
    frame_size = 480

    def __init__(self, gain=0.5, delay_samples=0):
        self.gain = gain
        self.delay_samples = delay_samples

    def process(self, mono_48k_pm1):
        wet = np.asarray(mono_48k_pm1, dtype=np.float32) * self.gain
        if self.delay_samples <= 0:
            return wet
        return np.concatenate(
            [
                np.zeros(self.delay_samples, dtype=np.float32),
                wet[:-self.delay_samples],
            ]
        )
```

In `DenoiseTests.setUp()`, instantiate the no-delay fake:

```python
        denoise._BACKEND = _FakeBackend()
```

Add these tests:

```python
    def test_latency_compensation_aligns_wet_signal_before_mix(self):
        denoise._BACKEND = _FakeBackend(gain=1.0, delay_samples=960)
        data = np.zeros((48000, 1), dtype=np.float32)
        data[12000, 0] = 0.8

        out, stats = reduce_noise(
            data,
            48000,
            NoiseReductionConfig(enabled=True, mix=1.0, latency_ms=20.0),
        )

        self.assertEqual(int(np.argmax(np.abs(out[:, 0]))), 12000)
        self.assertEqual(stats["latency_ms"], 20.0)

    def test_latency_compensation_preserves_length_when_advancing_wet(self):
        denoise._BACKEND = _FakeBackend(gain=1.0, delay_samples=960)
        data = np.full((48000, 1), 0.2, dtype=np.float32)

        out, stats = reduce_noise(
            data,
            48000,
            NoiseReductionConfig(enabled=True, mix=0.35, latency_ms=20.0),
        )

        self.assertEqual(out.shape, data.shape)
        self.assertTrue(stats["applied"])
        self.assertEqual(stats["mix"], 0.35)
```

Add this test to `tests/test_audio_output_profile.py`:

```python
    def test_denoise_uses_conservative_35_percent_mix(self):
        import numpy as np

        recorder = self._make_recorder("wav", "balanced", stereo=False)
        recorder.noise_reduction = True
        data = np.full((16000, 1), 0.1, dtype=np.float32)

        with patch("audio_recorder.denoise.reduce_noise") as reduce_noise:
            reduce_noise.return_value = (data.copy(), {"applied": True, "reason": "applied"})

            recorder._denoise_mic_data(data, 16000)

        config = reduce_noise.call_args.args[2]
        self.assertAlmostEqual(config.mix, 0.35)
        self.assertAlmostEqual(config.latency_ms, 20.0)
```

Ensure `tests/test_audio_output_profile.py` imports `patch` from `unittest.mock` if it does not already:

```python
from unittest.mock import patch
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```powershell
python -m unittest tests.test_denoise tests.test_audio_output_profile
```

Expected: FAIL because `NoiseReductionConfig.latency_ms` does not exist and `NOISE_REDUCTION_MIX` is still `1.0`.

- [ ] **Step 3: Implement latency compensation in `denoise.py`**

Change `NoiseReductionConfig`:

```python
@dataclass(frozen=True)
class NoiseReductionConfig:
    enabled: bool = False
    mix: float = 1.0
    latency_ms: float = 20.0
```

Add this helper below `_resample()`:

```python
def _advance_audio(data, frames):
    frames = int(max(0, frames))
    if frames <= 0 or len(data) == 0:
        return data
    if frames >= len(data):
        return np.zeros_like(data)
    return np.concatenate([data[frames:], np.zeros(frames, dtype=np.float32)])
```

In `reduce_noise()`, after `mix = ...`, add:

```python
    latency_ms = float(max(0.0, config.latency_ms))
    latency_frames = int(round(latency_ms * float(samplerate) / 1000.0))
```

After `wet` is resized to `len(mono)`, advance it before mixing:

```python
            wet = _advance_audio(wet.astype(np.float32, copy=False), latency_frames)
            out[:, ch] = (1.0 - mix) * mono + mix * wet
```

Return latency metadata:

```python
        "mix": round(mix, 3), "latency_ms": round(latency_ms, 3),
        "channels": int(audio.shape[1]), "samplerate": int(samplerate),
```

- [ ] **Step 4: Set application RNNoise mix to 35%**

In `audio_recorder.py`, change constants:

```python
NOISE_REDUCTION_MIX = 0.35
NOISE_REDUCTION_LATENCY_MS = 20.0
```

In `_denoise_mic_data()`, pass latency:

```python
            NoiseReductionConfig(
                enabled=self.noise_reduction,
                mix=NOISE_REDUCTION_MIX,
                latency_ms=NOISE_REDUCTION_LATENCY_MS,
            ),
```

- [ ] **Step 5: Run tests and verify pass**

Run:

```powershell
python -m unittest tests.test_denoise tests.test_audio_output_profile
```

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add denoise.py audio_recorder.py tests/test_denoise.py tests/test_audio_output_profile.py
git commit -m "Use conservative aligned RNNoise mixing"
```

---

### Task 3: Fix Normalize By Using Source Leveling For Single Sources

**Files:**
- Modify: `audio_recorder.py`
- Test: `tests/test_audio_output_profile.py`

- [ ] **Step 1: Write failing single-source source-leveling test**

Add this test to `tests/test_audio_output_profile.py`:

```python
    def test_single_mic_normalize_uses_source_leveling_stats(self):
        import os
        import tempfile

        import numpy as np
        import soundfile as sf

        with tempfile.TemporaryDirectory() as temp_dir:
            sr = 16000
            mic_file = os.path.join(temp_dir, "mic.wav")
            mic = np.zeros((sr * 2, 1), dtype=np.float32)
            mic[sr // 2:sr] = 0.02
            sf.write(mic_file, mic, sr, format="WAV", subtype="FLOAT")

            recorder = self._make_recorder("wav", "balanced", stereo=False)
            recorder.source_mode = "mic"
            recorder.normalize = True
            recorder.noise_reduction = False
            recorder.temp_files = [mic_file]

            prepared = recorder._prepare_source_wav("FLOAT")
            output, _ = sf.read(prepared, always_2d=True)

            self.assertNotEqual(prepared, mic_file)
            self.assertIn("mic", recorder.source_leveling_stats)
            self.assertTrue(recorder.source_leveling_stats["mic"]["applied"])
            self.assertGreater(float(np.median(np.abs(output[sr // 2:sr, 0]))), 0.10)
            self.assertLessEqual(float(np.max(np.abs(output))), 0.9801)
```

Add this loopback variant:

```python
    def test_single_loopback_normalize_uses_source_leveling_stats(self):
        import os
        import tempfile

        import numpy as np
        import soundfile as sf

        with tempfile.TemporaryDirectory() as temp_dir:
            sr = 16000
            loop_file = os.path.join(temp_dir, "loop.wav")
            loopback = np.zeros((sr * 2, 1), dtype=np.float32)
            loopback[sr // 2:sr] = 0.015
            sf.write(loop_file, loopback, sr, format="WAV", subtype="FLOAT")

            recorder = self._make_recorder("wav", "balanced", stereo=False)
            recorder.source_mode = "loopback"
            recorder.normalize = True
            recorder.temp_files = [loop_file]

            prepared = recorder._prepare_source_wav("FLOAT")

            self.assertNotEqual(prepared, loop_file)
            self.assertIn("loopback", recorder.source_leveling_stats)
            self.assertTrue(recorder.source_leveling_stats["loopback"]["applied"])
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```powershell
python -m unittest tests.test_audio_output_profile
```

Expected: FAIL because single-source normalize mutates the original temp file through `_normalize_audio()` and leaves `source_leveling_stats` empty.

- [ ] **Step 3: Add helper to write leveled source WAV**

In `audio_recorder.py`, add this helper after `_level_source_data()`:

```python
    def _write_leveled_source_wav(self, source_wav, source_name, subtype):
        info = sf.info(source_wav)
        data, sr = sf.read(source_wav, always_2d=True)
        leveled = self._level_source_data(data, sr, source_name)
        label = f"leveled_{source_name}"
        self._record_debug_audio(label, data=leveled, samplerate=sr, subtype=info.subtype)
        leveled_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        sf.write(leveled_wav, leveled, sr, format=info.format, subtype=subtype or info.subtype)
        self.temp_files.append(leveled_wav)
        return leveled_wav
```

- [ ] **Step 4: Route single-source normalize through source leveling**

In `_prepare_source_wav()`, replace:

```python
        if self.normalize:
            self._normalize_audio(source_wav)
        return self._maybe_trim_final_wav(source_wav)
```

with:

```python
        if self.normalize:
            source_name = "loopback" if self.source_mode == "loopback" else "mic"
            source_wav = self._write_leveled_source_wav(source_wav, source_name, subtype)
        return self._maybe_trim_final_wav(source_wav)
```

Keep `_normalize_audio()` and `_normalize_audio_data()` for compatibility with existing direct unit tests, but do not use them in the recorder pipeline.

- [ ] **Step 5: Run tests and verify pass**

Run:

```powershell
python -m unittest tests.test_audio_output_profile tests.test_audio_trim
```

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add audio_recorder.py tests/test_audio_output_profile.py
git commit -m "Use source leveling for single-source normalize"
```

---

### Task 4: Add Strong Loopback Ducking As An Independent Both-Mode Option

**Files:**
- Modify: `audio_processing.py`
- Modify: `audio_recorder.py`
- Test: `tests/test_audio_processing.py`
- Test: `tests/test_audio_output_profile.py`

- [ ] **Step 1: Write failing ducking core tests**

Add imports in `tests/test_audio_processing.py`:

```python
from audio_processing import DuckingConfig, duck_reference_audio
```

If the file already imports multiple names from `audio_processing`, add `DuckingConfig` and `duck_reference_audio` to that existing import list instead.

Add these tests:

```python
class DuckingTests(unittest.TestCase):
    def test_disabled_ducking_returns_reference_unchanged(self):
        sr = 16000
        reference = np.full((sr, 1), 0.2, dtype=np.float32)
        trigger = np.full((sr, 1), 0.2, dtype=np.float32)

        out, stats = duck_reference_audio(
            reference,
            trigger,
            sr,
            DuckingConfig(enabled=False),
        )

        np.testing.assert_allclose(out, reference)
        self.assertFalse(stats["applied"])
        self.assertEqual(stats["reason"], "disabled")

    def test_ducking_reduces_reference_while_trigger_is_active(self):
        sr = 16000
        reference = np.full((sr * 2, 1), 0.2, dtype=np.float32)
        trigger = np.zeros((sr * 2, 1), dtype=np.float32)
        trigger[sr // 2:sr] = 0.12

        out, stats = duck_reference_audio(
            reference,
            trigger,
            sr,
            DuckingConfig(
                enabled=True,
                reduction_db=9.0,
                threshold=0.02,
                attack_ms=20.0,
                release_ms=120.0,
            ),
        )

        self.assertTrue(stats["applied"])
        self.assertLess(float(np.mean(np.abs(out[sr // 2:sr, 0]))), 0.09)
        self.assertGreater(float(np.mean(np.abs(out[:sr // 4, 0]))), 0.18)
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```powershell
python -m unittest tests.test_audio_processing
```

Expected: FAIL because `DuckingConfig` and `duck_reference_audio` do not exist.

- [ ] **Step 3: Implement ducking core**

Add to `audio_processing.py` after `SourceLevelingConfig`:

```python
@dataclass(frozen=True)
class DuckingConfig:
    enabled: bool = False
    reduction_db: float = 9.0
    threshold: float = 0.012
    attack_ms: float = 35.0
    release_ms: float = 250.0
```

Add this function after `level_active_source()`:

```python
def duck_reference_audio(reference, trigger, samplerate, config):
    reference = as_2d_float_audio(reference)
    trigger = as_2d_float_audio(trigger)
    if not config.enabled:
        return reference.copy(), {"applied": False, "reason": "disabled"}

    frames = min(len(reference), len(trigger))
    if frames == 0:
        return reference.copy(), {"applied": False, "reason": "empty_audio"}

    out = reference.copy()
    trigger_mono = np.max(np.abs(trigger[:frames]), axis=1)
    threshold = float(max(0.0, config.threshold))
    active = trigger_mono >= threshold
    if not np.any(active):
        return reference.copy(), {
            "applied": False,
            "reason": "no_trigger_activity",
            "threshold": round(threshold, 6),
        }

    reduction_gain = 10.0 ** (-float(config.reduction_db) / 20.0)
    hop = max(1, int(round(float(samplerate) * 0.01)))
    attack = np.exp(-hop / max(1.0, float(samplerate) * float(config.attack_ms) / 1000.0))
    release = np.exp(-hop / max(1.0, float(samplerate) * float(config.release_ms) / 1000.0))
    envelope = np.ones(frames, dtype=np.float32)
    current = 1.0

    for start in range(0, frames, hop):
        end = min(frames, start + hop)
        desired = reduction_gain if bool(np.any(active[start:end])) else 1.0
        coef = attack if desired < current else release
        current = desired + (current - desired) * coef
        envelope[start:end] = current

    out[:frames] = out[:frames] * envelope[:, None]
    return out, {
        "applied": True,
        "reason": "applied",
        "reduction_db": round(float(config.reduction_db), 3),
        "threshold": round(threshold, 6),
        "active_seconds": round(float(np.count_nonzero(active)) / float(samplerate), 3),
        "min_gain": round(float(np.min(envelope)), 6),
    }
```

- [ ] **Step 4: Write failing recorder integration tests**

At the top of `tests/test_audio_output_profile.py`, ensure `patch` is imported:

```python
from unittest.mock import patch
```

Add this integration test:

```python
    def test_both_mode_ducks_loopback_after_source_leveling_when_enabled(self):
        import os
        import tempfile

        import numpy as np
        import soundfile as sf

        with tempfile.TemporaryDirectory() as temp_dir:
            sr = 16000
            mic_file = os.path.join(temp_dir, "mic.wav")
            loop_file = os.path.join(temp_dir, "loop.wav")
            mic = np.zeros((sr * 2, 1), dtype=np.float32)
            mic[sr // 2:sr] = 0.08
            loopback = np.full((sr * 2, 1), 0.08, dtype=np.float32)
            sf.write(mic_file, mic, sr, format="WAV", subtype="FLOAT")
            sf.write(loop_file, loopback, sr, format="WAV", subtype="FLOAT")

            recorder = self._make_recorder("wav", "balanced", stereo=False)
            recorder.source_mode = "both"
            recorder.normalize = False
            recorder.echo_suppression = False
            recorder.noise_reduction = False
            recorder.ducking = True
            recorder.temp_files = [mic_file, loop_file]

            recorder._prepare_source_wav("FLOAT")

            self.assertTrue(recorder.ducking_applied)
            self.assertEqual(recorder.ducking_reason, "applied")
            self.assertLess(recorder.ducking_stats["min_gain"], 0.5)
```

Add metadata assertions to `test_metadata_sidecar_records_pipeline_state`:

```python
            recorder.ducking = True
            recorder.ducking_applied = True
            recorder.ducking_reason = "applied"
            recorder.ducking_stats = {"reduction_db": 9.0}
```

and:

```python
            self.assertTrue(metadata["ducking_enabled"])
            self.assertTrue(metadata["ducking_applied"])
            self.assertEqual(metadata["ducking_reason"], "applied")
            self.assertEqual(metadata["ducking_stats"], {"reduction_db": 9.0})
```

Update `test_debug_pipeline_exports_intermediate_tracks` so the expected loopback debug file matches Ducking being enabled:

```python
            recorder.ducking = True
```

and replace the loopback artifact assertion:

```python
            self.assertIn("06_ducked_loopback.wav", exported)
```

- [ ] **Step 5: Run tests and verify failure**

Run:

```powershell
python -m unittest tests.test_audio_processing tests.test_audio_output_profile
```

Expected: FAIL because `AudioRecorder` has no `ducking` fields and does not call `duck_reference_audio()`.

- [ ] **Step 6: Integrate Ducking into `audio_recorder.py`**

Update imports:

```python
from audio_processing import (
    DuckingConfig,
    EchoSuppressionConfig,
    SourceLevelingConfig,
    duck_reference_audio,
    level_active_source,
    suppress_reference_echo,
    trim_edge_silence_data,
)
```

Add constants near existing post-processing constants:

```python
NORMALIZE_DEFAULT_ENABLED = True
ECHO_SUPPRESSION_DEFAULT_ENABLED = True
NOISE_REDUCTION_DEFAULT_ENABLED = False
DUCKING_DEFAULT_ENABLED = True
DUCKING_REDUCTION_DB = 9.0
DUCKING_THRESHOLD = 0.012
DUCKING_ATTACK_MS = 35.0
DUCKING_RELEASE_MS = 250.0
```

Change `AudioRecorder.__init__()` signature defaults:

```python
        normalize=NORMALIZE_DEFAULT_ENABLED,
        echo_suppression=ECHO_SUPPRESSION_DEFAULT_ENABLED,
        noise_reduction=NOISE_REDUCTION_DEFAULT_ENABLED,
        ducking=DUCKING_DEFAULT_ENABLED,
```

Assign and normalize ducking:

```python
        self.normalize = _normalize_bool_setting(normalize, NORMALIZE_DEFAULT_ENABLED)
        self.echo_suppression = normalize_echo_suppression_enabled(echo_suppression)
        self.noise_reduction = normalize_noise_reduction_enabled(noise_reduction)
        self.ducking = _normalize_bool_setting(ducking, DUCKING_DEFAULT_ENABLED)
```

Initialize state:

```python
        self.ducking_applied = False
        self.ducking_reason = "not_run"
        self.ducking_stats = {}
```

Add metadata fields in `_build_metadata()`:

```python
            "ducking_enabled": self.ducking,
            "ducking_applied": self.ducking_applied,
            "ducking_reason": self.ducking_reason,
            "ducking_stats": self.ducking_stats,
```

Add helper after `_denoise_mic_data()`:

```python
    def _duck_loopback_data(self, loopback_data, mic_data, samplerate):
        ducked, stats = duck_reference_audio(
            loopback_data,
            mic_data,
            samplerate,
            DuckingConfig(
                enabled=bool(self.ducking),
                reduction_db=DUCKING_REDUCTION_DB,
                threshold=DUCKING_THRESHOLD,
                attack_ms=DUCKING_ATTACK_MS,
                release_ms=DUCKING_RELEASE_MS,
            ),
        )
        self.ducking_applied = bool(stats.get("applied"))
        self.ducking_reason = str(stats.get("reason", "unknown"))
        self.ducking_stats = stats
        return ducked
```

In `_prepare_both_source_wav()`, after loopback source leveling and before recording `leveled_loopback`, apply ducking:

```python
        loopback_data = self._level_source_data(loopback_data, loopback_sr, "loopback")
        if self.ducking:
            loopback_data = self._duck_loopback_data(loopback_data, mic_data, loopback_sr)
        else:
            self.ducking_applied = False
            self.ducking_reason = "disabled"
            self.ducking_stats = {"applied": False, "reason": "disabled"}
```

Change the debug label to reflect ducking when enabled:

```python
        self._record_debug_audio(
            "ducked_loopback" if self.ducking else "leveled_loopback",
            data=loopback_data,
            samplerate=loopback_sr,
            subtype=subtype,
        )
```

For `mic_reference` and single-source paths, set Ducking to skipped:

```python
            self.ducking_applied = False
            self.ducking_reason = "not_mixed"
            self.ducking_stats = {"applied": False, "reason": "not_mixed"}
```

For single-source `_prepare_source_wav()`, after echo suppression state reset, add:

```python
        self.ducking_applied = False
        self.ducking_reason = "single_source"
        self.ducking_stats = {"applied": False, "reason": "single_source"}
```

- [ ] **Step 7: Run tests and verify pass**

Run:

```powershell
python -m unittest tests.test_audio_processing tests.test_audio_output_profile
```

Expected: PASS.

- [ ] **Step 8: Commit**

```powershell
git add audio_processing.py audio_recorder.py tests/test_audio_processing.py tests/test_audio_output_profile.py
git commit -m "Add independent loopback ducking"
```

---

### Task 5: Update Documentation And Full Verification

**Files:**
- Modify: `README.md`
- Test: all tests

- [ ] **Step 1: Update README Post-Processing section**

Replace the Post-Processing bullets in `README.md` with:

```markdown
- **Normalize Audio**: enabled by default. Performs source leveling before mixing: microphone and system audio are each raised toward a target level using active audio only and a maximum gain cap. This keeps quiet microphone speech intelligible without normalizing long silence or residual noise as the reference.
- **Reduce speaker echo (Both mode)**: enabled by default. Uses system audio as a reference to subtract speaker leakage from the microphone track before denoise, leveling, ducking, and mixing. It only affects Both mode and Mic + Echo Reference mode.
- **Reduce microphone noise (RNNoise)**: optional and off by default. Applies conservative RNNoise speech denoising to the microphone track after echo suppression and before leveling, using a 35% wet mix with latency compensation so weak speech is less likely to be removed. Requires `rnnoise.dll`; if the library is unavailable, recording still succeeds and denoising is skipped.
- **Lower system audio while microphone is active**: enabled by default. In Both mode, lowers loopback audio while the microphone track is active so local speech stays intelligible. This is side-chain ducking; it does not affect microphone-only or loopback-only recordings.
- **Silence trim**: optional final post-processing that trims only the start and end silence beyond 5 seconds, applied to the final mixed output rather than the raw sources.
- **Pipeline diagnostics**: every recording writes a same-name JSON sidecar with post-processing stats. Optional debug audio export saves raw and intermediate WAV files in a same-name `_debug` folder for echo/denoise/leveling/ducking comparison.
```

In the Usage section, replace step 8 with:

```markdown
8. In **Post-Processing & Clipboard**, decide whether to use source leveling/normalization, echo reduction, microphone noise reduction, loopback ducking, debug audio export, final edge-silence trim, clipboard copy, or delete-after-copy.
```

- [ ] **Step 2: Run focused tests**

Run:

```powershell
python -m unittest tests.test_denoise tests.test_audio_processing tests.test_audio_output_profile tests.test_gui_hotkeys
```

Expected: PASS.

- [ ] **Step 3: Run full test suite**

Run:

```powershell
python -m unittest discover -s tests
```

Expected: all tests PASS.

- [ ] **Step 4: Build package for manual listening**

Run:

```powershell
npm.cmd run dist
```

Expected: `dist\MeetRec\MeetRec.exe` exists. If the build fails because `MeetRec.exe` is locked, stop only the running `MeetRec.exe` process and rerun the command.

- [ ] **Step 5: Manual regression recording**

Use the packaged app to record one typical Both-mode sample with default post-processing. Verify the JSON sidecar contains:

```json
{
  "echo_suppression_enabled": true,
  "noise_reduction_enabled": false,
  "source_leveling_enabled": true,
  "ducking_enabled": true
}
```

Then enable RNNoise and record another sample. Verify the sidecar contains:

```json
{
  "noise_reduction_enabled": true,
  "noise_reduction_applied": true,
  "noise_reduction_reason": "applied"
}
```

Listen for these acceptance criteria:

- With default RNNoise off, local speech is continuous and loopback lowers under speech.
- With RNNoise on, speech remains continuous; if artifacts appear, they are weaker than the previous full-wet behavior.
- Debug folder includes separate intermediate tracks for raw mic, raw loopback, AEC mic, denoised mic when enabled, leveled mic, ducked loopback when Ducking is enabled, and mixed pre-trim.

- [ ] **Step 6: Commit**

```powershell
git add README.md
git commit -m "Document independent audio processing defaults"
```

---

## Self-Review

- Spec coverage: Normalize/AEC/RNNoise/Ducking independence is covered by Tasks 1 and 4; RNNoise 35% and latency compensation by Task 2; default only Normalize/AEC/Ducking by Task 1; single-source Normalize fix by Task 3; docs by Task 5.
- Placeholder scan: no placeholder markers or deferred-work wording are present.
- Type consistency: `ducking`, `ducking_applied`, `ducking_reason`, `ducking_stats`, `DuckingConfig`, `duck_reference_audio`, and `NoiseReductionConfig.latency_ms` are introduced before use in later tasks.
- Scope control: adaptive FIR AEC, UI sliders for reduction strength, and release publishing are intentionally outside this plan.
