# Reference Echo Suppression Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in Both-mode reference-track echo suppression path that uses system loopback as the reference, cleans speaker leakage from the microphone track, source-levels the cleaned mic and loopback tracks, mixes them, then trims the final mixed output.

**Architecture:** Keep live capture unchanged: `RawRecorder` still writes complete mic and loopback temporary WAV files. Post-processing for Both mode becomes `raw sources -> echo suppression -> source leveling -> mix -> limiter -> final trim -> export`; AEC must run before source leveling and before any destructive trim so the original two-track timing remains available. Single-source recording keeps the existing normalize/export behavior, except trim should be applied to the final prepared WAV instead of mutating source temp WAVs before processing.

**Tech Stack:** Python, `numpy`, `soundfile`, existing `SourceActivityDetector` timeline helpers, PyQt6 settings UI, `unittest`.

---

## File Structure

- Create: `audio_processing.py`
  - Pure numpy DSP helpers with no GUI or recorder thread dependencies.
  - Owns reference delay/gain estimation, echo subtraction, active-source leveling, padding/alignment helpers, and limiter helpers.
- Create: `tests/test_audio_processing.py`
  - Unit tests for echo suppression, reference alignment, silent-source skips, and source leveling gates.
- Modify: `audio_recorder.py`
  - Imports `audio_processing`.
  - Orchestrates Both-mode processing order.
  - Adds echo suppression setting/state/metadata.
  - Moves trim from source temp WAV mutation to final prepared WAV trimming.
  - Uses activity timelines to build source-leveling masks.
- Modify: `tests/test_audio_output_profile.py`
  - Updates current Both-mode normalize expectations to source-leveling-after-AEC behavior.
- Modify: `tests/test_audio_trim.py`
  - Updates trim integration tests to prove trim is final-output post-processing, not pre-AEC source mutation.
- Modify: `gui.py`
  - Adds `Reduce speaker echo in Both mode` under `Post-Processing & Clipboard`.
  - Persists `echo_suppression`.
  - Passes it into `AudioRecorder`.
- Modify: `tests/test_gui_hotkeys.py`
  - Adds UI/settings persistence and recorder argument tests for `echo_suppression`.
- Modify: `README.md`
  - Documents echo suppression, source leveling, and the revised trim order.

---

## Processing Rules

These rules are implementation constraints, not preferences:

1. AEC means Acoustic Echo Cancellation-style reference suppression: loopback is the reference; microphone is the target to clean.
2. AEC must operate on raw mic and raw loopback temp WAVs before source leveling.
3. AEC must not require pre-trimmed sources. If trim is enabled, trim only the final prepared WAV after mix/limiter.
4. Source leveling is allowed after AEC and before mix.
5. Source leveling must skip silent or insufficiently active tracks instead of amplifying residual noise.
6. Mix output may be limited after summing, but must not run a second whole-mix loudness normalize.
7. The feature is only meaningful in `source_mode == "both"`. In `mic` or `loopback` modes, `echo_suppression=True` must be ignored safely.
8. The existing `normalize` setting should become source leveling for Both mode. Do not add a second user-facing volume-normalize checkbox in this plan.

---

### Task 1: Add Pure Audio Processing Tests

**Files:**
- Create: `tests/test_audio_processing.py`
- Create in Task 2: `audio_processing.py`

- [ ] **Step 1: Write failing tests for reference alignment, echo suppression, and source leveling**

Create `tests/test_audio_processing.py` with this content:

```python
import unittest

import numpy as np

from audio_processing import (
    EchoSuppressionConfig,
    SourceLevelingConfig,
    align_reference_to_target,
    apply_limiter,
    level_active_source,
    suppress_reference_echo,
)


SAMPLE_RATE = 16000


def delayed_copy(reference, delay_frames, target_frames):
    aligned = np.zeros((target_frames, 1), dtype=np.float32)
    if delay_frames >= 0:
        available = min(len(reference), target_frames - delay_frames)
        if available > 0:
            aligned[delay_frames:delay_frames + available] = reference[:available]
    else:
        offset = -delay_frames
        available = min(len(reference) - offset, target_frames)
        if available > 0:
            aligned[:available] = reference[offset:offset + available]
    return aligned


class ReferenceEchoSuppressionTests(unittest.TestCase):
    def test_align_reference_to_target_supports_positive_delay(self):
        reference = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)

        aligned = align_reference_to_target(reference, target_frames=6, delay_frames=2)

        self.assertEqual(
            aligned[:, 0].tolist(),
            [0.0, 0.0, 1.0, 2.0, 3.0, 0.0],
        )

    def test_align_reference_to_target_supports_negative_delay(self):
        reference = np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32)

        aligned = align_reference_to_target(reference, target_frames=5, delay_frames=-1)

        self.assertEqual(
            aligned[:, 0].tolist(),
            [2.0, 3.0, 4.0, 0.0, 0.0],
        )

    def test_suppresses_delayed_loopback_leak_from_mic(self):
        rng = np.random.default_rng(123)
        frames = SAMPLE_RATE * 4
        reference = rng.normal(0.0, 0.08, size=(frames, 1)).astype(np.float32)
        local_voice = np.zeros((frames, 1), dtype=np.float32)
        local_voice[SAMPLE_RATE:SAMPLE_RATE * 2] = 0.03
        delay_frames = int(0.05 * SAMPLE_RATE)
        leaked_reference = delayed_copy(reference, delay_frames, frames) * 0.45
        mic = local_voice + leaked_reference

        cleaned, stats = suppress_reference_echo(
            mic,
            reference,
            SAMPLE_RATE,
            EchoSuppressionConfig(
                enabled=True,
                min_delay_ms=-100,
                max_delay_ms=250,
                min_correlation=0.20,
                max_echo_gain=1.2,
                suppression_strength=1.0,
            ),
        )

        before_residual = float(np.sqrt(np.mean((mic - local_voice) ** 2)))
        after_residual = float(np.sqrt(np.mean((cleaned - local_voice) ** 2)))
        self.assertTrue(stats["applied"])
        self.assertAlmostEqual(stats["delay_ms"], 50.0, delta=2.0)
        self.assertAlmostEqual(stats["gain"], 0.45, delta=0.08)
        self.assertLess(after_residual, before_residual * 0.45)

    def test_skips_echo_suppression_when_reference_is_silent(self):
        mic = np.full((SAMPLE_RATE, 1), 0.05, dtype=np.float32)
        reference = np.zeros((SAMPLE_RATE, 1), dtype=np.float32)

        cleaned, stats = suppress_reference_echo(
            mic,
            reference,
            SAMPLE_RATE,
            EchoSuppressionConfig(enabled=True),
        )

        np.testing.assert_allclose(cleaned, mic)
        self.assertFalse(stats["applied"])
        self.assertEqual(stats["reason"], "silent_reference")

    def test_skips_echo_suppression_when_reference_does_not_match_mic(self):
        rng = np.random.default_rng(456)
        mic = rng.normal(0.0, 0.05, size=(SAMPLE_RATE * 2, 1)).astype(np.float32)
        reference = rng.normal(0.0, 0.05, size=(SAMPLE_RATE * 2, 1)).astype(np.float32)

        cleaned, stats = suppress_reference_echo(
            mic,
            reference,
            SAMPLE_RATE,
            EchoSuppressionConfig(enabled=True, min_correlation=0.80),
        )

        np.testing.assert_allclose(cleaned, mic)
        self.assertFalse(stats["applied"])
        self.assertEqual(stats["reason"], "low_correlation")


class SourceLevelingTests(unittest.TestCase):
    def test_level_active_source_skips_all_silent_audio(self):
        data = np.zeros((SAMPLE_RATE, 1), dtype=np.float32)
        mask = np.zeros(SAMPLE_RATE, dtype=bool)

        leveled, stats = level_active_source(
            data,
            mask,
            SourceLevelingConfig(enabled=True),
        )

        np.testing.assert_allclose(leveled, data)
        self.assertFalse(stats["applied"])
        self.assertEqual(stats["reason"], "insufficient_activity")

    def test_level_active_source_skips_noise_only_with_too_little_activity(self):
        data = np.full((SAMPLE_RATE, 1), 0.002, dtype=np.float32)
        mask = np.zeros(SAMPLE_RATE, dtype=bool)
        mask[: int(0.1 * SAMPLE_RATE)] = True

        leveled, stats = level_active_source(
            data,
            mask,
            SourceLevelingConfig(enabled=True, min_active_seconds=0.5),
        )

        np.testing.assert_allclose(leveled, data)
        self.assertFalse(stats["applied"])
        self.assertEqual(stats["reason"], "insufficient_activity")

    def test_level_active_source_raises_active_voice_with_gain_cap(self):
        data = np.zeros((SAMPLE_RATE * 2, 1), dtype=np.float32)
        data[SAMPLE_RATE:SAMPLE_RATE * 2] = 0.02
        mask = np.zeros(SAMPLE_RATE * 2, dtype=bool)
        mask[SAMPLE_RATE:SAMPLE_RATE * 2] = True

        leveled, stats = level_active_source(
            data,
            mask,
            SourceLevelingConfig(
                enabled=True,
                target_level=0.12,
                max_gain=4.0,
                active_floor=0.001,
                min_active_seconds=0.5,
            ),
        )

        self.assertTrue(stats["applied"])
        self.assertAlmostEqual(stats["gain"], 4.0, places=5)
        self.assertAlmostEqual(float(np.median(leveled[SAMPLE_RATE:, 0])), 0.08, places=5)
        self.assertEqual(float(np.max(np.abs(leveled[:SAMPLE_RATE, 0]))), 0.0)

    def test_apply_limiter_clips_to_limit(self):
        data = np.array([[-2.0], [-0.5], [0.5], [2.0]], dtype=np.float32)

        limited = apply_limiter(data, limit=0.98)

        self.assertEqual(limited[:, 0].tolist(), [-0.98, -0.5, 0.5, 0.98])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
python -m unittest tests.test_audio_processing
```

Expected: FAIL with `ModuleNotFoundError: No module named 'audio_processing'`.

- [ ] **Step 3: Commit the failing test**

```powershell
git add tests/test_audio_processing.py
git commit -m "Add echo suppression processing tests"
```

---

### Task 2: Implement Pure Echo Suppression And Source Leveling Helpers

**Files:**
- Create: `audio_processing.py`
- Test: `tests/test_audio_processing.py`

- [ ] **Step 1: Create `audio_processing.py`**

Create `audio_processing.py` with this content:

```python
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EchoSuppressionConfig:
    enabled: bool = False
    min_delay_ms: int = -100
    max_delay_ms: int = 500
    min_correlation: float = 0.20
    max_echo_gain: float = 1.5
    suppression_strength: float = 1.0
    analysis_floor: float = 0.001


@dataclass(frozen=True)
class SourceLevelingConfig:
    enabled: bool = False
    active_floor: float = 0.001
    target_level: float = 0.12
    max_gain: float = 8.0
    limit: float = 0.98
    reference_percentile: float = 95.0
    min_active_seconds: float = 0.5
    samplerate: int = 16000


def as_2d_float_audio(data):
    audio = np.asarray(data, dtype=np.float32)
    if audio.ndim == 1:
        audio = audio.reshape(-1, 1)
    return audio


def apply_limiter(data, limit=0.98):
    return np.clip(np.asarray(data, dtype=np.float32), -float(limit), float(limit))


def align_reference_to_target(reference, target_frames, delay_frames):
    reference = as_2d_float_audio(reference)
    target_frames = int(target_frames)
    delay_frames = int(delay_frames)
    aligned = np.zeros((target_frames, reference.shape[1]), dtype=np.float32)

    if target_frames <= 0 or len(reference) == 0:
        return aligned

    if delay_frames >= 0:
        source_start = 0
        target_start = delay_frames
    else:
        source_start = -delay_frames
        target_start = 0

    if source_start >= len(reference) or target_start >= target_frames:
        return aligned

    frames = min(len(reference) - source_start, target_frames - target_start)
    if frames > 0:
        aligned[target_start:target_start + frames] = reference[source_start:source_start + frames]
    return aligned


def _mono_for_analysis(data):
    audio = as_2d_float_audio(data)
    return np.mean(audio, axis=1).astype(np.float32, copy=False)


def _preemphasize(data):
    if len(data) == 0:
        return data
    return np.concatenate(([data[0]], data[1:] - 0.95 * data[:-1])).astype(np.float32)


def _normalized_corr(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if len(a) == 0 or len(b) == 0:
        return 0.0
    a = a - float(np.mean(a))
    b = b - float(np.mean(b))
    denom = float(np.sqrt(np.dot(a, a) * np.dot(b, b)))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def _iter_delay_frames(config, samplerate):
    min_delay = int(round(config.min_delay_ms * samplerate / 1000.0))
    max_delay = int(round(config.max_delay_ms * samplerate / 1000.0))
    if min_delay > max_delay:
        min_delay, max_delay = max_delay, min_delay
    return range(min_delay, max_delay + 1)


def estimate_reference_echo(reference, mic, samplerate, config):
    reference = as_2d_float_audio(reference)
    mic = as_2d_float_audio(mic)
    target_frames = min(len(reference), len(mic))
    if target_frames <= 0:
        return {"applied": False, "reason": "empty_audio"}

    ref_mono = _mono_for_analysis(reference[:target_frames])
    mic_mono = _mono_for_analysis(mic[:target_frames])
    if float(np.percentile(np.abs(ref_mono), 95)) < config.analysis_floor:
        return {"applied": False, "reason": "silent_reference"}

    ref_analysis = _preemphasize(ref_mono)
    mic_analysis = _preemphasize(mic_mono)

    best_delay = 0
    best_corr = -1.0
    for delay in _iter_delay_frames(config, samplerate):
        aligned = align_reference_to_target(ref_analysis.reshape(-1, 1), target_frames, delay)[:, 0]
        corr = _normalized_corr(aligned, mic_analysis)
        if corr > best_corr:
            best_corr = corr
            best_delay = delay

    if best_corr < config.min_correlation:
        return {
            "applied": False,
            "reason": "low_correlation",
            "correlation": round(float(best_corr), 6),
            "delay_frames": int(best_delay),
            "delay_ms": round(1000.0 * best_delay / float(samplerate), 3),
        }

    aligned_reference = align_reference_to_target(reference, len(mic), best_delay)
    ref_energy = np.sum(aligned_reference * aligned_reference, axis=0)
    gains = []
    for channel in range(mic.shape[1]):
        ref_channel = aligned_reference[:, min(channel, aligned_reference.shape[1] - 1)]
        mic_channel = mic[:, channel]
        denom = float(np.dot(ref_channel, ref_channel))
        gain = 0.0 if denom <= 1e-12 else float(np.dot(mic_channel, ref_channel) / denom)
        gains.append(float(np.clip(gain, 0.0, config.max_echo_gain)))

    gain = float(np.median(gains))
    if gain <= 0.0 or not np.isfinite(gain):
        return {
            "applied": False,
            "reason": "non_positive_gain",
            "correlation": round(float(best_corr), 6),
            "delay_frames": int(best_delay),
            "delay_ms": round(1000.0 * best_delay / float(samplerate), 3),
        }

    return {
        "applied": True,
        "reason": "applied",
        "delay_frames": int(best_delay),
        "delay_ms": round(1000.0 * best_delay / float(samplerate), 3),
        "correlation": round(float(best_corr), 6),
        "gain": round(gain, 6),
        "reference_energy": round(float(np.sum(ref_energy)), 6),
    }


def suppress_reference_echo(mic, reference, samplerate, config):
    mic = as_2d_float_audio(mic)
    reference = as_2d_float_audio(reference)
    if not config.enabled:
        return mic.copy(), {"applied": False, "reason": "disabled"}

    stats = estimate_reference_echo(reference, mic, samplerate, config)
    if not stats.get("applied"):
        return mic.copy(), stats

    aligned_reference = align_reference_to_target(reference, len(mic), stats["delay_frames"])
    if aligned_reference.shape[1] != mic.shape[1]:
        if aligned_reference.shape[1] == 1:
            aligned_reference = np.repeat(aligned_reference, mic.shape[1], axis=1)
        else:
            aligned_reference = aligned_reference[:, :mic.shape[1]]

    cleaned = mic - aligned_reference * float(stats["gain"]) * float(config.suppression_strength)
    return apply_limiter(cleaned), stats


def level_active_source(data, active_mask, config):
    data = as_2d_float_audio(data)
    if not config.enabled:
        return data.copy(), {"applied": False, "reason": "disabled", "gain": 1.0}

    active_mask = np.asarray(active_mask, dtype=bool)
    if len(active_mask) != len(data):
        raise ValueError("active_mask length must match audio length")

    active_seconds = float(np.count_nonzero(active_mask)) / float(config.samplerate)
    if active_seconds < config.min_active_seconds:
        return data.copy(), {
            "applied": False,
            "reason": "insufficient_activity",
            "gain": 1.0,
            "active_seconds": round(active_seconds, 3),
        }

    active = np.abs(data[active_mask])
    active = active[active >= config.active_floor]
    if active.size == 0:
        return data.copy(), {
            "applied": False,
            "reason": "below_active_floor",
            "gain": 1.0,
            "active_seconds": round(active_seconds, 3),
        }

    rms = float(np.sqrt(np.mean(active ** 2)))
    percentile = float(np.percentile(active, config.reference_percentile))
    reference_level = max(rms, percentile)
    if not np.isfinite(reference_level) or reference_level <= 0.0:
        return data.copy(), {
            "applied": False,
            "reason": "invalid_reference_level",
            "gain": 1.0,
            "active_seconds": round(active_seconds, 3),
        }

    gain = min(config.target_level / reference_level, config.max_gain)
    return apply_limiter(data * gain, config.limit), {
        "applied": True,
        "reason": "applied",
        "gain": round(float(gain), 6),
        "active_seconds": round(active_seconds, 3),
        "reference_level": round(reference_level, 6),
    }
```

- [ ] **Step 2: Run focused tests**

Run:

```powershell
python -m unittest tests.test_audio_processing
```

Expected: PASS.

- [ ] **Step 3: Commit helper implementation**

```powershell
git add audio_processing.py tests/test_audio_processing.py
git commit -m "Add reference echo suppression helpers"
```

---

### Task 3: Add Echo Suppression Wiring To AudioRecorder

**Files:**
- Modify: `audio_recorder.py`
- Modify: `tests/test_audio_output_profile.py`
- Test: `tests/test_audio_processing.py`

- [ ] **Step 1: Write failing recorder integration tests**

Append these tests to `tests/test_audio_output_profile.py` before `_make_recorder()`:

```python
    def test_prepare_source_wav_suppresses_echo_before_source_leveling(self):
        import os
        import tempfile

        import numpy as np
        import soundfile as sf

        with tempfile.TemporaryDirectory() as temp_dir:
            samplerate = 16000
            frames = samplerate * 4
            rng = np.random.default_rng(321)
            loopback = rng.normal(0.0, 0.05, size=(frames, 1)).astype(np.float32)
            local_voice = np.zeros((frames, 1), dtype=np.float32)
            local_voice[samplerate:samplerate * 2] = 0.025
            delay = int(0.05 * samplerate)
            leak = np.zeros_like(loopback)
            leak[delay:] = loopback[:-delay] * 0.5
            mic = local_voice + leak

            mic_file = os.path.join(temp_dir, "mic.wav")
            loop_file = os.path.join(temp_dir, "loop.wav")
            sf.write(mic_file, mic, samplerate, format="WAV", subtype="FLOAT")
            sf.write(loop_file, loopback, samplerate, format="WAV", subtype="FLOAT")

            recorder = self._make_recorder("wav", "balanced", stereo=False)
            recorder.source_mode = "both"
            recorder.normalize = True
            recorder.echo_suppression = True
            recorder.temp_files = [mic_file, loop_file]

            mixed_file = recorder._prepare_source_wav("FLOAT")
            mixed, _ = sf.read(mixed_file, always_2d=True)

            self.assertTrue(recorder.echo_suppression_applied)
            self.assertGreater(recorder.echo_suppression_correlation, 0.20)
            self.assertLess(np.max(np.abs(mixed)), 0.981)

    def test_prepare_source_wav_ignores_echo_suppression_for_single_source(self):
        import os
        import tempfile

        import numpy as np
        import soundfile as sf

        with tempfile.TemporaryDirectory() as temp_dir:
            samplerate = 16000
            mic_file = os.path.join(temp_dir, "mic.wav")
            data = np.full((samplerate, 1), 0.02, dtype=np.float32)
            sf.write(mic_file, data, samplerate, format="WAV", subtype="FLOAT")

            recorder = self._make_recorder("wav", "balanced", stereo=False)
            recorder.source_mode = "mic"
            recorder.normalize = False
            recorder.echo_suppression = True
            recorder.temp_files = [mic_file]

            prepared = recorder._prepare_source_wav("FLOAT")

            self.assertEqual(prepared, mic_file)
            self.assertFalse(recorder.echo_suppression_applied)
            self.assertEqual(recorder.echo_suppression_reason, "single_source")
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
python -m unittest tests.test_audio_output_profile.OutputProfileTests.test_prepare_source_wav_suppresses_echo_before_source_leveling tests.test_audio_output_profile.OutputProfileTests.test_prepare_source_wav_ignores_echo_suppression_for_single_source
```

Expected: FAIL with `AttributeError: 'AudioRecorder' object has no attribute 'echo_suppression'` or missing metadata fields.

- [ ] **Step 3: Add imports and constants to `audio_recorder.py`**

Add these imports after `from app_metadata import RECORDING_FILENAME_PREFIX`:

```python
from audio_processing import (
    EchoSuppressionConfig,
    SourceLevelingConfig,
    apply_limiter,
    level_active_source,
    suppress_reference_echo,
)
```

Add these constants after the existing normalize constants:

```python
ECHO_SUPPRESSION_DEFAULT_ENABLED = False
ECHO_SUPPRESSION_MIN_DELAY_MS = -100
ECHO_SUPPRESSION_MAX_DELAY_MS = 500
ECHO_SUPPRESSION_MIN_CORRELATION = 0.20
ECHO_SUPPRESSION_MAX_GAIN = 1.5
ECHO_SUPPRESSION_STRENGTH = 1.0
SOURCE_LEVELING_MIN_ACTIVE_SECONDS = 0.5
```

- [ ] **Step 4: Add bool normalization helper**

Add this helper after `normalize_trim_silence_enabled()`:

```python
def normalize_echo_suppression_enabled(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return ECHO_SUPPRESSION_DEFAULT_ENABLED
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("true", "1", "yes", "on"):
            return True
        if normalized in ("false", "0", "no", "off", ""):
            return False
    return bool(value)
```

- [ ] **Step 5: Extend `AudioRecorder.__init__()`**

Change the signature:

```python
        normalize=False,
        echo_suppression=ECHO_SUPPRESSION_DEFAULT_ENABLED,
        trim_silence=TRIM_SILENCE_DEFAULT_ENABLED,
```

After `self.normalize = normalize`, add:

```python
        self.echo_suppression = normalize_echo_suppression_enabled(echo_suppression)
```

After trim metadata fields, add:

```python
        self.echo_suppression_applied = False
        self.echo_suppression_reason = "not_run"
        self.echo_suppression_delay_ms = 0.0
        self.echo_suppression_gain = 0.0
        self.echo_suppression_correlation = 0.0
        self.source_leveling_stats = {}
```

- [ ] **Step 6: Add echo suppression metadata**

In `build_finish_metadata()`, add:

```python
            "echo_suppression_enabled": self.echo_suppression,
            "echo_suppression_applied": self.echo_suppression_applied,
            "echo_suppression_reason": self.echo_suppression_reason,
            "echo_suppression_delay_ms": round(self.echo_suppression_delay_ms, 3),
            "echo_suppression_gain": round(self.echo_suppression_gain, 6),
            "echo_suppression_correlation": round(self.echo_suppression_correlation, 6),
            "source_leveling": self.source_leveling_stats,
```

- [ ] **Step 7: Add activity mask helper**

Add this method before `_prepare_source_wav()`:

```python
    def _build_activity_mask(self, data, samplerate, source_name):
        audio = _as_2d_audio(data)
        timeline = build_source_activity_timeline(audio, samplerate, source_name)
        mask = np.zeros(len(audio), dtype=bool)
        for start_frame, end_frame, active in timeline:
            if active:
                mask[start_frame:end_frame] = True
        return mask
```

- [ ] **Step 8: Add source leveling helper**

Add this method after `_build_activity_mask()`:

```python
    def _level_source_data(self, data, samplerate, source_name):
        if not self.normalize:
            return _as_2d_audio(data).astype(np.float32, copy=False), {
                "applied": False,
                "reason": "disabled",
                "gain": 1.0,
            }
        mask = self._build_activity_mask(data, samplerate, source_name)
        leveled, stats = level_active_source(
            data,
            mask,
            SourceLevelingConfig(
                enabled=True,
                active_floor=NORMALIZE_ACTIVE_FLOOR,
                target_level=NORMALIZE_TARGET_LEVEL,
                max_gain=NORMALIZE_MAX_GAIN,
                limit=NORMALIZE_LIMIT,
                reference_percentile=NORMALIZE_REFERENCE_PERCENTILE,
                min_active_seconds=SOURCE_LEVELING_MIN_ACTIVE_SECONDS,
                samplerate=samplerate,
            ),
        )
        self.source_leveling_stats[source_name] = stats
        return leveled, stats
```

- [ ] **Step 9: Add Both-mode processing helper**

Add this method after `_level_source_data()`:

```python
    def _prepare_both_source_wav(self, subtype):
        mic_path, loopback_path = self.temp_files[:2]
        mic_data, mic_sr = sf.read(mic_path, always_2d=True)
        loopback_data, loopback_sr = sf.read(loopback_path, always_2d=True)
        if mic_sr != loopback_sr:
            raise ValueError("Cannot process both sources with different sample rates.")
        if mic_data.shape[1] != loopback_data.shape[1]:
            raise ValueError("Cannot process both sources with different channel counts.")

        self.source_leveling_stats = {}
        if self.echo_suppression:
            mic_data, echo_stats = suppress_reference_echo(
                mic_data,
                loopback_data,
                mic_sr,
                EchoSuppressionConfig(
                    enabled=True,
                    min_delay_ms=ECHO_SUPPRESSION_MIN_DELAY_MS,
                    max_delay_ms=ECHO_SUPPRESSION_MAX_DELAY_MS,
                    min_correlation=ECHO_SUPPRESSION_MIN_CORRELATION,
                    max_echo_gain=ECHO_SUPPRESSION_MAX_GAIN,
                    suppression_strength=ECHO_SUPPRESSION_STRENGTH,
                    analysis_floor=NORMALIZE_ACTIVE_FLOOR,
                ),
            )
            self.echo_suppression_applied = bool(echo_stats.get("applied"))
            self.echo_suppression_reason = str(echo_stats.get("reason", "unknown"))
            self.echo_suppression_delay_ms = float(echo_stats.get("delay_ms", 0.0))
            self.echo_suppression_gain = float(echo_stats.get("gain", 0.0))
            self.echo_suppression_correlation = float(echo_stats.get("correlation", 0.0))
        else:
            self.echo_suppression_applied = False
            self.echo_suppression_reason = "disabled"
            self.echo_suppression_delay_ms = 0.0
            self.echo_suppression_gain = 0.0
            self.echo_suppression_correlation = 0.0

        mic_data, _mic_level_stats = self._level_source_data(mic_data, mic_sr, "mic")
        loopback_data, _loopback_level_stats = self._level_source_data(loopback_data, loopback_sr, "loopback")

        mixed_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        self._mix_audio_data(mic_data, loopback_data, mixed_wav, mic_sr, subtype, limit_output=True)
        self.temp_files.append(mixed_wav)
        return mixed_wav
```

- [ ] **Step 10: Add data-based mix helper**

Add this method before `_mix_audio()`:

```python
    def _mix_audio_data(self, d1, d2, out_file, samplerate, subtype, limit_output=False):
        d1 = _as_2d_audio(d1).astype(np.float32, copy=False)
        d2 = _as_2d_audio(d2).astype(np.float32, copy=False)
        if d1.shape[1] != d2.shape[1]:
            raise ValueError("Cannot mix audio with different channel counts.")

        max_len = max(len(d1), len(d2))
        if len(d1) < max_len:
            d1 = np.concatenate((d1, np.zeros((max_len - len(d1), d1.shape[1]), dtype=d1.dtype)))
        if len(d2) < max_len:
            d2 = np.concatenate((d2, np.zeros((max_len - len(d2), d2.shape[1]), dtype=d2.dtype)))

        mixed = d1 + d2
        mixed = apply_limiter(mixed, NORMALIZE_LIMIT if limit_output else 1.0)
        sf.write(out_file, mixed, samplerate, format="WAV", subtype=subtype)
```

- [ ] **Step 11: Update file-based `_mix_audio()` to reuse `_mix_audio_data()`**

Replace the body after sample-rate/channel checks and file reads with:

```python
        self._mix_audio_data(d1, d2, out_file, sr1, subtype, limit_output=limit_output)
```

- [ ] **Step 12: Update `_prepare_source_wav()` ordering**

Replace `_prepare_source_wav()` with:

```python
    def _prepare_source_wav(self, subtype):
        if len(self.temp_files) == 2:
            source_wav = self._prepare_both_source_wav(subtype)
            return self._maybe_trim_final_wav(source_wav)

        self.echo_suppression_applied = False
        self.echo_suppression_reason = "single_source"
        self.echo_suppression_delay_ms = 0.0
        self.echo_suppression_gain = 0.0
        self.echo_suppression_correlation = 0.0
        self.source_leveling_stats = {}

        source_wav = self.temp_files[0]
        if self.normalize:
            self._normalize_audio(source_wav)
        return self._maybe_trim_final_wav(source_wav)
```

This intentionally removes `_maybe_trim_temp_sources()` from the start of `_prepare_source_wav()`.

- [ ] **Step 13: Run focused tests**

Run:

```powershell
python -m unittest tests.test_audio_processing tests.test_audio_output_profile
```

Expected: echo suppression tests pass. Existing trim tests may fail until Task 4 changes trim order.

- [ ] **Step 14: Commit recorder wiring**

```powershell
git add audio_recorder.py tests/test_audio_output_profile.py audio_processing.py tests/test_audio_processing.py
git commit -m "Wire reference echo suppression into recorder"
```

---

### Task 4: Move Trim To Final Prepared WAV

**Files:**
- Modify: `audio_recorder.py`
- Modify: `tests/test_audio_trim.py`

- [ ] **Step 1: Update trim integration tests for final-output trim**

In `tests/test_audio_trim.py`, replace `test_prepare_source_wav_applies_shared_bounds_before_both_mix` with:

```python
    def test_prepare_source_wav_trims_after_both_mix(self):
        import os
        import tempfile

        import numpy as np
        import soundfile as sf

        with tempfile.TemporaryDirectory() as temp_dir:
            mic_file = os.path.join(temp_dir, "mic.wav")
            loop_file = os.path.join(temp_dir, "loop.wav")
            mic = np.zeros((SAMPLE_RATE * 10, 1), dtype=np.float32)
            loopback = np.zeros((SAMPLE_RATE * 10, 1), dtype=np.float32)
            mic[SAMPLE_RATE * 6:SAMPLE_RATE * 7] = 0.4
            sf.write(mic_file, mic, SAMPLE_RATE, format="WAV", subtype="FLOAT")
            sf.write(loop_file, loopback, SAMPLE_RATE, format="WAV", subtype="FLOAT")

            recorder = AudioRecorder(
                mic_id="mic",
                source_mode="both",
                output_folder=temp_dir,
                output_format="wav",
                quality="balanced",
                stereo=False,
                normalize=False,
                trim_silence=True,
            )
            recorder.temp_files = [mic_file, loop_file]

            mixed_wav = recorder._prepare_source_wav("FLOAT")
            mixed, sr = sf.read(mixed_wav, always_2d=True)

            self.assertEqual(sr, SAMPLE_RATE)
            self.assertEqual(len(mixed), INTEGRATION_TRIMMED_FRAMES)
            self.assertTrue(recorder.trim_silence_applied)
            self.assertAlmostEqual(
                recorder.trim_silence_removed_seconds,
                3.7,
                places=1,
            )
```

In the same file, update any test name or assertion that says trim happens "before both mix" to "after final mix".

- [ ] **Step 2: Run tests to verify failure**

Run:

```powershell
python -m unittest tests.test_audio_trim
```

Expected: FAIL because `AudioRecorder` still uses `_maybe_trim_temp_sources()` or lacks `_maybe_trim_final_wav()`.

- [ ] **Step 3: Add final WAV trim helper**

Add this method before `_prepare_source_wav()`:

```python
    def _maybe_trim_final_wav(self, source_wav):
        if not self.trim_silence:
            self.trim_silence_applied = False
            self.trim_silence_removed_seconds = 0.0
            return source_wav

        try:
            info = sf.info(source_wav)
            data, samplerate = sf.read(source_wav, always_2d=True)
            trim_source_name = "loopback" if self.source_mode == "loopback" else "mic"
            trimmed, stats = trim_edge_silence_data(
                data,
                samplerate,
                source_name=trim_source_name,
            )
            if stats["applied"]:
                trimmed_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
                sf.write(trimmed_wav, trimmed, samplerate, format=info.format, subtype=info.subtype)
                self.temp_files.append(trimmed_wav)
                self.trim_silence_applied = True
                self.trim_silence_removed_seconds = (
                    float(stats["start_removed_seconds"]) + float(stats["end_removed_seconds"])
                )
                return trimmed_wav

            self.trim_silence_applied = False
            self.trim_silence_removed_seconds = 0.0
            return source_wav
        except Exception as e:
            print(f"Final silence trim failed: {e}")
            self.trim_silence_applied = False
            self.trim_silence_removed_seconds = 0.0
            return source_wav
```

- [ ] **Step 4: Stop calling `_maybe_trim_temp_sources()`**

Do not delete `_maybe_trim_temp_sources()` in this task. Leave it unused until tests are green; a later cleanup can remove it if no code references it.

Confirm `_prepare_source_wav()` contains no call to:

```python
self._maybe_trim_temp_sources()
```

- [ ] **Step 5: Run trim tests**

Run:

```powershell
python -m unittest tests.test_audio_trim
```

Expected: PASS.

- [ ] **Step 6: Run output profile tests**

Run:

```powershell
python -m unittest tests.test_audio_output_profile
```

Expected: PASS.

- [ ] **Step 7: Commit trim order change**

```powershell
git add audio_recorder.py tests/test_audio_trim.py tests/test_audio_output_profile.py
git commit -m "Move silence trim after final processing"
```

---

### Task 5: Add Echo Suppression Setting To GUI

**Files:**
- Modify: `gui.py`
- Modify: `tests/test_gui_hotkeys.py`
- Modify: `audio_recorder.py`

- [ ] **Step 1: Add failing GUI tests**

Append these tests to `tests/test_gui_hotkeys.py` near the existing trim/normalize settings tests:

```python
    def test_echo_suppression_setting_defaults_off(self):
        window = self.make_window({})

        settings = window.get_settings()

        self.assertIs(settings["echo_suppression"], False)

    def test_echo_suppression_setting_can_be_enabled(self):
        window = self.make_window({"echo_suppression": True})

        settings = window.get_settings()

        self.assertIs(settings["echo_suppression"], True)

    def test_echo_suppression_setting_is_in_post_processing_group(self):
        window = self.make_window({})

        post_group = window.findChild(QGroupBox, "postProcessingSettingsGroup")

        self.assertIs(window.chk_echo_suppression.parentWidget(), post_group)
```

In `test_start_recording_passes_trim_silence_setting`, extend the mocked settings payload with:

```python
                    "echo_suppression": True,
```

Add this assertion:

```python
        self.assertEqual(AudioRecorder.call_args.kwargs["echo_suppression"], True)
```

- [ ] **Step 2: Run GUI tests to verify failure**

Run:

```powershell
python -m unittest tests.test_gui_hotkeys
```

Expected: FAIL with `AttributeError: 'SettingsWindow' object has no attribute 'chk_echo_suppression'`.

- [ ] **Step 3: Add checkbox in `SettingsWindow.init_ui()`**

After:

```python
        self.chk_normalize = QCheckBox("Normalize Audio (Apply first)")
```

add:

```python
        self.chk_echo_suppression = QCheckBox("Reduce speaker echo in Both mode")
        self.chk_echo_suppression.setToolTip(
            "Uses system audio as a reference to reduce speaker leakage from the microphone track before source leveling and mixing."
        )
```

After:

```python
        layout_post.addWidget(self.chk_normalize)
```

add:

```python
        layout_post.addWidget(self.chk_echo_suppression)
```

- [ ] **Step 4: Load and save setting**

In `load_settings()`, after normalize loading, add:

```python
        self.chk_echo_suppression.setChecked(
            self._parse_bool_setting(data.get("echo_suppression"))
        )
```

In `get_settings()`, after `"normalize": self.chk_normalize.isChecked(),`, add:

```python
            "echo_suppression": self.chk_echo_suppression.isChecked(),
```

- [ ] **Step 5: Pass setting into `AudioRecorder`**

In `TrayApplication.start_recording()`, after:

```python
            normalize=settings['normalize'],
```

add:

```python
            echo_suppression=settings.get("echo_suppression", False),
```

- [ ] **Step 6: Run GUI tests**

Run:

```powershell
python -m unittest tests.test_gui_hotkeys
```

Expected: PASS.

- [ ] **Step 7: Commit GUI setting**

```powershell
git add gui.py tests/test_gui_hotkeys.py audio_recorder.py
git commit -m "Add echo suppression setting"
```

---

### Task 6: Update User-Facing Text And Normalize Semantics

**Files:**
- Modify: `README.md`
- Modify: `gui.py`
- Modify: `tests/test_product_identity_text.py` only if product text scan expects README wording

- [ ] **Step 1: Update checkbox label text**

In `gui.py`, change:

```python
        self.chk_normalize = QCheckBox("Normalize Audio (Apply first)")
```

to:

```python
        self.chk_normalize = QCheckBox("Source Leveling / Normalize")
        self.chk_normalize.setToolTip(
            "Balances active source levels. In Both mode this runs after echo suppression and before mixing."
        )
```

- [ ] **Step 2: Update README feature bullets**

Replace the current post-processing bullets:

```markdown
- **Normalize Audio (Apply first)**: raises the main voice/body of each source before mixing and limits sharp peaks.
- **Silence trim**: optional post-processing that trims only the start and end silence beyond 5 seconds, using the same mic/loopback activity detection rules as silence auto-stop.
```

with:

```markdown
- **Reduce speaker echo in Both mode**: optionally uses system audio as a reference to reduce speaker leakage from the microphone track before mixing.
- **Source Leveling / Normalize**: raises active source segments with a maximum gain cap. In Both mode it runs after echo suppression and before mixing, so quiet microphone speech is not buried by louder system audio.
- **Silence trim**: optional final post-processing that trims only the start and end silence beyond 5 seconds, using the same activity detection rules as silence auto-stop.
```

Replace usage step 8:

```markdown
8. In **Post-Processing & Clipboard**, enable normalization, edge-silence trim, clipboard copy, or delete-after-copy as needed.
```

with:

```markdown
8. In **Post-Processing & Clipboard**, enable echo reduction, source leveling, final edge-silence trim, clipboard copy, or delete-after-copy as needed.
```

- [ ] **Step 3: Run text and GUI tests**

Run:

```powershell
python -m unittest tests.test_product_identity_text tests.test_gui_hotkeys
```

Expected: PASS.

- [ ] **Step 4: Commit docs and labels**

```powershell
git add README.md gui.py tests/test_product_identity_text.py
git commit -m "Document echo suppression processing order"
```

---

### Task 7: Full Regression And Manual Artifact Check

**Files:**
- No code changes unless tests reveal a defect.

- [ ] **Step 1: Run full unit test suite**

Run:

```powershell
python -m unittest discover -s tests
```

Expected: all tests pass.

- [ ] **Step 2: Run fixed read-only analysis on the archived bad sample**

Run this command from the repository root. It reads the bad archived sample and prints short-delay correlation peaks without modifying the file:

```powershell
$code = @'
import math
import numpy as np
import soundfile as sf

path = r"E:\Archived\2026-05\PC\MeetRec_20260521_200500.flac"
data, sr = sf.read(path, always_2d=True)
x = data[:, 0].astype(np.float64)
win = int(2.5 * sr)
hop = int(0.75 * sr)
lags_ms = [25, 30, 35, 40, 50, 60, 70, 80, 100, 120, 150, 200, 300]

def db(value):
    return 20.0 * math.log10(max(float(value), 1e-12))

def corr_at_lag(seg, lag):
    a = seg[lag:]
    b = seg[:-lag]
    denom = math.sqrt(float(np.dot(a, a)) * float(np.dot(b, b)))
    return 0.0 if denom <= 1e-12 else float(np.dot(a, b) / denom)

scores = []
for start in range(0, len(x) - win + 1, hop):
    seg = x[start:start + win]
    rms = float(np.sqrt(np.mean(seg * seg)))
    if db(rms) < -40.0:
        continue
    seg = seg - np.mean(seg)
    seg = np.concatenate([[seg[0]], seg[1:] - 0.95 * seg[:-1]])
    best = max(corr_at_lag(seg, int(ms * sr / 1000.0)) for ms in lags_ms)
    scores.append(best)

print("windows", len(scores))
print("median", round(float(np.median(scores)), 3))
print("p90", round(float(np.percentile(scores, 90)), 3))
print("max", round(float(np.max(scores)), 3))
'@
$code | python -
```

Expected: output shows the sample still has strong short-delay correlation, with `p90` above `0.30` and `max` above `0.50`. Do not modify the archived file during this step.

- [ ] **Step 3: Build package**

Run:

```powershell
pyinstaller --noconfirm MeetRec.spec
```

Expected: `dist\MeetRec\MeetRec.exe` exists and the build exits with code 0.

- [ ] **Step 4: Smoke-test processing manually**

Manual path:

1. Start `dist\MeetRec\MeetRec.exe`.
2. Open Settings.
3. Enable `Reduce speaker echo in Both mode`.
4. Enable `Source Leveling / Normalize`.
5. Enable `Trim start/end silence over 5s`.
6. Record a short Both-mode sample with speaker playback audible to the microphone.
7. Confirm a FLAC/WAV file is saved.

Expected:

- Recording succeeds.
- Saved file is playable.
- Mic speech is not obviously buried by loopback.
- Speaker echo is reduced compared with Both mode with echo suppression off.
- Start/end trim only affects final output, not AEC alignment.

- [ ] **Step 5: Commit any regression fix**

If Step 1 through Step 4 required a fix, commit it with:

```powershell
git add audio_recorder.py audio_processing.py gui.py README.md tests
git commit -m "Fix echo suppression regression"
```

If no fix was needed, do not create an empty commit.

---

## Self-Review

- Spec coverage: This plan covers the agreed processing order: raw mic/loopback -> AEC -> source leveling -> mix -> limiter -> final trim -> export.
- Spec coverage: It explicitly changes old trim behavior because pre-AEC source trimming can damage reference alignment.
- Spec coverage: It handles silent mic, silent loopback, insufficient activity, low correlation, single-source mode, and disabled feature states.
- Placeholder scan: No task asks the implementer to invent unspecified error handling, tests, function names, or paths.
- Type consistency: `EchoSuppressionConfig`, `SourceLevelingConfig`, `suppress_reference_echo()`, `level_active_source()`, `echo_suppression`, and metadata field names are used consistently across tasks.
- Risk control: Default setting is off, AEC skips on silent/low-correlation references, source leveling skips insufficient activity, and mix output only applies a limiter.
