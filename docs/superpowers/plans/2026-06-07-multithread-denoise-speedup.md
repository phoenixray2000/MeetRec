# Multithreaded RNNoise Denoise Speedup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut MeetRec's post-recording denoise from ~12 min to ~35-40 s for a 2-hour recording (~20x), by combining an AVX2-compiled rnnoise.dll with multithreaded, segment-parallel processing that releases the GIL.

**Architecture:** The AVX2 dll (already built and installed) provides ~2.9x. On top of that, `denoise.reduce_noise` splits each mono 48k stream into K contiguous segments (K ≈ physical cores), each with a 1 s warmup-overlap prefix to let RNNoise's stateful RNN converge before the kept region. Each segment is denoised in its own thread via a single C call to `rnnoise_process_buffer`; because ctypes releases the GIL for the duration of that call, the C compute runs truly in parallel. Results are concatenated, warmup prefixes discarded. Any failure falls back to single-threaded whole-buffer processing, and a backend without `rnnoise_process_buffer` falls back to the existing per-frame `process`.

**Tech Stack:** Python 3, numpy, ctypes, `concurrent.futures.ThreadPoolExecutor`, RNNoise (xiph/rnnoise) built with MSVC `/arch:AVX2`.

---

## Prerequisite State (already done in the originating session — verify, don't redo)

- `rnnoise_avx2.dll` was built and **already installed as the root `rnnoise.dll`**; the previous scalar dll is backed up at `rnnoise.dll.scalar.bak`. Verify: a fresh `python -c "import denoise; print(denoise.is_available())"` prints `True`, and `denoise._BACKEND._lib.rnnoise_process_buffer` exists.
- The custom batch function lives at `.deps/rnnoise/src/rnnoise_buffer.c` and the build scripts at `.deps/rnnoise/build_msvc/` — but `.deps/` is gitignored. Task 1 persists copies into a tracked `build/rnnoise/` directory.
- rnnoise source: `https://github.com/xiph/rnnoise.git` @ commit `70f1d256acd4b34a572f999a05c87bf00b67730d`.

Measured facts this plan relies on (from spikes): per-frame vs `process_buffer` output is bitwise identical; AVX2 vs scalar corr=0.9997; 1 s warmup-overlap gives corr=0.9999 vs whole-buffer with boundary diffs <2% peak; 8 threads gave 7.25x (35 s/2h) on this 8-physical-core machine.

---

## File Structure

- **Modify `denoise.py`** — add `concurrent.futures` import, parallel/warmup/threshold constants, bind `rnnoise_process_buffer` in `_RNNoiseBackend.__init__`, add `_RNNoiseBackend._process_buffer_segment`, add module-level `_parallel_denoise`, add `_RNNoiseBackend.process_parallel`, add `_denoise_mono` helper, switch `reduce_noise` to call it. One file, one responsibility (RNNoise access + denoise orchestration) — keep it here.
- **Modify `tests/test_denoise.py`** — add tests for `_parallel_denoise` orchestration (mocked segment fn), `process_parallel` fallback, and a parallel-path wiring test; keep existing `_FakeBackend` tests passing.
- **Create `build/rnnoise/build_rnnoise_dll.bat`** — tracked AVX2 build script.
- **Create `build/rnnoise/rnnoise_buffer.c`** — tracked copy of the batch wrapper.
- **Create `build/rnnoise/README.md`** — source provenance + reproducible build steps.

---

## Task 1: Persist the rnnoise build artifacts into version control

**Files:**
- Create: `build/rnnoise/rnnoise_buffer.c`
- Create: `build/rnnoise/build_rnnoise_dll.bat`
- Create: `build/rnnoise/README.md`

- [ ] **Step 1: Create the batch-wrapper source**

Create `build/rnnoise/rnnoise_buffer.c`:

```c
/* Batch wrapper around rnnoise_process_frame so the per-frame loop runs in C.
   The single long C call releases the Python GIL for its whole duration, which
   is what lets denoise.py run segments in parallel threads. Kept in a separate
   file so the upstream rnnoise sources stay untouched. */
#include <stddef.h>
#include "rnnoise.h"

RNNOISE_EXPORT void rnnoise_process_buffer(DenoiseState *st, int nframes,
                                           float *out, const float *in) {
    int fs = rnnoise_get_frame_size();
    int i;
    for (i = 0; i < nframes; i++) {
        size_t off = (size_t)i * (size_t)fs;
        rnnoise_process_frame(st, out + off, in + off);
    }
}
```

- [ ] **Step 2: Create the build script**

Create `build/rnnoise/build_rnnoise_dll.bat`. The `vcvars64` path is hard-coded because vswhere is broken on the build machine. `/arch:AVX2` makes MSVC define `__AVX__`, routing rnnoise's `vec.h` to the SIMD `vec_avx.h` path. `/MT` statically links the CRT so the dll is self-contained.

```bat
@echo off
REM Build an AVX2, self-contained rnnoise.dll with MSVC (VS2019 BuildTools).
REM Prereq: clone https://github.com/xiph/rnnoise.git @ 70f1d256 into ..\..\.deps\rnnoise
REM and copy rnnoise_buffer.c into its src\ (see README.md). vcvars path hard-coded
REM because vswhere is broken on this machine.
setlocal
cd /d "%~dp0"
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 ( echo [build] vcvars64 failed & exit /b 1 )

set SRC=%~dp0..\..\.deps\rnnoise

cl /nologo /O2 /MT /LD /arch:AVX2 ^
  /I "%SRC%\include" /I "%SRC%\src" ^
  /DWIN32 /DRNNOISE_BUILD /DDLL_EXPORT ^
  "%SRC%\src\denoise.c" ^
  "%SRC%\src\rnn.c" ^
  "%SRC%\src\pitch.c" ^
  "%SRC%\src\kiss_fft.c" ^
  "%SRC%\src\celt_lpc.c" ^
  "%SRC%\src\nnet.c" ^
  "%SRC%\src\nnet_default.c" ^
  "%SRC%\src\parse_lpcnet_weights.c" ^
  "%SRC%\src\rnnoise_data.c" ^
  "%SRC%\src\rnnoise_tables.c" ^
  "%SRC%\src\rnnoise_buffer.c" ^
  /Fe:rnnoise.dll
set RC=%errorlevel%
del *.obj *.exp *.lib 2>nul
echo [build] cl exit code: %RC%
endlocal & exit /b %RC%
```

- [ ] **Step 3: Create the README**

Create `build/rnnoise/README.md`:

```markdown
# rnnoise.dll build (AVX2, with batch wrapper)

The root `rnnoise.dll` shipped with MeetRec is **not** stock — it is rnnoise built
with MSVC `/arch:AVX2` (SIMD, ~2.9x faster than scalar) plus one custom exported
function, `rnnoise_process_buffer`, that denoise.py uses for GIL-releasing
multithreaded processing.

## Source
- Upstream: https://github.com/xiph/rnnoise.git
- Pinned commit: `70f1d256acd4b34a572f999a05c87bf00b67730d`

## Reproduce
1. `git clone https://github.com/xiph/rnnoise.git .deps/rnnoise`
2. `cd .deps/rnnoise && git checkout 70f1d256acd4b34a572f999a05c87bf00b67730d`
3. Copy `build/rnnoise/rnnoise_buffer.c` into `.deps/rnnoise/src/`
4. Run `build/rnnoise/build_rnnoise_dll.bat` (needs VS2019 BuildTools w/ C++; edit
   the hard-coded `vcvars64.bat` path inside if yours differs)
5. It emits `rnnoise.dll` next to the script; copy it to the repo root, replacing
   the existing one. Keep a scalar backup if desired.

## Notes
- `/MT` statically links the CRT so the dll has no VC-runtime dependency.
- No x86 RTCD / SIMD source files are needed; global `/arch:AVX2` is enough for
  `vec.h` to pick the `vec_avx.h` path (the "no vectorization" warning disappears).
- `rnnoise_process_buffer` just loops `rnnoise_process_frame` in C; output is
  bitwise identical to calling per-frame from Python.
```

- [ ] **Step 4: Commit**

```bash
git add build/rnnoise/rnnoise_buffer.c build/rnnoise/build_rnnoise_dll.bat build/rnnoise/README.md
git commit -m "build: persist AVX2 rnnoise.dll build script and batch wrapper"
```

---

## Task 2: Bind `rnnoise_process_buffer` and add a single-segment helper

**Files:**
- Modify: `denoise.py` (imports near line 1-6; `_RNNoiseBackend.__init__` lines 46-56; add method after `process`)
- Test: `tests/test_denoise.py`

- [ ] **Step 1: Add import and a shared pointer type**

In `denoise.py`, change the import block (currently lines 1-6) to add `ThreadPoolExecutor` and a module-level `c_float` pointer alias. Replace:

```python
import ctypes
import os
import sys
from dataclasses import dataclass

import numpy as np
```

with:

```python
import ctypes
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

_C_FLOAT_P = ctypes.POINTER(ctypes.c_float)
```

- [ ] **Step 2: Bind the batch symbol in `_RNNoiseBackend.__init__`**

At the end of `_RNNoiseBackend.__init__` (after the `rnnoise_process_frame.argtypes` block ending at line 56), append a guarded binding so older dlls without the symbol still work:

```python
        try:
            lib.rnnoise_process_buffer.restype = None
            lib.rnnoise_process_buffer.argtypes = [
                ctypes.c_void_p,
                ctypes.c_int,
                _C_FLOAT_P,
                _C_FLOAT_P,
            ]
            self._has_buffer = True
        except AttributeError:
            self._has_buffer = False
```

- [ ] **Step 3: Add `_process_buffer_segment` method**

Immediately after the `process` method (after line 78) in `_RNNoiseBackend`, add:

```python
    def _process_buffer_segment(self, scaled):
        """Denoise one contiguous, already-scaled (*32768), RNNOISE_FRAME-aligned
        float32 buffer with a single GIL-releasing C call. Returns scaled output."""
        scaled = np.ascontiguousarray(scaled, dtype=np.float32)
        out = np.empty_like(scaled)
        state = self._lib.rnnoise_create(None)
        try:
            nframes = len(scaled) // RNNOISE_FRAME
            self._lib.rnnoise_process_buffer(
                state,
                nframes,
                out.ctypes.data_as(_C_FLOAT_P),
                scaled.ctypes.data_as(_C_FLOAT_P),
            )
        finally:
            self._lib.rnnoise_destroy(state)
        return out
```

- [ ] **Step 4: Write the failing test**

Add to `tests/test_denoise.py` a new test class that uses the real backend when available:

```python
class RealBackendBufferTests(unittest.TestCase):
    def setUp(self):
        if not denoise.is_available() or not getattr(denoise._BACKEND, "_has_buffer", False):
            self.skipTest("real rnnoise backend with process_buffer not available")

    def test_buffer_segment_matches_per_frame(self):
        backend = denoise._BACKEND
        rng = np.random.default_rng(0)
        x = (0.2 * np.sin(2 * np.pi * 220 * np.arange(48000) / 48000)
             + 0.05 * rng.standard_normal(48000)).astype(np.float32)
        per_frame = backend.process(x)
        scaled = np.ascontiguousarray((x * 32768.0).astype(np.float32))
        seg = backend._process_buffer_segment(scaled) / 32768.0
        np.testing.assert_allclose(seg, per_frame, atol=1e-6)
```

- [ ] **Step 5: Run the test**

Run: `python -m pytest tests/test_denoise.py::RealBackendBufferTests -v`
Expected: PASS if the AVX2 dll is installed; SKIPPED otherwise. (On the dev machine it must PASS.)

- [ ] **Step 6: Commit**

```bash
git add denoise.py tests/test_denoise.py
git commit -m "feat(denoise): bind rnnoise_process_buffer and add single-segment helper"
```

---

## Task 3: Add the `_parallel_denoise` orchestration function

**Files:**
- Modify: `denoise.py` (add constants after line 15; add function near other module helpers, e.g. after `_advance_audio`)
- Test: `tests/test_denoise.py`

- [ ] **Step 1: Add tuning constants**

In `denoise.py`, after `RNNOISE_FRAME = 480` (line 15), add:

```python
# Parallel denoise tuning. Threads default to physical-core estimate; warmup is a
# 1 s overlap prefix per segment so RNNoise's stateful RNN converges before the
# kept region; audio shorter than the min isn't worth splitting.
NOISE_REDUCTION_THREADS = max(1, (os.cpu_count() or 2) // 2)
NOISE_REDUCTION_WARMUP_FRAMES = 100          # 100 * 480 / 48000 = 1.0 s
NOISE_REDUCTION_MIN_PARALLEL_FRAMES = 6000   # ~60 s @48k; below this, single-thread
```

- [ ] **Step 2: Write the failing tests**

Add to `tests/test_denoise.py`:

```python
class ParallelDenoiseOrchestrationTests(unittest.TestCase):
    def _scaled(self, n_frames):
        rng = np.random.default_rng(1)
        return np.ascontiguousarray(
            rng.standard_normal(n_frames * denoise.RNNOISE_FRAME).astype(np.float32)
        )

    def test_identity_segment_reconstructs_input(self):
        scaled = self._scaled(8000)  # > MIN_PARALLEL_FRAMES so it splits
        out = denoise._parallel_denoise(
            scaled, num_threads=4, warmup_frames=100, segment_fn=lambda c: c
        )
        np.testing.assert_array_equal(out, scaled)

    def test_short_audio_stays_single_segment(self):
        scaled = self._scaled(100)  # < MIN_PARALLEL_FRAMES
        seen = []
        denoise._parallel_denoise(
            scaled, num_threads=4, warmup_frames=100,
            segment_fn=lambda c: seen.append(len(c)) or c,
        )
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0], len(scaled))

    def test_non_first_segments_get_warmup_prefix(self):
        scaled = self._scaled(8000)
        lengths = []
        denoise._parallel_denoise(
            scaled, num_threads=4, warmup_frames=100,
            segment_fn=lambda c: lengths.append(len(c)) or c,
        )
        seg = (8000 // 4) * denoise.RNNOISE_FRAME
        warm = 100 * denoise.RNNOISE_FRAME
        # first segment has no warmup, later ones carry a warmup prefix
        self.assertEqual(min(lengths), seg)
        self.assertGreaterEqual(max(lengths), seg + warm)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/test_denoise.py::ParallelDenoiseOrchestrationTests -v`
Expected: FAIL with `AttributeError: module 'denoise' has no attribute '_parallel_denoise'`

- [ ] **Step 4: Implement `_parallel_denoise`**

In `denoise.py`, after `_advance_audio` (after line 114), add:

```python
def _parallel_denoise(scaled, num_threads, warmup_frames, segment_fn,
                      min_parallel_frames=NOISE_REDUCTION_MIN_PARALLEL_FRAMES):
    """Split a scaled, RNNOISE_FRAME-aligned buffer into contiguous segments,
    denoise each in a thread via segment_fn, and concatenate. Each non-first
    segment is processed with a warmup_frames overlap prefix that is discarded
    from the output (lets RNNoise state converge across the cut). Frame-aligned
    throughout. segment_fn(chunk)->same-length chunk."""
    total_frames = len(scaled) // RNNOISE_FRAME
    k = max(1, min(int(num_threads), total_frames))
    if k <= 1 or total_frames < min_parallel_frames:
        return segment_fn(scaled)

    seg_frames = total_frames // k
    bounds = [
        (i * seg_frames, total_frames if i == k - 1 else (i + 1) * seg_frames)
        for i in range(k)
    ]
    results = [None] * k

    def work(idx):
        start_f, end_f = bounds[idx]
        warm_start = max(0, start_f - int(warmup_frames))
        chunk = np.ascontiguousarray(
            scaled[warm_start * RNNOISE_FRAME:end_f * RNNOISE_FRAME]
        )
        processed = segment_fn(chunk)
        keep_from = (start_f - warm_start) * RNNOISE_FRAME
        keep_to = (end_f - warm_start) * RNNOISE_FRAME
        results[idx] = processed[keep_from:keep_to]

    with ThreadPoolExecutor(max_workers=k) as executor:
        list(executor.map(work, range(k)))
    return np.concatenate(results)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_denoise.py::ParallelDenoiseOrchestrationTests -v`
Expected: PASS (3 tests)

- [ ] **Step 6: Commit**

```bash
git add denoise.py tests/test_denoise.py
git commit -m "feat(denoise): add segment-parallel orchestration with warmup overlap"
```

---

## Task 4: Add `process_parallel` with fallback

**Files:**
- Modify: `denoise.py` (add method to `_RNNoiseBackend` after `_process_buffer_segment`)
- Test: `tests/test_denoise.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_denoise.py`. These use a tiny fake lib so no real dll is needed; they exercise the pad/scale wrapper and the fallback path:

```python
class ProcessParallelTests(unittest.TestCase):
    def _make_backend(self):
        backend = denoise._RNNoiseBackend.__new__(denoise._RNNoiseBackend)
        backend._lib = None
        backend._has_buffer = True
        backend._process_buffer_segment = lambda scaled: scaled  # identity
        return backend

    def test_parallel_identity_roundtrips_signal(self):
        backend = self._make_backend()
        x = np.linspace(-0.5, 0.5, 8000 * denoise.RNNOISE_FRAME, dtype=np.float32)
        out = backend.process_parallel(x, num_threads=4, warmup_frames=100)
        self.assertEqual(len(out), len(x))
        np.testing.assert_allclose(out, x, atol=1e-3)

    def test_parallel_falls_back_to_single_segment(self):
        from unittest.mock import patch
        backend = self._make_backend()  # identity segment -> fallback succeeds
        x = np.linspace(-0.5, 0.5, 8000 * denoise.RNNOISE_FRAME, dtype=np.float32)
        with patch.object(denoise, "_parallel_denoise", side_effect=RuntimeError("boom")):
            out = backend.process_parallel(x, num_threads=4, warmup_frames=100)
        # parallel path raised -> single-segment fallback (identity) returned input
        np.testing.assert_allclose(out, x, atol=1e-3)

    def test_parallel_without_buffer_uses_per_frame(self):
        backend = self._make_backend()
        backend._has_buffer = False
        backend.process = lambda mono: np.asarray(mono, dtype=np.float32) * 0.0
        out = backend.process_parallel(np.ones(960, dtype=np.float32))
        np.testing.assert_allclose(out, np.zeros(960, dtype=np.float32))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_denoise.py::ProcessParallelTests -v`
Expected: FAIL with `AttributeError: '_RNNoiseBackend' object has no attribute 'process_parallel'`

- [ ] **Step 3: Implement `process_parallel`**

In `_RNNoiseBackend`, after `_process_buffer_segment`, add:

```python
    def process_parallel(self, mono_48k_pm1, num_threads=None, warmup_frames=None):
        """Denoise a full mono 48k [-1,1] stream using segment-parallel threads.
        Falls back to per-frame `process` if the dll lacks process_buffer, and to
        single-segment buffering if the parallel path raises."""
        if not getattr(self, "_has_buffer", False):
            return self.process(mono_48k_pm1)
        if num_threads is None:
            num_threads = NOISE_REDUCTION_THREADS
        if warmup_frames is None:
            warmup_frames = NOISE_REDUCTION_WARMUP_FRAMES

        x = np.asarray(mono_48k_pm1, dtype=np.float32)
        pad = (-len(x)) % RNNOISE_FRAME
        if pad:
            x = np.concatenate([x, np.zeros(pad, dtype=np.float32)])
        scaled = np.ascontiguousarray((x * 32768.0).astype(np.float32))

        try:
            out = _parallel_denoise(
                scaled, num_threads, warmup_frames, self._process_buffer_segment
            )
        except Exception as exc:
            print(f"Parallel denoise failed, using single segment: {exc}")
            out = self._process_buffer_segment(scaled)

        out = out / 32768.0
        if pad:
            out = out[:len(out) - pad]
        return out.astype(np.float32)
```

Note: `test_parallel_falls_back_to_single_segment` patches `_parallel_denoise` to raise, so the `except` runs `_process_buffer_segment` (identity here) and returns the input — proving the fallback path is taken and its result is used.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_denoise.py::ProcessParallelTests -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add denoise.py tests/test_denoise.py
git commit -m "feat(denoise): add process_parallel with per-frame and single-segment fallbacks"
```

---

## Task 5: Route `reduce_noise` through `process_parallel`

**Files:**
- Modify: `denoise.py` (add `_denoise_mono` helper; change line 136 in `reduce_noise`)
- Test: `tests/test_denoise.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_denoise.py` a fake backend exposing `process_parallel`, to prove `reduce_noise` routes through it when present:

```python
class ReduceNoiseParallelRoutingTests(unittest.TestCase):
    def setUp(self):
        self._orig = denoise._BACKEND
        self.addCleanup(lambda: setattr(denoise, "_BACKEND", self._orig))

    def test_reduce_noise_uses_process_parallel_when_present(self):
        calls = {"parallel": 0, "frame": 0}

        class FakeParallel:
            samplerate = 48000
            frame_size = 480

            def process(self, mono):
                calls["frame"] += 1
                return np.asarray(mono, dtype=np.float32)

            def process_parallel(self, mono, num_threads=None, warmup_frames=None):
                calls["parallel"] += 1
                return np.asarray(mono, dtype=np.float32)

        denoise._BACKEND = FakeParallel()
        data = np.full((48000, 1), 0.2, dtype=np.float32)
        out, stats = reduce_noise(data, 48000, NoiseReductionConfig(enabled=True, mix=1.0))
        self.assertTrue(stats["applied"])
        self.assertEqual(calls["parallel"], 1)
        self.assertEqual(calls["frame"], 0)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_denoise.py::ReduceNoiseParallelRoutingTests -v`
Expected: FAIL (`calls["parallel"] == 0`, because `reduce_noise` still calls `process`)

- [ ] **Step 3: Add the helper and switch `reduce_noise`**

In `denoise.py`, add a module-level helper near `_parallel_denoise`:

```python
def _denoise_mono(backend, up):
    """Use the backend's parallel path when it offers one, else per-frame."""
    if hasattr(backend, "process_parallel"):
        return backend.process_parallel(up)
    return backend.process(up)
```

Then in `reduce_noise`, change line 136 from:

```python
            wet_up = _BACKEND.process(up)
```

to:

```python
            wet_up = _denoise_mono(_BACKEND, up)
```

- [ ] **Step 4: Run the new test and the full denoise suite**

Run: `python -m pytest tests/test_denoise.py -v`
Expected: PASS — the new routing test passes, and all pre-existing `_FakeBackend` tests still pass (it has no `process_parallel`, so `_denoise_mono` uses `process`).

- [ ] **Step 5: Commit**

```bash
git add denoise.py tests/test_denoise.py
git commit -m "feat(denoise): route reduce_noise through process_parallel when available"
```

---

## Task 6: End-to-end verification

**Files:** none (verification only)

- [ ] **Step 1: Full test suite**

Run: `python -m pytest tests/ -v`
Expected: all green (no regressions in `test_denoise.py`, `test_audio_processing.py`, etc.).

- [ ] **Step 2: Real speed check (dev machine, AVX2 dll installed)**

Run this one-off and confirm the parallel path is ~5-7x faster than single-thread and output matches single-thread closely:

```bash
python -c "import time,numpy as np,denoise; b=denoise._BACKEND; N=48000*120; rng=np.random.default_rng(3); x=((0.2*np.sin(2*np.pi*220*np.arange(N)/48000)+0.05*rng.standard_normal(N))).astype(np.float32); t=time.perf_counter(); a=b.process_parallel(x); dtp=time.perf_counter()-t; t=time.perf_counter(); s=b._process_buffer_segment(np.ascontiguousarray((x*32768.0).astype(np.float32)))/32768.0; dts=time.perf_counter()-t; print('parallel 120s=%.3fs -> 2h %.2fmin'%(dtp,dtp)); print('single   120s=%.3fs'%dts); print('speedup=%.2fx'%(dts/dtp))"
```

Expected: parallel ~0.6-1.1 s/120s (→ ~35-65 s/2h), speedup ~5-7x.

- [ ] **Step 3: Confirm the GUI path is untouched**

Confirm `audio_recorder.py` was not modified and still calls `denoise.reduce_noise` unchanged (the speedup is entirely inside `denoise.py`). No commit needed.

---

## Self-Review

- **Spec coverage:** AVX2 dll (prereq, done) ✓; persisted build artifacts (Task 1) ✓; `process_buffer` binding + single segment (Task 2) ✓; segment-parallel + warmup overlap (Task 3) ✓; `process_parallel` + fallbacks (Task 4) ✓; `reduce_noise` integration, audio_recorder untouched (Task 5) ✓; thread/warmup/threshold constants (Task 3) ✓; verification (Task 6) ✓.
- **Type/name consistency:** `_process_buffer_segment`, `_parallel_denoise(scaled, num_threads, warmup_frames, segment_fn, min_parallel_frames=...)`, `process_parallel(mono_48k_pm1, num_threads=None, warmup_frames=None)`, `_denoise_mono(backend, up)`, `_has_buffer`, `_C_FLOAT_P`, constants `NOISE_REDUCTION_THREADS/_WARMUP_FRAMES/_MIN_PARALLEL_FRAMES` — used consistently across tasks.
- **Frame alignment:** all slicing is in whole RNNOISE_FRAME units; `scaled` is padded to a frame multiple before any split; warmup and bounds are in frames. No sub-frame cuts.
- **Backward compat:** older dll without `rnnoise_process_buffer` → `_has_buffer=False` → `process_parallel` delegates to per-frame `process`; existing `_FakeBackend` (no `process_parallel`) → `_denoise_mono` uses `process`. Existing tests unaffected.
