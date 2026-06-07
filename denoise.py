import ctypes
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

_C_FLOAT_P = ctypes.POINTER(ctypes.c_float)

try:
    from scipy.signal import resample_poly
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

RNNOISE_SAMPLERATE = 48000
RNNOISE_FRAME = 480

# Parallel denoise tuning. Threads default to physical-core estimate; warmup is a
# 1 s overlap prefix per segment so RNNoise's stateful RNN converges before the
# kept region; audio shorter than the min isn't worth splitting.
NOISE_REDUCTION_THREADS = max(1, (os.cpu_count() or 2) // 2)
NOISE_REDUCTION_WARMUP_FRAMES = 100          # 100 * 480 / 48000 = 1.0 s
NOISE_REDUCTION_MIN_PARALLEL_FRAMES = 6000   # ~60 s @48k; below this, single-thread


@dataclass(frozen=True)
class NoiseReductionConfig:
    enabled: bool = False
    mix: float = 1.0
    latency_ms: float = 20.0


def _candidate_dll_paths():
    names = ["rnnoise.dll", "librnnoise.dll", "librnnoise-0.dll"]
    roots = []
    env = os.environ.get("RNNOISE_DLL")
    if env:
        roots.append(os.path.dirname(env))
        names.insert(0, os.path.basename(env))
    if getattr(sys, "frozen", False):
        roots.append(os.path.dirname(sys.executable))
        roots.append(getattr(sys, "_MEIPASS", os.path.dirname(sys.executable)))
    roots.append(os.path.dirname(os.path.abspath(__file__)))
    roots.append(os.getcwd())
    for root in roots:
        for name in names:
            yield os.path.join(root, name)


class _RNNoiseBackend:
    samplerate = RNNOISE_SAMPLERATE
    frame_size = RNNOISE_FRAME

    def __init__(self, lib):
        self._lib = lib
        lib.rnnoise_create.restype = ctypes.c_void_p
        lib.rnnoise_create.argtypes = [ctypes.c_void_p]
        lib.rnnoise_destroy.argtypes = [ctypes.c_void_p]
        lib.rnnoise_process_frame.restype = ctypes.c_float
        lib.rnnoise_process_frame.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
        ]
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

    def process(self, mono_48k_pm1):
        x = np.asarray(mono_48k_pm1, dtype=np.float32)
        pad = (-len(x)) % RNNOISE_FRAME
        if pad:
            x = np.concatenate([x, np.zeros(pad, dtype=np.float32)])
        scaled = (x * 32768.0).astype(np.float32)
        state = self._lib.rnnoise_create(None)
        try:
            out = np.empty_like(scaled)
            in_buf = (ctypes.c_float * RNNOISE_FRAME)()
            out_buf = (ctypes.c_float * RNNOISE_FRAME)()
            for i in range(0, len(scaled), RNNOISE_FRAME):
                in_buf[:] = scaled[i:i + RNNOISE_FRAME]
                self._lib.rnnoise_process_frame(state, out_buf, in_buf)
                out[i:i + RNNOISE_FRAME] = np.ctypeslib.as_array(out_buf)
        finally:
            self._lib.rnnoise_destroy(state)
        out = out / 32768.0
        if pad:
            out = out[:len(out) - pad]
        return out.astype(np.float32)

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


def _load_backend():
    for path in _candidate_dll_paths():
        if os.path.exists(path):
            try:
                return _RNNoiseBackend(ctypes.CDLL(path))
            except Exception as e:
                print(f"Failed to load RNNoise from {path}: {e}")
    return None


_BACKEND = _load_backend()


def is_available():
    return _BACKEND is not None


def _resample(x, sr_in, sr_out):
    if sr_in == sr_out:
        return x
    if not _HAVE_SCIPY:
        raise RuntimeError("scipy is required for resampling to RNNoise samplerate")
    from math import gcd
    g = gcd(int(sr_in), int(sr_out))
    return resample_poly(x, int(sr_out) // g, int(sr_in) // g).astype(np.float32)


def _advance_audio(data, frames):
    frames = int(max(0, frames))
    if frames <= 0 or len(data) == 0:
        return data
    if frames >= len(data):
        return np.zeros_like(data)
    return np.concatenate([data[frames:], np.zeros(frames, dtype=np.float32)])


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


def reduce_noise(data, samplerate, config):
    audio = np.asarray(data, dtype=np.float32)
    if audio.ndim == 1:
        audio = audio.reshape(-1, 1)

    if not config.enabled:
        return audio.copy(), {"applied": False, "reason": "disabled"}
    if _BACKEND is None:
        return audio.copy(), {"applied": False, "reason": "backend_unavailable"}

    target_sr = _BACKEND.samplerate
    mix = float(np.clip(config.mix, 0.0, 1.0))
    latency_ms = float(max(0.0, config.latency_ms))
    latency_frames = int(round(latency_ms * float(samplerate) / 1000.0))
    out = np.empty_like(audio)
    try:
        for ch in range(audio.shape[1]):
            mono = audio[:, ch]
            up = _resample(mono, samplerate, target_sr)
            wet_up = _BACKEND.process(up)
            wet = _resample(wet_up, target_sr, samplerate)
            if len(wet) < len(mono):
                wet = np.concatenate([wet, np.zeros(len(mono) - len(wet), dtype=np.float32)])
            else:
                wet = wet[:len(mono)]
            wet = _advance_audio(wet.astype(np.float32, copy=False), latency_frames)
            out[:, ch] = (1.0 - mix) * mono + mix * wet
    except Exception as e:
        print(f"Noise reduction failed: {e}")
        return audio.copy(), {"applied": False, "reason": "error"}

    return out, {
        "applied": True, "reason": "applied",
        "mix": round(mix, 3), "latency_ms": round(latency_ms, 3),
        "channels": int(audio.shape[1]),
        "samplerate": int(samplerate),
    }
