from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EchoSuppressionConfig:
    enabled: bool = False
    block_seconds: float = 15.0
    overlap_seconds: float = 0.15
    min_delay_ms: int = -100
    max_delay_ms: int = 500
    max_delay_jump_ms: int = 60
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
    return np.clip(np.asarray(data, dtype=np.float64), -float(limit), float(limit))


def align_reference_to_target(reference, target_frames, delay_frames):
    reference = as_2d_float_audio(reference)
    target_frames = int(target_frames)
    delay_frames = int(delay_frames)
    aligned = np.zeros((target_frames, reference.shape[1]), dtype=np.float32)
    if target_frames <= 0 or len(reference) == 0:
        return aligned
    if delay_frames >= 0:
        source_start, target_start = 0, delay_frames
    else:
        source_start, target_start = -delay_frames, 0
    if source_start >= len(reference) or target_start >= target_frames:
        return aligned
    frames = min(len(reference) - source_start, target_frames - target_start)
    if frames > 0:
        aligned[target_start:target_start + frames] = reference[source_start:source_start + frames]
    return aligned


def _take_aligned_global(reference, start, length, delay):
    reference = as_2d_float_audio(reference)
    length = int(length)
    out = np.zeros((length, reference.shape[1]), dtype=np.float32)
    dst_start = max(0, delay - start)
    src_start = max(0, start - delay)
    n = min(length - dst_start, len(reference) - src_start)
    if n > 0:
        out[dst_start:dst_start + n] = reference[src_start:src_start + n]
    return out


def _mono(data):
    audio = as_2d_float_audio(data)
    return np.mean(audio, axis=1).astype(np.float32, copy=False)


def _preemphasize(x):
    if len(x) == 0:
        return x
    return np.concatenate(([x[0]], x[1:] - 0.95 * x[:-1])).astype(np.float32)


def _next_pow2(n):
    p = 1
    while p < n:
        p <<= 1
    return p


def estimate_block_delay(mic_block, ref_block, lo_frame, hi_frame):
    """Find delay where mic[m] is best explained by ref[m - delay]."""
    a = _preemphasize(_mono(mic_block))
    b = _preemphasize(_mono(ref_block))
    n = len(a)
    if n == 0 or len(b) == 0:
        return 0, 0.0
    nfft = _next_pow2(2 * max(n, len(b)))
    fa = np.fft.rfft(a, nfft)
    fb = np.fft.rfft(b, nfft)
    res = np.fft.irfft(fa * np.conj(fb), nfft)
    energy = float(np.sqrt(np.dot(a, a) * np.dot(b, b)))
    if energy <= 1e-12:
        return 0, 0.0

    lo = int(lo_frame)
    hi = int(hi_frame)
    if lo > hi:
        lo, hi = hi, lo
    best_lag, best_val = lo, -np.inf
    for lag in range(lo, hi + 1):
        idx = lag if lag >= 0 else nfft + lag
        if 0 <= idx < nfft:
            v = res[idx]
            if v > best_val:
                best_val = v
                best_lag = lag
    return int(best_lag), float(best_val / energy)


def _trapezoid_window(length, ramp):
    length = int(length)
    ramp = int(min(ramp, length // 2))
    w = np.ones(length, dtype=np.float32)
    if ramp > 0:
        edge = np.linspace(0.0, 1.0, ramp, endpoint=False, dtype=np.float32)
        w[:ramp] = edge
        w[length - ramp:] = edge[::-1]
    return w


def suppress_reference_echo(mic, reference, samplerate, config):
    mic = as_2d_float_audio(mic)
    reference = as_2d_float_audio(reference)
    if not config.enabled:
        return mic.copy(), {"applied": False, "reason": "disabled"}

    n = len(mic)
    channels = mic.shape[1]
    if n == 0:
        return mic.copy(), {"applied": False, "reason": "empty_audio"}

    ref_mono = _mono(reference[:min(len(reference), n)])
    if len(ref_mono) == 0 or float(np.percentile(np.abs(ref_mono), 95)) < config.analysis_floor:
        return mic.copy(), {"applied": False, "reason": "silent_reference"}

    block = max(1, int(round(config.block_seconds * samplerate)))
    overlap = max(0, int(round(config.overlap_seconds * samplerate)))
    overlap = min(overlap, block // 2)
    hop = max(1, block - overlap)
    lo = int(round(config.min_delay_ms * samplerate / 1000.0))
    hi = int(round(config.max_delay_ms * samplerate / 1000.0))
    jump = max(0, int(round(config.max_delay_jump_ms * samplerate / 1000.0)))

    out = np.zeros((n, channels), dtype=np.float32)
    wsum = np.zeros(n, dtype=np.float32)

    blocks_total = 0
    blocks_suppressed = 0
    delays = []
    gains = []
    corrs = []
    prev_delay = None

    start = 0
    while start < n:
        end = min(n, start + block)
        length = end - start
        mic_block = mic[start:end]
        ref_same = _take_aligned_global(reference, start, length, 0)

        if prev_delay is None:
            search_lo, search_hi = lo, hi
        else:
            search_lo = max(lo, prev_delay - jump)
            search_hi = min(hi, prev_delay + jump)
        delay, corr = estimate_block_delay(mic_block, ref_same, search_lo, search_hi)

        aligned_for_energy = _take_aligned_global(reference, start, length, delay)
        ref_energy = float(np.sum(aligned_for_energy ** 2))
        win = _trapezoid_window(length, overlap)

        blocks_total += 1
        if corr < config.min_correlation or ref_energy <= 1e-9:
            cleaned_block = mic_block
            if prev_delay is not None:
                delay = prev_delay
        else:
            aligned = aligned_for_energy
            num = 0.0
            den = 0.0
            for ch in range(channels):
                ref_ch = aligned[:, min(ch, aligned.shape[1] - 1)]
                num += float(np.dot(mic_block[:, ch], ref_ch))
                den += float(np.dot(ref_ch, ref_ch))
            gain = 0.0 if den <= 1e-12 else float(np.clip(num / den, 0.0, config.max_echo_gain))
            cleaned_block = mic_block - aligned * gain * float(config.suppression_strength)
            if prev_delay is not None and abs(delay - prev_delay) > int(round(0.02 * samplerate)):
                previous_aligned = _take_aligned_global(reference, start, length, prev_delay)
                residual_num = 0.0
                residual_den = 0.0
                for ch in range(channels):
                    ref_ch = previous_aligned[:, min(ch, previous_aligned.shape[1] - 1)]
                    residual_num += float(np.dot(cleaned_block[:, ch], ref_ch))
                    residual_den += float(np.dot(ref_ch, ref_ch))
                residual_gain = (
                    0.0
                    if residual_den <= 1e-12
                    else float(np.clip(residual_num / residual_den, 0.0, config.max_echo_gain))
                )
                cleaned_block = cleaned_block - previous_aligned * residual_gain * float(config.suppression_strength)
            blocks_suppressed += 1
            gains.append(gain)
            corrs.append(corr)
            delays.append(delay)
            prev_delay = delay

        if prev_delay is None:
            prev_delay = delay

        out[start:end] += cleaned_block * win[:, None]
        wsum[start:end] += win
        start += hop

    wsum = np.maximum(wsum, 1e-6)
    out = out / wsum[:, None]
    out = apply_limiter(out)

    applied = blocks_suppressed > 0
    stats = {
        "applied": applied,
        "reason": "applied" if applied else "low_correlation",
        "blocks_total": blocks_total,
        "blocks_suppressed": blocks_suppressed,
        "mean_delay_ms": round(1000.0 * float(np.mean(delays)) / samplerate, 3) if delays else 0.0,
        "delay_spread_ms": round(1000.0 * float(np.max(delays) - np.min(delays)) / samplerate, 3) if delays else 0.0,
        "mean_gain": round(float(np.mean(gains)), 6) if gains else 0.0,
        "mean_correlation": round(float(np.mean(corrs)), 6) if corrs else 0.0,
    }
    return out, stats


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
            "applied": False, "reason": "insufficient_activity",
            "gain": 1.0, "active_seconds": round(active_seconds, 3),
        }

    active = np.abs(data[active_mask])
    active = active[active >= config.active_floor]
    if active.size == 0:
        return data.copy(), {
            "applied": False, "reason": "below_active_floor",
            "gain": 1.0, "active_seconds": round(active_seconds, 3),
        }

    rms = float(np.sqrt(np.mean(active ** 2)))
    percentile = float(np.percentile(active, config.reference_percentile))
    reference_level = max(rms, percentile)
    if not np.isfinite(reference_level) or reference_level <= 0.0:
        return data.copy(), {
            "applied": False, "reason": "invalid_reference_level",
            "gain": 1.0, "active_seconds": round(active_seconds, 3),
        }

    gain = min(config.target_level / reference_level, config.max_gain)
    return apply_limiter(data * gain, config.limit), {
        "applied": True, "reason": "applied",
        "gain": round(float(gain), 6), "active_seconds": round(active_seconds, 3),
        "reference_level": round(reference_level, 6),
    }
