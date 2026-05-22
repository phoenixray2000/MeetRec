# Both 模式回声抑制 + RNNoise 降噪 + 源调平 实施方案

> 本方案取代 `2026-05-22-reference-echo-suppression.md`。它修复了旧方案的两处测试缺陷,
> 把"全长暴力延迟搜索"换成"15 秒分块 + FFT 互相关 + 延迟跟踪"(同时解决性能与时钟漂移),
> 把"裁剪活动检测器门控"换成"能量地板门控"(救回轻声麦克风),并新增 RNNoise 降噪环节。

> **给执行者:** 按 Task 顺序逐步执行,每个 Step 都有"先写失败测试 → 实现 → 跑测试 → 提交"。
> 步骤用 `- [ ]` 跟踪。**注意:当前分支 `codex-silence-auto-stop` 工作区有大量与本方案无关的
> 未提交改动,提交时务必只 `git add` 本方案明确列出的文件,不要用 `git add -A`/`git add .`。**

---

## 目标

为 Both 模式提供一条"先抑制扬声器回声、再对麦克风降噪、再分轨调平、最后混音"的后处理链路,
解决三个真实痛点:

1. **扬声器声漏进麦克风(回声)**:用系统声(loopback)作参考,从麦克风轨里减掉泄漏成分。
2. **麦克风天生小声、被系统声盖住**:分轨调平,把麦克风和系统声各自拉到相近电平再混音。
3. **麦克风 +30dB 增益带来的底噪**:对麦克风轨做 RNNoise 降噪。

麦克风增益不可下调(否则会议对端听不到),因此降噪是必需项,而非可选优化。

---

## 关键决策摘要

| 决策点 | 选择 | 理由 |
|---|---|---|
| 延迟搜索算法 | 15 秒分块 + FFT 互相关 + 延迟跟踪 | 全长暴力搜索在 10 分钟录音上要数分钟、近乎卡死;分块 FFT 把它降到亚秒级,且逐块对齐天然吸收时钟漂移 |
| 分块大小 | 15 秒 | 只看稳定性与性能、不追求完美漂移消除时的甜点:块够大使延迟估计稳、性能对块大小不敏感,15s 内漂移 < ~1ms 不破坏块内对齐 |
| 调平门控 | 能量地板(0.001 ≈ -60dB) | 裁剪用的活动检测器阈值高达 -30dB,会把 -34dB 的轻声麦克风误判为"无声"而不提升;能量地板能救回轻声,又能跳过纯静音轨 |
| 降噪库 | RNNoise(ctypes 调 `rnnoise.dll`) | 专为语音、保护清辅音、运行时几乎零开销;代价仅在打包(把 DLL 塞进 PyInstaller)与 48k 重采样 |
| 降噪可调 | wet/dry 混合系数 `mix` | RNNoise 无强度参数,用干湿混合实现"温和度",缓解吃人声风险 |
| 处理顺序 | AEC → (mic)降噪 → 分轨调平 → 混音 → 限幅 → 裁剪 | 降噪要在 AEC 之后(降噪改频谱会破坏 AEC 对齐),在调平之前(先压噪再放大,调平电平更准) |

---

## 处理链路

**Both 模式:**

```
raw mic.wav, raw loopback.wav
  → AEC: suppress_reference_echo(mic, loopback)          # 分块,loopback 作参考清洗 mic
  → mic 降噪: reduce_noise(mic)                           # RNNoise,可选开关
  → 分轨调平: level_active_source(mic) / level_active_source(loopback)   # 能量地板门控
  → 混音 _mix_audio_data(mic, loopback, limit_output=True)
  → 最终裁剪 _maybe_trim_final_wav(mixed)                 # 可选,作用于最终输出
  → 导出 _write_final_output
```

**单源模式(mic 或 loopback):**

```
raw source.wav
  → (仅 mic 源且降噪开启)降噪 reduce_noise
  → (normalize 开启)整轨归一化 _normalize_audio        # 沿用旧逻辑,地板 0.001,对轻声有效
  → 最终裁剪 _maybe_trim_final_wav
  → 导出
```

实时采集(`RawRecorder`)完全不变,仍写完整的 mic / loopback 临时 WAV;所有处理都在停止录音后进行。

---

## 处理规则(实现约束,非偏好)

1. AEC 以 loopback 为参考、mic 为待清洗目标;必须在降噪与调平之前、在任何破坏性裁剪之前运行。
2. AEC 分块处理:块长 15s,块间重叠淡化,逐块估计延迟/增益,块间延迟做跳变限幅以跟踪漂移。
3. AEC 对参考能量过低或相关过低的块跳过减法(不硬减),并继承上一块延迟保持跟踪连续。
4. 降噪只作用于含麦克风的轨(Both 的 mic 轨、单源 mic);loopback 轨不降噪。
5. 降噪在 AEC 之后、调平之前。降噪库不可用时必须优雅降级(跳过降噪、照常出片、元数据记录原因)。
6. 调平门控用能量地板(`NORMALIZE_ACTIVE_FLOOR`),不得用裁剪活动检测器掩码。
7. 调平必须跳过"几乎全是 < 地板 的纯静音轨",避免放大残余噪声。
8. 混音后只做一次限幅,不得再做整体响度归一化。
9. 仅 `source_mode == "both"` 时 AEC 才有意义;`mic`/`loopback` 模式下 `echo_suppression=True` 必须被安全忽略。
10. 裁剪只作用于最终准备好的 WAV,不得再像旧代码那样原地改写源临时 WAV。
11. 所有新设置默认关闭(`echo_suppression`、`noise_reduction` 默认 `False`)。

---

## 依赖与打包

- **新增 Python 依赖:`scipy`**(仅用于降噪环节的 16k↔48k 重采样 `scipy.signal.resample_poly`)。
  核心 DSP(`audio_processing.py`)保持纯 numpy,可独立测试。
- **新增原生依赖:`rnnoise.dll`**(RNNoise 编译产物)。通过 ctypes 加载,打包时作为 binary 收进 PyInstaller。
- 项目当前无 `requirements.txt`;本方案 Task 7 会创建一份,记录已知依赖。

---

## 风险与降级(RNNoise)

RNNoise 的 Python 生态不稳定,**这是本方案风险最高的一环**。应对策略:

- 降噪后端隔离在独立模块 `denoise.py`,对外只暴露 `reduce_noise()` / `is_available()`。
- DLL 加载失败、找不到、或处理异常 → `is_available()` 返回 `False`,录音流程跳过降噪并在元数据写
  `noise_reduction_reason="backend_unavailable"`,**绝不让降噪问题导致录音失败**。
- 单元测试通过 monkeypatch 注入假后端,不依赖真实 DLL,保证 CI / 无 DLL 环境也能跑全测试。
- Task 7 给出获取 / 编译 `rnnoise.dll` 的具体步骤;若执行环境暂时拿不到 DLL,前 6 个 Task 仍可
  完整开发与测试(降噪走降级路径)。

---

## 文件结构

- **新建** `audio_processing.py` — 纯 numpy DSP:对齐、FFT 互相关延迟估计、分块 AEC、调平、限幅。
- **新建** `denoise.py` — RNNoise 封装 + 重采样 + 降级。
- **新建** `tests/test_audio_processing.py` — DSP 单元测试。
- **新建** `tests/test_denoise.py` — 降噪封装测试(mock 后端)。
- **修改** `audio_recorder.py` — 编排处理顺序、设置/状态/元数据、裁剪改为最终输出裁剪。
- **修改** `tests/test_audio_output_profile.py` — Both 模式调平改为地板门控后的期望(基本无需改断言,见 Task 4)。
- **修改** `tests/test_audio_trim.py` — Both 模式裁剪测试改为"混音后裁剪",复用既有 16k 数据与帧数常量。
- **修改** `gui.py` — 新增"减少 Both 模式回声""麦克风降噪"两个开关并持久化、传入 `AudioRecorder`。
- **修改** `tests/test_gui_hotkeys.py` — 新增设置项的 UI/持久化/传参测试。
- **修改** `README.md` — 文档化回声抑制、降噪、调平与裁剪顺序。
- **修改** `MeetRec.spec` — 把 `rnnoise.dll` 收进 binaries。
- **新建** `requirements.txt` — 记录依赖。

---

### Task 1: 纯 DSP —— FFT 延迟估计 + 分块 AEC + 地板调平

**Files:**
- 新建 `tests/test_audio_processing.py`
- 新建(本 Task)`audio_processing.py`

- [ ] **Step 1: 写失败测试**

创建 `tests/test_audio_processing.py`:

```python
import unittest

import numpy as np

from audio_processing import (
    EchoSuppressionConfig,
    SourceLevelingConfig,
    align_reference_to_target,
    apply_limiter,
    estimate_block_delay,
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


class AlignmentTests(unittest.TestCase):
    def test_align_positive_delay(self):
        reference = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
        aligned = align_reference_to_target(reference, target_frames=6, delay_frames=2)
        self.assertEqual(aligned[:, 0].tolist(), [0.0, 0.0, 1.0, 2.0, 3.0, 0.0])

    def test_align_negative_delay(self):
        reference = np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32)
        aligned = align_reference_to_target(reference, target_frames=5, delay_frames=-1)
        self.assertEqual(aligned[:, 0].tolist(), [2.0, 3.0, 4.0, 0.0, 0.0])


class BlockDelayTests(unittest.TestCase):
    def test_estimate_block_delay_detects_positive_delay(self):
        rng = np.random.default_rng(7)
        frames = SAMPLE_RATE * 2
        ref = rng.normal(0.0, 0.1, size=(frames, 1)).astype(np.float32)
        delay = int(0.04 * SAMPLE_RATE)
        mic = delayed_copy(ref, delay, frames) * 0.6

        best_delay, corr = estimate_block_delay(
            mic, ref, lo_frame=-1600, hi_frame=8000
        )

        self.assertAlmostEqual(best_delay, delay, delta=2)
        self.assertGreater(corr, 0.5)

    def test_estimate_block_delay_detects_negative_delay(self):
        rng = np.random.default_rng(8)
        frames = SAMPLE_RATE * 2
        ref = rng.normal(0.0, 0.1, size=(frames, 1)).astype(np.float32)
        delay = -int(0.01 * SAMPLE_RATE)
        mic = delayed_copy(ref, delay, frames) * 0.6

        best_delay, corr = estimate_block_delay(
            mic, ref, lo_frame=-1600, hi_frame=8000
        )

        self.assertAlmostEqual(best_delay, delay, delta=2)
        self.assertGreater(corr, 0.5)


class EchoSuppressionTests(unittest.TestCase):
    def test_suppresses_constant_delay_leak(self):
        rng = np.random.default_rng(123)
        frames = SAMPLE_RATE * 20
        reference = rng.normal(0.0, 0.08, size=(frames, 1)).astype(np.float32)
        local_voice = np.zeros((frames, 1), dtype=np.float32)
        local_voice[SAMPLE_RATE * 2:SAMPLE_RATE * 4] = 0.03
        delay = int(0.05 * SAMPLE_RATE)
        mic = local_voice + delayed_copy(reference, delay, frames) * 0.45

        cleaned, stats = suppress_reference_echo(
            mic, reference, SAMPLE_RATE,
            EchoSuppressionConfig(enabled=True),
        )

        before = float(np.sqrt(np.mean((mic - local_voice) ** 2)))
        after = float(np.sqrt(np.mean((cleaned - local_voice) ** 2)))
        self.assertTrue(stats["applied"])
        self.assertGreaterEqual(stats["blocks_suppressed"], 1)
        self.assertLess(after, before * 0.5)

    def test_tracks_drifting_delay_across_blocks(self):
        # 两段不同延迟,模拟时钟漂移。单一全局延迟无法同时压制两段。
        rng = np.random.default_rng(321)
        seg = SAMPLE_RATE * 18
        frames = seg * 2
        reference = rng.normal(0.0, 0.08, size=(frames, 1)).astype(np.float32)
        d1 = int(0.03 * SAMPLE_RATE)
        d2 = int(0.09 * SAMPLE_RATE)
        leak = np.zeros((frames, 1), dtype=np.float32)
        leak[d1:seg] = reference[:seg - d1] * 0.5
        leak[seg + d2:] = reference[seg:frames - d2] * 0.5
        mic = leak.copy()

        cleaned, stats = suppress_reference_echo(
            mic, reference, SAMPLE_RATE,
            EchoSuppressionConfig(enabled=True, block_seconds=15.0),
        )

        first_before = float(np.sqrt(np.mean(mic[:seg] ** 2)))
        first_after = float(np.sqrt(np.mean(cleaned[:seg] ** 2)))
        second_before = float(np.sqrt(np.mean(mic[seg:] ** 2)))
        second_after = float(np.sqrt(np.mean(cleaned[seg:] ** 2)))
        self.assertLess(first_after, first_before * 0.5)
        self.assertLess(second_after, second_before * 0.5)
        self.assertGreaterEqual(stats["blocks_total"], 2)

    def test_skips_silent_reference(self):
        mic = np.full((SAMPLE_RATE, 1), 0.05, dtype=np.float32)
        reference = np.zeros((SAMPLE_RATE, 1), dtype=np.float32)
        cleaned, stats = suppress_reference_echo(
            mic, reference, SAMPLE_RATE, EchoSuppressionConfig(enabled=True)
        )
        np.testing.assert_allclose(cleaned, mic)
        self.assertFalse(stats["applied"])
        self.assertEqual(stats["reason"], "silent_reference")

    def test_disabled_returns_copy(self):
        mic = np.full((SAMPLE_RATE, 1), 0.05, dtype=np.float32)
        reference = np.full((SAMPLE_RATE, 1), 0.05, dtype=np.float32)
        cleaned, stats = suppress_reference_echo(
            mic, reference, SAMPLE_RATE, EchoSuppressionConfig(enabled=False)
        )
        np.testing.assert_allclose(cleaned, mic)
        self.assertFalse(stats["applied"])
        self.assertEqual(stats["reason"], "disabled")


class SourceLevelingTests(unittest.TestCase):
    def test_skips_all_silent(self):
        data = np.zeros((SAMPLE_RATE, 1), dtype=np.float32)
        mask = np.zeros(SAMPLE_RATE, dtype=bool)
        leveled, stats = level_active_source(data, mask, SourceLevelingConfig(enabled=True))
        np.testing.assert_allclose(leveled, data)
        self.assertFalse(stats["applied"])
        self.assertEqual(stats["reason"], "insufficient_activity")

    def test_raises_quiet_voice_with_cap(self):
        data = np.zeros((SAMPLE_RATE * 2, 1), dtype=np.float32)
        data[SAMPLE_RATE:SAMPLE_RATE * 2] = 0.02
        mask = np.zeros(SAMPLE_RATE * 2, dtype=bool)
        mask[SAMPLE_RATE:SAMPLE_RATE * 2] = True
        leveled, stats = level_active_source(
            data, mask,
            SourceLevelingConfig(
                enabled=True, target_level=0.12, max_gain=4.0,
                active_floor=0.001, min_active_seconds=0.5,
            ),
        )
        self.assertTrue(stats["applied"])
        self.assertAlmostEqual(stats["gain"], 4.0, places=5)
        self.assertAlmostEqual(float(np.median(leveled[SAMPLE_RATE:, 0])), 0.08, places=5)

    def test_apply_limiter_clips(self):
        data = np.array([[-2.0], [-0.5], [0.5], [2.0]], dtype=np.float32)
        limited = apply_limiter(data, limit=0.98)
        self.assertEqual(limited[:, 0].tolist(), [-0.98, -0.5, 0.5, 0.98])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 跑测试,确认失败**

```powershell
python -m unittest tests.test_audio_processing
```

预期:`ModuleNotFoundError: No module named 'audio_processing'`。

- [ ] **Step 3: 创建 `audio_processing.py`**

```python
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EchoSuppressionConfig:
    enabled: bool = False
    block_seconds: float = 15.0
    overlap_seconds: float = 0.15
    min_delay_ms: int = -100
    max_delay_ms: int = 500
    max_delay_jump_ms: int = 30
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
    """把 reference 放到长度 target_frames 的零数组里,使 out[i] = reference[i - delay]."""
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
    """返回 length 帧,out[i] = reference[start + i - delay],越界补 0。用于逐块精确减法。"""
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
    """用 FFT 互相关在 [lo_frame, hi_frame] 内找最佳延迟。
    约定:mic[m] ~= gain * ref[m - delay]. 返回 (best_delay, normalized_corr)."""
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

    # 整体静参考判定
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
        ref_block = align_reference_to_target(reference, length, 0)  # same region, delay 0
        # 用全局 reference 的同区间(对齐 delay=0 即 reference[start:end])
        ref_same = _take_aligned_global(reference, start, length, 0)

        if prev_delay is None:
            search_lo, search_hi = lo, hi
        else:
            search_lo = max(lo, prev_delay - jump)
            search_hi = min(hi, prev_delay + jump)
        delay, corr = estimate_block_delay(mic_block, ref_same, search_lo, search_hi)

        ref_energy = float(np.sum(_take_aligned_global(reference, start, length, delay) ** 2))
        win = _trapezoid_window(length, overlap)

        blocks_total += 1
        if corr < config.min_correlation or ref_energy <= 1e-9:
            cleaned_block = mic_block
            if prev_delay is not None:
                delay = prev_delay
        else:
            aligned = _take_aligned_global(reference, start, length, delay)
            num = 0.0
            den = 0.0
            for ch in range(channels):
                ref_ch = aligned[:, min(ch, aligned.shape[1] - 1)]
                num += float(np.dot(mic_block[:, ch], ref_ch))
                den += float(np.dot(ref_ch, ref_ch))
            gain = 0.0 if den <= 1e-12 else float(np.clip(num / den, 0.0, config.max_echo_gain))
            cleaned_block = mic_block - aligned * gain * float(config.suppression_strength)
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
```

- [ ] **Step 4: 跑测试**

```powershell
python -m unittest tests.test_audio_processing
```

预期:全部通过。若 `test_tracks_drifting_delay_across_blocks` 偶发不稳,确认 `block_seconds=15`
使两段(各 18s)落入不同块。

- [ ] **Step 5: 提交**

```powershell
git add audio_processing.py tests/test_audio_processing.py
git commit -m "Add blocked reference echo suppression and source leveling DSP"
```

---

### Task 2: RNNoise 降噪封装 + 重采样 + 降级

**Files:**
- 新建 `denoise.py`
- 新建 `tests/test_denoise.py`

- [ ] **Step 1: 写失败测试(用假后端,不依赖真实 DLL)**

创建 `tests/test_denoise.py`:

```python
import unittest

import numpy as np

import denoise
from denoise import NoiseReductionConfig, reduce_noise


class _FakeBackend:
    """假 RNNoise 后端:把 48k 单声道帧整体衰减到 0.5,用于验证 wiring 与重采样往返。"""
    samplerate = 48000
    frame_size = 480

    def process(self, mono_48k_pm1):
        return (np.asarray(mono_48k_pm1, dtype=np.float32) * 0.5)


class DenoiseTests(unittest.TestCase):
    def setUp(self):
        self._orig = denoise._BACKEND
        denoise._BACKEND = _FakeBackend()
        self.addCleanup(lambda: setattr(denoise, "_BACKEND", self._orig))

    def test_disabled_returns_input(self):
        data = np.full((16000, 1), 0.2, dtype=np.float32)
        out, stats = reduce_noise(data, 16000, NoiseReductionConfig(enabled=False))
        np.testing.assert_allclose(out, data)
        self.assertFalse(stats["applied"])
        self.assertEqual(stats["reason"], "disabled")

    def test_backend_unavailable_degrades(self):
        denoise._BACKEND = None
        data = np.full((16000, 1), 0.2, dtype=np.float32)
        out, stats = reduce_noise(data, 16000, NoiseReductionConfig(enabled=True))
        np.testing.assert_allclose(out, data)
        self.assertFalse(stats["applied"])
        self.assertEqual(stats["reason"], "backend_unavailable")

    def test_applies_and_preserves_length_16k(self):
        data = np.full((16000, 1), 0.2, dtype=np.float32)
        out, stats = reduce_noise(data, 16000, NoiseReductionConfig(enabled=True, mix=1.0))
        self.assertTrue(stats["applied"])
        self.assertEqual(out.shape, data.shape)
        # 全湿:整体被衰减(经过重采样往返,数值近似 0.1)
        self.assertLess(float(np.mean(np.abs(out))), 0.15)

    def test_mix_blends_dry_wet(self):
        data = np.full((16000, 1), 0.2, dtype=np.float32)
        out, stats = reduce_noise(data, 16000, NoiseReductionConfig(enabled=True, mix=0.0))
        # 全干:近似等于输入
        np.testing.assert_allclose(out, data, atol=0.02)
        self.assertTrue(stats["applied"])

    def test_passthrough_for_48k_no_resample(self):
        data = np.full((48000, 1), 0.2, dtype=np.float32)
        out, stats = reduce_noise(data, 48000, NoiseReductionConfig(enabled=True, mix=1.0))
        self.assertEqual(out.shape, data.shape)
        self.assertTrue(stats["applied"])
        self.assertEqual(stats["samplerate"], 48000)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 跑测试,确认失败**

```powershell
python -m unittest tests.test_denoise
```

预期:`ModuleNotFoundError: No module named 'denoise'`。

- [ ] **Step 3: 创建 `denoise.py`**

```python
import ctypes
import os
import sys
from dataclasses import dataclass

import numpy as np

try:
    from scipy.signal import resample_poly
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

RNNOISE_SAMPLERATE = 48000
RNNOISE_FRAME = 480


@dataclass(frozen=True)
class NoiseReductionConfig:
    enabled: bool = False
    mix: float = 1.0  # 0.0=全干(原声), 1.0=全湿(降噪)


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

    def process(self, mono_48k_pm1):
        """输入/输出:48k 单声道 float,范围 [-1,1]。内部按 RNNoise 的 int16 标度处理。"""
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


def _load_backend():
    for path in _candidate_dll_paths():
        if os.path.exists(path):
            try:
                return _RNNoiseBackend(ctypes.CDLL(path))
            except Exception as e:
                print(f"Failed to load RNNoise from {path}: {e}")
    return None


# 模块级单例;测试可 monkeypatch。加载失败为 None → 走降级路径。
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
    out = np.empty_like(audio)
    try:
        for ch in range(audio.shape[1]):
            mono = audio[:, ch]
            up = _resample(mono, samplerate, target_sr)
            wet_up = _BACKEND.process(up)
            wet = _resample(wet_up, target_sr, samplerate)
            # 重采样往返长度可能差 1~2 帧,对齐到原长
            if len(wet) < len(mono):
                wet = np.concatenate([wet, np.zeros(len(mono) - len(wet), dtype=np.float32)])
            else:
                wet = wet[:len(mono)]
            out[:, ch] = (1.0 - mix) * mono + mix * wet
    except Exception as e:
        print(f"Noise reduction failed: {e}")
        return audio.copy(), {"applied": False, "reason": "error"}

    return out, {
        "applied": True, "reason": "applied",
        "mix": round(mix, 3), "channels": int(audio.shape[1]),
        "samplerate": int(samplerate),
    }
```

> 测试里 `_FakeBackend.samplerate == 48000`,所以 16k 测试会触发重采样(需要 scipy);48k 测试不触发。
> 若执行环境没装 scipy,`test_applies_*` 会因 `_resample` 抛错走 `reason="error"` —— 这正是 Task 7
> 要装 scipy 的原因。先在 Task 7 之前确保 `pip install scipy`,否则 16k 降噪测试无法验证。

- [ ] **Step 4: 安装 scipy 并跑测试**

```powershell
python -m pip install scipy
python -m unittest tests.test_denoise
```

预期:全部通过。

- [ ] **Step 5: 提交**

```powershell
git add denoise.py tests/test_denoise.py
git commit -m "Add RNNoise-backed mic noise reduction with graceful fallback"
```

---

### Task 3: 把 AEC / 降噪 / 调平接进 AudioRecorder

**Files:**
- 修改 `audio_recorder.py`
- 修改 `tests/test_audio_output_profile.py`(新增集成测试)

- [ ] **Step 1: 写失败的集成测试**

在 `tests/test_audio_output_profile.py` 的 `_make_recorder()` 之前追加:

```python
    def test_both_mode_suppresses_echo_then_levels(self):
        import os
        import tempfile

        import numpy as np
        import soundfile as sf

        with tempfile.TemporaryDirectory() as temp_dir:
            sr = 16000
            frames = sr * 20
            rng = np.random.default_rng(99)
            loopback = rng.normal(0.0, 0.05, size=(frames, 1)).astype(np.float32)
            voice = np.zeros((frames, 1), dtype=np.float32)
            voice[sr * 2:sr * 4] = 0.02
            delay = int(0.05 * sr)
            leak = np.zeros_like(loopback)
            leak[delay:] = loopback[:-delay] * 0.5
            mic = voice + leak

            mic_file = os.path.join(temp_dir, "mic.wav")
            loop_file = os.path.join(temp_dir, "loop.wav")
            sf.write(mic_file, mic, sr, format="WAV", subtype="FLOAT")
            sf.write(loop_file, loopback, sr, format="WAV", subtype="FLOAT")

            recorder = self._make_recorder("wav", "balanced", stereo=False)
            recorder.source_mode = "both"
            recorder.normalize = True
            recorder.echo_suppression = True
            recorder.noise_reduction = False
            recorder.temp_files = [mic_file, loop_file]

            mixed_file = recorder._prepare_source_wav("FLOAT")
            mixed, _ = sf.read(mixed_file, always_2d=True)

            self.assertTrue(recorder.echo_suppression_applied)
            self.assertLessEqual(np.max(np.abs(mixed)), 0.981)

    def test_both_mode_ignores_echo_for_single_source(self):
        import os
        import tempfile

        import numpy as np
        import soundfile as sf

        with tempfile.TemporaryDirectory() as temp_dir:
            sr = 16000
            mic_file = os.path.join(temp_dir, "mic.wav")
            sf.write(mic_file, np.full((sr, 1), 0.02, dtype=np.float32), sr,
                     format="WAV", subtype="FLOAT")

            recorder = self._make_recorder("wav", "balanced", stereo=False)
            recorder.source_mode = "mic"
            recorder.normalize = False
            recorder.echo_suppression = True
            recorder.noise_reduction = False
            recorder.temp_files = [mic_file]

            prepared = recorder._prepare_source_wav("FLOAT")

            self.assertEqual(prepared, mic_file)
            self.assertFalse(recorder.echo_suppression_applied)
            self.assertEqual(recorder.echo_suppression_reason, "single_source")
```

- [ ] **Step 2: 跑测试,确认失败**

```powershell
python -m unittest tests.test_audio_output_profile.OutputProfileTests.test_both_mode_suppresses_echo_then_levels tests.test_audio_output_profile.OutputProfileTests.test_both_mode_ignores_echo_for_single_source
```

预期:`AttributeError`(缺 `echo_suppression` 等属性)。

- [ ] **Step 3: 加 import 与常量**

在 `from app_metadata import RECORDING_FILENAME_PREFIX` 之后:

```python
from audio_processing import (
    EchoSuppressionConfig,
    SourceLevelingConfig,
    level_active_source,
    suppress_reference_echo,
)
import denoise
from denoise import NoiseReductionConfig
```

在 normalize 常量(`NORMALIZE_LIMIT = 0.98`)之后:

```python
ECHO_SUPPRESSION_DEFAULT_ENABLED = False
NOISE_REDUCTION_DEFAULT_ENABLED = False
NOISE_REDUCTION_MIX = 1.0
SOURCE_LEVELING_MIN_ACTIVE_SECONDS = 0.5
```

- [ ] **Step 4: 加两个 bool 归一化助手**

在 `normalize_trim_silence_enabled()` 之后:

```python
def _normalize_bool_setting(value, default):
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("true", "1", "yes", "on"):
            return True
        if normalized in ("false", "0", "no", "off", ""):
            return False
    return bool(value)


def normalize_echo_suppression_enabled(value):
    return _normalize_bool_setting(value, ECHO_SUPPRESSION_DEFAULT_ENABLED)


def normalize_noise_reduction_enabled(value):
    return _normalize_bool_setting(value, NOISE_REDUCTION_DEFAULT_ENABLED)
```

- [ ] **Step 5: 扩展 `AudioRecorder.__init__()`**

签名里 `normalize=False,` 之后插入:

```python
        echo_suppression=ECHO_SUPPRESSION_DEFAULT_ENABLED,
        noise_reduction=NOISE_REDUCTION_DEFAULT_ENABLED,
```

`self.normalize = normalize` 之后:

```python
        self.echo_suppression = normalize_echo_suppression_enabled(echo_suppression)
        self.noise_reduction = normalize_noise_reduction_enabled(noise_reduction)
```

在 trim 元数据字段(`self.trim_silence_removed_seconds = 0.0`)之后、`self.finish_metadata = ...` 之前:

```python
        self.echo_suppression_applied = False
        self.echo_suppression_reason = "not_run"
        self.echo_suppression_stats = {}
        self.noise_reduction_applied = False
        self.noise_reduction_reason = "not_run"
        self.source_leveling_stats = {}
```

- [ ] **Step 6: 元数据**

在 `build_finish_metadata()` 的返回字典里追加:

```python
            "echo_suppression_enabled": self.echo_suppression,
            "echo_suppression_applied": self.echo_suppression_applied,
            "echo_suppression_reason": self.echo_suppression_reason,
            "echo_suppression_stats": self.echo_suppression_stats,
            "noise_reduction_enabled": self.noise_reduction,
            "noise_reduction_applied": self.noise_reduction_applied,
            "noise_reduction_reason": self.noise_reduction_reason,
            "noise_reduction_available": denoise.is_available(),
            "source_leveling": self.source_leveling_stats,
```

- [ ] **Step 7: 加助手方法(放在 `_prepare_source_wav()` 之前)**

```python
    def _build_leveling_mask(self, data):
        audio = _as_2d_audio(data)
        return np.any(np.abs(audio) >= NORMALIZE_ACTIVE_FLOOR, axis=1)

    def _level_source_data(self, data, samplerate, source_name):
        if not self.normalize:
            self.source_leveling_stats[source_name] = {"applied": False, "reason": "disabled", "gain": 1.0}
            return _as_2d_audio(data).astype(np.float32, copy=False)
        mask = self._build_leveling_mask(data)
        leveled, stats = level_active_source(
            data, mask,
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
        return leveled

    def _denoise_mic_data(self, data, samplerate):
        if not self.noise_reduction:
            self.noise_reduction_applied = False
            self.noise_reduction_reason = "disabled"
            return _as_2d_audio(data).astype(np.float32, copy=False)
        cleaned, stats = denoise.reduce_noise(
            data, samplerate, NoiseReductionConfig(enabled=True, mix=NOISE_REDUCTION_MIX)
        )
        self.noise_reduction_applied = bool(stats.get("applied"))
        self.noise_reduction_reason = str(stats.get("reason", "unknown"))
        return cleaned

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
        mixed = np.clip(mixed, -(NORMALIZE_LIMIT if limit_output else 1.0),
                        (NORMALIZE_LIMIT if limit_output else 1.0))
        sf.write(out_file, mixed, samplerate, format="WAV", subtype=subtype)

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
                mic_data, loopback_data, mic_sr,
                EchoSuppressionConfig(enabled=True, analysis_floor=NORMALIZE_ACTIVE_FLOOR),
            )
            self.echo_suppression_applied = bool(echo_stats.get("applied"))
            self.echo_suppression_reason = str(echo_stats.get("reason", "unknown"))
            self.echo_suppression_stats = echo_stats
        else:
            self.echo_suppression_applied = False
            self.echo_suppression_reason = "disabled"
            self.echo_suppression_stats = {}

        mic_data = self._denoise_mic_data(mic_data, mic_sr)

        mic_data = self._level_source_data(mic_data, mic_sr, "mic")
        loopback_data = self._level_source_data(loopback_data, loopback_sr, "loopback")

        mixed_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        self._mix_audio_data(mic_data, loopback_data, mixed_wav, mic_sr, subtype, limit_output=True)
        self.temp_files.append(mixed_wav)
        return mixed_wav

    def _maybe_trim_final_wav(self, source_wav):
        if not self.trim_silence:
            self.trim_silence_applied = False
            self.trim_silence_removed_seconds = 0.0
            return source_wav
        try:
            info = sf.info(source_wav)
            data, samplerate = sf.read(source_wav, always_2d=True)
            trim_source_name = "loopback" if self.source_mode == "loopback" else "mic"
            trimmed, stats = trim_edge_silence_data(data, samplerate, source_name=trim_source_name)
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

- [ ] **Step 8: 重写 `_prepare_source_wav()`**

```python
    def _prepare_source_wav(self, subtype):
        if len(self.temp_files) == 2:
            mixed = self._prepare_both_source_wav(subtype)
            return self._maybe_trim_final_wav(mixed)

        self.echo_suppression_applied = False
        self.echo_suppression_reason = "single_source"
        self.echo_suppression_stats = {}
        self.source_leveling_stats = {}

        source_wav = self.temp_files[0]
        if self.source_mode == "mic" and self.noise_reduction:
            data, sr = sf.read(source_wav, always_2d=True)
            info = sf.info(source_wav)
            cleaned = self._denoise_mic_data(data, sr)
            denoised_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
            sf.write(denoised_wav, cleaned, sr, format=info.format, subtype=info.subtype)
            self.temp_files.append(denoised_wav)
            source_wav = denoised_wav
        else:
            self.noise_reduction_applied = False
            self.noise_reduction_reason = "disabled" if not self.noise_reduction else "not_mic_source"

        if self.normalize:
            self._normalize_audio(source_wav)
        return self._maybe_trim_final_wav(source_wav)
```

> 这一步刻意移除了旧 `_prepare_source_wav()` 开头对 `_maybe_trim_temp_sources()` 的调用,改为
> 最终输出裁剪。**不要删** `_maybe_trim_temp_sources()` 方法本身,留作未引用代码,Task 8 全绿后
> 再清理。同时保留旧的 `_mix_audio()`/`_normalize_audio()`/`_normalize_audio_data()`/`_apply_limiter()`
> (单源路径与既有测试仍依赖它们)。

- [ ] **Step 9: 跑集成测试**

```powershell
python -m unittest tests.test_audio_processing tests.test_audio_output_profile
```

预期:Task 1 测试 + 两个新集成测试通过。`test_audio_trim` 可能仍失败(Task 4 修)。

- [ ] **Step 10: 提交**

```powershell
git add audio_recorder.py audio_processing.py denoise.py tests/test_audio_output_profile.py
git commit -m "Wire echo suppression, denoise and source leveling into recorder"
```

---

### Task 4: 修复受影响的现有测试

**Files:**
- 修改 `tests/test_audio_trim.py`
- 检查 `tests/test_audio_output_profile.py`

- [ ] **Step 1: 确认 `test_prepare_source_wav_normalizes_both_sources_before_mixing` 仍通过**

该测试用 mic=0.02、loopback=0.01 的定常信号,断言混音后 `median(|body|) > 0.15`。
采用**能量地板门控**后:0.02 与 0.01 均 ≥ 地板 0.001 → 掩码几乎全 True → 麦克风提升至约 0.12、
系统声提升至约 0.08(受 max_gain=8 限制)→ 混音约 0.20 > 0.15。**因此该测试无需改动即可通过。**
跑一遍确认:

```powershell
python -m unittest tests.test_audio_output_profile.OutputProfileTests.test_prepare_source_wav_normalizes_both_sources_before_mixing
```

若失败,检查 `_build_leveling_mask` 是否用 `NORMALIZE_ACTIVE_FLOOR`(而非活动检测器)。

- [ ] **Step 2: 把 Both 模式裁剪测试改为"混音后裁剪"**

旧方案的错误是替换该测试时把采样率/时长换成 1000Hz/10s 却仍断言 16k 的帧数常量,导致不可能通过。
**正确做法是复用既有 16k 数据与 `INTEGRATION_TRIMMED_FRAMES`**:由于 loopback 有声、mic 全静音,
混音结果等于 loopback,对混音单轨裁剪得到的边界与旧"双轨合并裁剪"一致,帧数仍是 205056。

在 `tests/test_audio_trim.py` 中,把 `test_prepare_source_wav_applies_shared_bounds_before_both_mix`
重命名为 `test_prepare_source_wav_trims_after_both_mix`,函数体保持原数据与断言不变,仅更新方法名与注释
(说明裁剪现在作用于最终混音输出):

```python
    def test_prepare_source_wav_trims_after_both_mix(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            mic_wav = os.path.join(temp_dir, "mic.wav")
            loopback_wav = os.path.join(temp_dir, "loopback.wav")
            mic = integration_silence(17)
            loopback = np.concatenate(
                [integration_silence(7), integration_tone(2), integration_silence(8)],
                axis=0,
            )
            sf.write(mic_wav, mic, INTEGRATION_SAMPLE_RATE, format="WAV", subtype="FLOAT")
            sf.write(loopback_wav, loopback, INTEGRATION_SAMPLE_RATE, format="WAV", subtype="FLOAT")

            recorder = AudioRecorder(
                mic_id="mic1", source_mode="both", output_folder=temp_dir,
                output_format="wav", quality="balanced", stereo=False,
                normalize=False, trim_silence=True,
            )
            recorder.temp_files = [mic_wav, loopback_wav]

            mixed_wav = recorder._prepare_source_wav("FLOAT")
            mixed, sr = sf.read(mixed_wav, always_2d=True)

            self.assertEqual(sr, INTEGRATION_SAMPLE_RATE)
            self.assertEqual(len(mixed), INTEGRATION_TRIMMED_FRAMES)
            self.assertTrue(recorder.trim_silence_applied)
            self.assertAlmostEqual(
                recorder.trim_silence_removed_seconds, INTEGRATION_REMOVED_SECONDS
            )
```

`test_prepare_source_wav_applies_trim_before_single_source_output` 与
`test_prepare_source_wav_skips_trim_when_disabled` 保持不变(单源路径行为未变,应继续通过)。

- [ ] **Step 3: 跑测试**

```powershell
python -m unittest tests.test_audio_trim tests.test_audio_output_profile
```

预期:全部通过。

- [ ] **Step 4: 提交**

```powershell
git add tests/test_audio_trim.py tests/test_audio_output_profile.py audio_recorder.py
git commit -m "Move silence trim to final output and keep both-mode leveling green"
```

---

### Task 5: GUI 设置

**Files:**
- 修改 `gui.py`
- 修改 `tests/test_gui_hotkeys.py`

- [ ] **Step 1: 写失败的 GUI 测试**

在 `tests/test_gui_hotkeys.py` 的 trim/normalize 设置测试附近追加:

```python
    def test_echo_suppression_setting_defaults_off(self):
        window = self.make_window({})
        self.assertIs(window.get_settings()["echo_suppression"], False)

    def test_echo_suppression_setting_can_be_enabled(self):
        window = self.make_window({"echo_suppression": True})
        self.assertIs(window.get_settings()["echo_suppression"], True)

    def test_noise_reduction_setting_defaults_off(self):
        window = self.make_window({})
        self.assertIs(window.get_settings()["noise_reduction"], False)

    def test_noise_reduction_setting_can_be_enabled(self):
        window = self.make_window({"noise_reduction": True})
        self.assertIs(window.get_settings()["noise_reduction"], True)

    def test_post_processing_group_contains_new_toggles(self):
        window = self.make_window({})
        post_group = window.findChild(QGroupBox, "postProcessingSettingsGroup")
        self.assertIs(window.chk_echo_suppression.parentWidget(), post_group)
        self.assertIs(window.chk_noise_reduction.parentWidget(), post_group)
```

在 `make_subject()` 的 mock 配置字典(`tests/test_gui_hotkeys.py` 约 580 行)里追加两项:

```python
                    "echo_suppression": True,
                    "noise_reduction": True,
```

并新增传参断言测试:

```python
    def test_start_recording_passes_new_audio_settings(self):
        subject, _indicator = self.make_subject(show_indicator=True)
        with patch("gui.QIcon"), patch("gui.AudioRecorder") as AudioRecorder:
            TrayApplication.start_recording(subject, "both")
        self.assertEqual(AudioRecorder.call_args.kwargs["echo_suppression"], True)
        self.assertEqual(AudioRecorder.call_args.kwargs["noise_reduction"], True)
```

- [ ] **Step 2: 跑测试,确认失败**

```powershell
python -m unittest tests.test_gui_hotkeys
```

预期:`AttributeError: 'SettingsWindow' object has no attribute 'chk_echo_suppression'`。

- [ ] **Step 3: 在 `SettingsWindow.init_ui()` 增加复选框**

在 `self.chk_normalize = QCheckBox("Normalize Audio (Apply first)")` 之后:

```python
        self.chk_echo_suppression = QCheckBox("Reduce speaker echo (Both mode)")
        self.chk_echo_suppression.setToolTip(
            "Uses system audio as a reference to reduce speaker leakage from the microphone "
            "track before noise reduction, leveling and mixing. Only affects Both mode."
        )
        self.chk_noise_reduction = QCheckBox("Reduce microphone noise (RNNoise)")
        self.chk_noise_reduction.setToolTip(
            "Applies RNNoise speech denoising to the microphone track after echo suppression "
            "and before leveling. Requires rnnoise.dll; silently skipped if unavailable."
        )
```

在 `layout_post.addWidget(self.chk_normalize)` 之后:

```python
        layout_post.addWidget(self.chk_echo_suppression)
        layout_post.addWidget(self.chk_noise_reduction)
```

- [ ] **Step 4: 读取与保存**

在 `load_settings()` 里 normalize 加载之后:

```python
        self.chk_echo_suppression.setChecked(self._parse_bool_setting(data.get("echo_suppression")))
        self.chk_noise_reduction.setChecked(self._parse_bool_setting(data.get("noise_reduction")))
```

在 `get_settings()` 里 `"normalize": self.chk_normalize.isChecked(),` 之后:

```python
            "echo_suppression": self.chk_echo_suppression.isChecked(),
            "noise_reduction": self.chk_noise_reduction.isChecked(),
```

- [ ] **Step 5: 传入 `AudioRecorder`**

在 `TrayApplication.start_recording()` 里 `normalize=settings['normalize'],` 之后:

```python
            echo_suppression=settings.get("echo_suppression", False),
            noise_reduction=settings.get("noise_reduction", False),
```

- [ ] **Step 6: 跑测试**

```powershell
python -m unittest tests.test_gui_hotkeys
```

预期:全部通过。

- [ ] **Step 7: 提交**

```powershell
git add gui.py tests/test_gui_hotkeys.py audio_recorder.py
git commit -m "Add echo suppression and noise reduction settings to GUI"
```

---

### Task 6: README 与文案

**Files:**
- 修改 `README.md`

- [ ] **Step 1: 替换后处理特性条目**

把 `README.md` 第 45-46 行的两条:

```markdown
- **Normalize Audio (Apply first)**: raises the main voice/body of each source before mixing and limits sharp peaks.
- **Silence trim**: optional post-processing that trims only the start and end silence beyond 5 seconds, using the same mic/loopback activity detection rules as silence auto-stop.
```

替换为:

```markdown
- **Reduce speaker echo (Both mode)**: uses system audio as a reference to subtract speaker leakage from the microphone track. Processed in 15-second blocks with per-block delay tracking, so it follows clock drift between the two capture streams.
- **Reduce microphone noise (RNNoise)**: applies RNNoise speech denoising to the microphone track after echo suppression and before leveling. Requires `rnnoise.dll`; if the library is unavailable, recording still succeeds and denoising is skipped.
- **Normalize Audio**: in Both mode this performs per-source leveling (microphone and system audio are each raised toward a target level using an energy floor, so quiet microphone speech is not buried by louder system audio); single-source recordings keep whole-track normalization.
- **Silence trim**: optional final post-processing that trims only the start and end silence beyond 5 seconds, applied to the final mixed output rather than the raw sources.
```

- [ ] **Step 2: 替换使用步骤 8**

把第 59 行:

```markdown
8. In **Post-Processing & Clipboard**, enable normalization, edge-silence trim, clipboard copy, or delete-after-copy as needed.
```

替换为:

```markdown
8. In **Post-Processing & Clipboard**, enable echo reduction, microphone noise reduction, source leveling/normalization, final edge-silence trim, clipboard copy, or delete-after-copy as needed.
```

- [ ] **Step 3: 跑产品文案测试**

```powershell
python -m unittest tests.test_product_identity_text
```

预期:通过(若该测试对具体文案有断言,按其要求微调措辞)。

- [ ] **Step 4: 提交**

```powershell
git add README.md
git commit -m "Document echo suppression, RNNoise denoise and leveling order"
```

---

### Task 7: 依赖与打包(scipy + rnnoise.dll)

**Files:**
- 新建 `requirements.txt`
- 修改 `MeetRec.spec`

- [ ] **Step 1: 创建 `requirements.txt`**

记录已知运行时依赖(版本以当前 .venv 实际为准,可用 `python -m pip freeze` 核对):

```
soundcard
soundfile
numpy
scipy
lameenc
PyQt6
keyboard
pyinstaller
```

- [ ] **Step 2: 获取 `rnnoise.dll`**

RNNoise 无官方 Windows 二进制,需自行获取其一:

- **从源码编译(推荐可控)**:克隆 `https://github.com/xiph/rnnoise`,在 MSYS2/MinGW64 下
  `./autogen.sh && ./configure && make`,产物 `.libs/librnnoise-0.dll`(或用 CMake 分支)。
- 或使用可信的预编译 `rnnoise.dll`。

把 DLL 放到仓库根目录并命名为 `rnnoise.dll`(`denoise.py` 的 `_candidate_dll_paths` 会在根目录、
exe 目录、`_MEIPASS`、`RNNOISE_DLL` 环境变量指向处查找)。

> 验证后端可用:
> ```powershell
> python -c "import denoise; print('available:', denoise.is_available())"
> ```
> 若打印 `False`,降噪会走降级路径(录音正常、不降噪)。这是预期的安全行为,不是阻塞项。

- [ ] **Step 3: 把 DLL 收进 PyInstaller**

修改 `MeetRec.spec` 的 `binaries=[]`:

```python
import os as _os
_rnnoise_dll = _os.path.join(_os.getcwd(), "rnnoise.dll")
_binaries = [(_rnnoise_dll, ".")] if _os.path.exists(_rnnoise_dll) else []
```

并在 `Analysis(...)` 里把 `binaries=[],` 改为 `binaries=_binaries,`。
若需要 scipy 的隐藏依赖,PyInstaller 通常自动处理;如打包后报缺失,在 `hiddenimports` 里补
`scipy.signal`。

- [ ] **Step 4: 提交**

```powershell
git add requirements.txt MeetRec.spec
git commit -m "Bundle RNNoise DLL and record dependencies for packaging"
```

> 不要把 `rnnoise.dll` 二进制提交进版本库,除非团队约定如此;否则在 `.gitignore` 里忽略它,
> 并在 README/构建说明里写明获取步骤。

---

### Task 8: 全量回归 + 手动冒烟

**Files:**
- 无代码改动,除非测试暴露缺陷。

- [ ] **Step 1: 全量单元测试**

```powershell
python -m unittest discover -s tests
```

预期:全部通过。

- [ ] **Step 2: 性能抽查(确认分块 AEC 在长录音上够快)**

```powershell
$code = @'
import time
import numpy as np
from audio_processing import EchoSuppressionConfig, suppress_reference_echo

sr = 16000
frames = sr * 600  # 10 minutes
rng = np.random.default_rng(0)
ref = rng.normal(0, 0.08, size=(frames, 1)).astype("float32")
leak = np.zeros_like(ref); d = int(0.05 * sr); leak[d:] = ref[:-d] * 0.5
mic = leak + rng.normal(0, 0.01, size=(frames, 1)).astype("float32")

t0 = time.time()
cleaned, stats = suppress_reference_echo(mic, ref, sr, EchoSuppressionConfig(enabled=True))
print("elapsed_s", round(time.time() - t0, 2))
print(stats)
'@
$code | python -
```

预期:`elapsed_s` 在数秒以内(远非分钟级),`blocks_total` ≈ 40,`applied` 为 True。

- [ ] **Step 3: 打包**

```powershell
pyinstaller --noconfirm MeetRec.spec
```

预期:`dist\MeetRec\MeetRec.exe` 生成,退出码 0;若 `rnnoise.dll` 已放置,它出现在 `dist\MeetRec\` 下。

- [ ] **Step 4: 手动冒烟**

1. 启动 `dist\MeetRec\MeetRec.exe`,打开设置。
2. 勾选 `Reduce speaker echo (Both mode)`、`Reduce microphone noise (RNNoise)`、`Normalize Audio`、
   `Trim start/end silence over 5s`。
3. 用 Both 模式录一段:扬声器放音(能被麦克风听到)+ 自己小声说话。
4. 确认:文件成功保存且可播放;麦克风语音清晰、未被系统声盖住;扬声器回声相比关闭时明显减弱;
   底噪相比关闭降噪时明显降低且没有明显"吞字";起止裁剪只影响最终输出。
5. 关掉降噪重录一次对比,确认降噪确实在压底噪而非损伤人声(若吞字,降低 `NOISE_REDUCTION_MIX` 再试)。

- [ ] **Step 5: 如有修复则提交**

```powershell
git add audio_processing.py denoise.py audio_recorder.py gui.py README.md tests
git commit -m "Fix echo/denoise/leveling regressions found in regression pass"
```

无需修复则不要创建空提交。

---

## 自检

- 顺序覆盖:raw → AEC → (mic)降噪 → 分轨调平 → 混音 → 限幅 → 最终裁剪 → 导出,与处理规则一致。
- 性能:AEC 改为 15s 分块 + FFT 互相关 + 延迟跟踪,10 分钟录音从"分钟级/近卡死"降到数秒(Task 8 Step 2 抽查)。
- 漂移:逐块延迟跟踪 + 跳变限幅吸收时钟漂移(`test_tracks_drifting_delay_across_blocks` 验证两段不同延迟均被压制)。
- 轻声麦克风:调平改用能量地板门控,-34dB 级语音可被提升;同时跳过纯静音轨。这同时让既有
  `test_prepare_source_wav_normalizes_both_sources_before_mixing` 无需改动即通过(修复旧方案缺陷一)。
- 裁剪测试:复用 16k 数据与 `INTEGRATION_TRIMMED_FRAMES`,断言"混音后裁剪",修复旧方案缺陷二
  (旧版误用 1000Hz/10s 却断言 205056 帧,不可能通过)。
- 降噪稳健:RNNoise 后端隔离、加载失败优雅降级、测试用假后端不依赖 DLL;干湿混合可调温和度。
- 边界状态:静参考、低相关块、单源模式、各开关关闭、降噪库缺失,均有明确处理与元数据。
```
