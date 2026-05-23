import unittest

import numpy as np

from audio_processing import (
    DuckingConfig,
    EchoSuppressionConfig,
    SourceLevelingConfig,
    align_reference_to_target,
    apply_limiter,
    duck_reference_audio,
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
        # Two delay regions simulate clock drift. A single global delay cannot suppress both well.
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


if __name__ == "__main__":
    unittest.main()
