import unittest

import numpy as np

import denoise
from denoise import NoiseReductionConfig, reduce_noise


class _FakeBackend:
    """Fake RNNoise backend for wiring tests."""
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
        self.assertLess(float(np.mean(np.abs(out))), 0.15)

    def test_mix_blends_dry_wet(self):
        data = np.full((16000, 1), 0.2, dtype=np.float32)
        out, stats = reduce_noise(data, 16000, NoiseReductionConfig(enabled=True, mix=0.0))
        np.testing.assert_allclose(out, data, atol=0.02)
        self.assertTrue(stats["applied"])

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

    def test_passthrough_for_48k_no_resample(self):
        data = np.full((48000, 1), 0.2, dtype=np.float32)
        out, stats = reduce_noise(data, 48000, NoiseReductionConfig(enabled=True, mix=1.0))
        self.assertEqual(out.shape, data.shape)
        self.assertTrue(stats["applied"])
        self.assertEqual(stats["samplerate"], 48000)


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


if __name__ == "__main__":
    unittest.main()
