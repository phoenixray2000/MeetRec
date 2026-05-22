import unittest

import numpy as np

import denoise
from denoise import NoiseReductionConfig, reduce_noise


class _FakeBackend:
    """Fake RNNoise backend: halves 48 kHz mono frames for wiring tests."""
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
        self.assertLess(float(np.mean(np.abs(out))), 0.15)

    def test_mix_blends_dry_wet(self):
        data = np.full((16000, 1), 0.2, dtype=np.float32)
        out, stats = reduce_noise(data, 16000, NoiseReductionConfig(enabled=True, mix=0.0))
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
