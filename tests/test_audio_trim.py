import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from audio_recorder import (
    AudioRecorder,
    SourceActivityDetector,
    calculate_trim_bounds_for_sources,
    trim_edge_silence_data,
)


SAMPLE_RATE = 1000
CHUNK_FRAMES = 100
INTEGRATION_SAMPLE_RATE = 16000
INTEGRATION_TRIMMED_FRAMES = 205056
INTEGRATION_REMOVED_SECONDS = 4.184


def silence(seconds, channels=1):
    return np.zeros((int(seconds * SAMPLE_RATE), channels), dtype=np.float32)


def tone(seconds, channels=1, level=0.2):
    return np.full((int(seconds * SAMPLE_RATE), channels), level, dtype=np.float32)


def integration_silence(seconds, channels=1):
    return np.zeros((int(seconds * INTEGRATION_SAMPLE_RATE), channels), dtype=np.float32)


def integration_tone(seconds, channels=1, level=0.2):
    return np.full(
        (int(seconds * INTEGRATION_SAMPLE_RATE), channels),
        level,
        dtype=np.float32,
    )


class DetectorBasedTrimTests(unittest.TestCase):
    def test_trims_using_source_activity_detector_window(self):
        data = np.concatenate([silence(7), tone(2), silence(8)], axis=0)

        trimmed, stats = trim_edge_silence_data(
            data,
            SAMPLE_RATE,
            source_name="mic",
            chunk_frames=CHUNK_FRAMES,
        )

        self.assertEqual(len(trimmed), 12600)
        self.assertTrue(stats["applied"])
        self.assertAlmostEqual(stats["start_removed_seconds"], 2.2)
        self.assertAlmostEqual(stats["end_removed_seconds"], 2.2)
        self.assertEqual(stats["source_names"], ["mic"])

    def test_short_spike_is_ignored_by_same_active_window_rule(self):
        data = np.concatenate(
            [silence(7), tone(0.1), silence(2), tone(2), silence(7)],
            axis=0,
        )

        trimmed, stats = trim_edge_silence_data(
            data,
            SAMPLE_RATE,
            source_name="mic",
            chunk_frames=CHUNK_FRAMES,
        )

        self.assertEqual(len(trimmed), 12600)
        self.assertTrue(stats["applied"])
        self.assertAlmostEqual(stats["start_removed_seconds"], 4.3)
        self.assertAlmostEqual(stats["end_removed_seconds"], 1.2)

    def test_uses_requested_source_detector(self):
        data = np.concatenate([silence(7), tone(2), silence(8)], axis=0)

        with patch("audio_recorder.build_activity_detector") as build_detector:
            build_detector.return_value = SourceActivityDetector(
                margin_db=12.0,
                min_threshold_db=-60.0,
                max_threshold_db=-28.0,
            )
            trim_edge_silence_data(
                data,
                SAMPLE_RATE,
                source_name="loopback",
                chunk_frames=CHUNK_FRAMES,
            )

        build_detector.assert_called_once_with("loopback")

    def test_combines_both_sources_with_or_activity(self):
        mic = silence(17)
        loopback = np.concatenate([silence(7), tone(2), silence(8)], axis=0)

        start_frame, end_frame, stats = calculate_trim_bounds_for_sources(
            [("mic", mic), ("loopback", loopback)],
            SAMPLE_RATE,
            chunk_frames=CHUNK_FRAMES,
        )

        self.assertEqual(start_frame, 2200)
        self.assertEqual(end_frame, 14800)
        self.assertEqual(end_frame - start_frame, 12600)
        self.assertEqual(stats["source_names"], ["mic", "loopback"])

    def test_all_silent_recording_keeps_five_second_safety_clip(self):
        data = silence(12)

        trimmed, stats = trim_edge_silence_data(
            data,
            SAMPLE_RATE,
            source_name="mic",
            chunk_frames=CHUNK_FRAMES,
        )

        self.assertEqual(len(trimmed), SAMPLE_RATE * 5)
        self.assertTrue(stats["applied"])
        self.assertAlmostEqual(stats["start_removed_seconds"], 0.0)
        self.assertAlmostEqual(stats["end_removed_seconds"], 7.0)


class AudioRecorderTrimIntegrationTests(unittest.TestCase):
    def test_prepare_source_wav_applies_trim_before_single_source_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source_wav = os.path.join(temp_dir, "source.wav")
            data = np.concatenate(
                [integration_silence(7), integration_tone(2), integration_silence(8)],
                axis=0,
            )
            sf.write(source_wav, data, INTEGRATION_SAMPLE_RATE, format="WAV", subtype="FLOAT")

            recorder = AudioRecorder(
                mic_id="mic1",
                source_mode="mic",
                output_folder=temp_dir,
                output_format="wav",
                quality="balanced",
                stereo=False,
                normalize=False,
                trim_silence=True,
            )
            recorder.temp_files = [source_wav]

            prepared_wav = recorder._prepare_source_wav("FLOAT")
            trimmed, sr = sf.read(prepared_wav, always_2d=True)

            self.assertEqual(sr, INTEGRATION_SAMPLE_RATE)
            self.assertEqual(len(trimmed), INTEGRATION_TRIMMED_FRAMES)
            self.assertTrue(recorder.trim_silence_applied)
            self.assertAlmostEqual(
                recorder.trim_silence_removed_seconds,
                INTEGRATION_REMOVED_SECONDS,
            )

    def test_prepare_source_wav_applies_shared_bounds_before_both_mix(self):
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
                mic_id="mic1",
                source_mode="both",
                output_folder=temp_dir,
                output_format="wav",
                quality="balanced",
                stereo=False,
                normalize=False,
                trim_silence=True,
            )
            recorder.temp_files = [mic_wav, loopback_wav]

            mixed_wav = recorder._prepare_source_wav("FLOAT")
            mixed, sr = sf.read(mixed_wav, always_2d=True)

            self.assertEqual(sr, INTEGRATION_SAMPLE_RATE)
            self.assertEqual(len(mixed), INTEGRATION_TRIMMED_FRAMES)
            self.assertTrue(recorder.trim_silence_applied)
            self.assertAlmostEqual(
                recorder.trim_silence_removed_seconds,
                INTEGRATION_REMOVED_SECONDS,
            )

    def test_prepare_source_wav_skips_trim_when_disabled(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source_wav = os.path.join(temp_dir, "source.wav")
            data = np.concatenate(
                [integration_silence(7), integration_tone(2), integration_silence(8)],
                axis=0,
            )
            sf.write(source_wav, data, INTEGRATION_SAMPLE_RATE, format="WAV", subtype="FLOAT")

            recorder = AudioRecorder(
                mic_id="mic1",
                source_mode="mic",
                output_folder=temp_dir,
                output_format="wav",
                quality="balanced",
                stereo=False,
                normalize=False,
                trim_silence=False,
            )
            recorder.temp_files = [source_wav]

            prepared_wav = recorder._prepare_source_wav("FLOAT")
            untrimmed, sr = sf.read(prepared_wav, always_2d=True)

            self.assertEqual(sr, INTEGRATION_SAMPLE_RATE)
            self.assertEqual(len(untrimmed), len(data))
            self.assertFalse(recorder.trim_silence_applied)
            self.assertAlmostEqual(recorder.trim_silence_removed_seconds, 0.0)


class AudioRecorderTrimMetadataTests(unittest.TestCase):
    def test_finish_metadata_records_trim_state(self):
        recorder = AudioRecorder(
            mic_id="mic1",
            source_mode="mic",
            output_folder=".",
            trim_silence=True,
        )
        recorder.trim_silence_applied = True
        recorder.trim_silence_removed_seconds = 4.4

        metadata = recorder.build_finish_metadata()

        self.assertEqual(metadata["trim_silence_enabled"], True)
        self.assertEqual(metadata["trim_silence_keep_seconds"], 5.0)
        self.assertEqual(metadata["trim_silence_applied"], True)
        self.assertEqual(metadata["trim_silence_removed_seconds"], 4.4)


if __name__ == "__main__":
    unittest.main()
