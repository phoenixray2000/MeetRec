import unittest

import numpy as np

from audio_recorder import (
    AUTO_STOP_DEFAULT_SECONDS,
    AutoStopController,
    AudioRecorder,
    SourceActivityDetector,
    normalize_auto_stop_silence_seconds,
)


def chunk(level, frames=2048, channels=1):
    return np.full((frames, channels), level, dtype=np.float32)


class SourceActivityDetectorTests(unittest.TestCase):
    def test_silence_below_dynamic_threshold_is_inactive(self):
        detector = SourceActivityDetector(
            margin_db=9.0,
            min_threshold_db=-55.0,
            max_threshold_db=-30.0,
            history_seconds=30.0,
            active_window_seconds=1.0,
            active_ratio=0.2,
        )

        active = detector.update(chunk(0.00001), samplerate=2048, now=0.0)

        self.assertFalse(active)
        self.assertEqual(detector.noise_floor_db, -100.0)
        self.assertEqual(detector.threshold_db, -55.0)

    def test_signal_above_dynamic_threshold_becomes_active(self):
        detector = SourceActivityDetector(
            margin_db=9.0,
            min_threshold_db=-55.0,
            max_threshold_db=-30.0,
            history_seconds=30.0,
            active_window_seconds=1.0,
            active_ratio=0.2,
        )

        detector.update(chunk(0.00001), samplerate=2048, now=0.0)
        active = detector.update(chunk(0.05), samplerate=2048, now=1.0)

        self.assertTrue(active)
        self.assertGreater(detector.current_db, detector.threshold_db)

    def test_isolated_spike_does_not_make_window_active(self):
        detector = SourceActivityDetector(
            margin_db=9.0,
            min_threshold_db=-55.0,
            max_threshold_db=-30.0,
            history_seconds=30.0,
            active_window_seconds=1.0,
            active_ratio=0.2,
        )

        now = 0.0
        for _ in range(9):
            active = detector.update(chunk(0.00001, frames=1600), samplerate=16000, now=now)
            now += 0.1
            self.assertFalse(active)

        active = detector.update(chunk(0.2, frames=1600), samplerate=16000, now=now)

        self.assertFalse(active)


class AutoStopControllerTests(unittest.TestCase):
    def test_normalizes_supported_values(self):
        self.assertIsNone(normalize_auto_stop_silence_seconds(None))
        self.assertIsNone(normalize_auto_stop_silence_seconds("off"))
        self.assertEqual(normalize_auto_stop_silence_seconds("300"), 300)
        self.assertEqual(normalize_auto_stop_silence_seconds(600), 600)
        self.assertEqual(normalize_auto_stop_silence_seconds(1200), 1200)
        self.assertEqual(normalize_auto_stop_silence_seconds("invalid"), AUTO_STOP_DEFAULT_SECONDS)

    def test_disabled_controller_never_stops(self):
        controller = AutoStopController(
            silence_seconds=None,
            min_record_seconds=60.0,
            start_ts=0.0,
        )

        self.assertFalse(controller.update({"mic": False}, now=10000.0))

    def test_respects_min_record_seconds(self):
        controller = AutoStopController(
            silence_seconds=300,
            min_record_seconds=60.0,
            start_ts=0.0,
        )

        self.assertFalse(controller.update({"mic": False}, now=59.0))

    def test_stops_after_single_source_silence_timeout(self):
        controller = AutoStopController(
            silence_seconds=300,
            min_record_seconds=60.0,
            start_ts=0.0,
        )

        self.assertFalse(controller.update({"mic": False}, now=299.0))
        self.assertTrue(controller.update({"mic": False}, now=300.0))
        self.assertEqual(controller.reason(), "silence_timeout_5min")

    def test_any_active_source_resets_silence_timer(self):
        controller = AutoStopController(
            silence_seconds=300,
            min_record_seconds=60.0,
            start_ts=0.0,
        )

        self.assertFalse(controller.update({"mic": True}, now=200.0))
        self.assertFalse(controller.update({"mic": False}, now=499.0))
        self.assertTrue(controller.update({"mic": False}, now=500.0))

    def test_both_mode_only_stops_when_all_sources_are_silent(self):
        controller = AutoStopController(
            silence_seconds=300,
            min_record_seconds=60.0,
            start_ts=0.0,
        )

        self.assertFalse(controller.update({"mic": False, "loopback": True}, now=200.0))
        self.assertFalse(controller.update({"mic": False, "loopback": False}, now=499.0))
        self.assertTrue(controller.update({"mic": False, "loopback": False}, now=500.0))


class AudioRecorderAutoStopTests(unittest.TestCase):
    def test_single_source_activity_can_request_auto_stop(self):
        recorder = AudioRecorder(
            mic_id="mic1",
            source_mode="mic",
            output_folder=".",
            auto_stop_silence_seconds=300,
        )
        recorder._setup_auto_stop()
        recorder.auto_stop_controller = AutoStopController(
            silence_seconds=300,
            min_record_seconds=60.0,
            start_ts=0.0,
        )

        recorder.report_activity("mic", chunk(0.00001), now=300.0)

        self.assertTrue(recorder.stop_event.is_set())
        self.assertTrue(recorder.auto_stop_triggered)
        self.assertEqual(recorder.auto_stop_reason, "silence_timeout_5min")

    def test_both_mode_activity_from_one_source_prevents_auto_stop(self):
        recorder = AudioRecorder(
            mic_id="mic1",
            source_mode="both",
            output_folder=".",
            auto_stop_silence_seconds=300,
        )
        recorder._setup_auto_stop()
        recorder.auto_stop_controller = AutoStopController(
            silence_seconds=300,
            min_record_seconds=60.0,
            start_ts=0.0,
        )

        recorder.source_active["loopback"] = True
        recorder.report_activity("mic", chunk(0.00001), now=300.0)

        self.assertFalse(recorder.stop_event.is_set())
        self.assertFalse(recorder.auto_stop_triggered)

    def test_finish_metadata_records_auto_stop_state(self):
        recorder = AudioRecorder(
            mic_id="mic1",
            source_mode="mic",
            output_folder=".",
            auto_stop_silence_seconds=600,
        )
        recorder.auto_stop_triggered = True
        recorder.auto_stop_reason = "silence_timeout_10min"

        metadata = recorder.build_finish_metadata()

        expected_core = {
            "auto_stop_enabled": True,
            "auto_stop_silence_seconds": 600,
            "auto_stop_triggered": True,
            "auto_stop_reason": "silence_timeout_10min",
            "trim_silence_enabled": False,
            "trim_silence_keep_seconds": 5.0,
            "trim_silence_applied": False,
            "trim_silence_removed_seconds": 0.0,
        }
        self.assertEqual({key: metadata[key] for key in expected_core}, expected_core)
        self.assertEqual(metadata["echo_suppression_enabled"], False)
        self.assertEqual(metadata["echo_suppression_applied"], False)
        self.assertEqual(metadata["echo_suppression_reason"], "not_run")
        self.assertEqual(metadata["echo_suppression_stats"], {})
        self.assertEqual(metadata["noise_reduction_enabled"], False)
        self.assertIn("noise_reduction_available", metadata)
        self.assertEqual(metadata["noise_reduction_applied"], False)
        self.assertEqual(metadata["noise_reduction_reason"], "not_run")
        self.assertEqual(metadata["source_leveling_enabled"], False)
        self.assertEqual(metadata["source_leveling_stats"], {})


if __name__ == "__main__":
    unittest.main()
