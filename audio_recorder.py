import soundcard as sc
import soundfile as sf
import threading
import time
import os
import json
import shutil
import lameenc
import numpy as np
import tempfile
from collections import deque
from app_metadata import RECORDING_FILENAME_PREFIX
from audio_processing import (
    EchoSuppressionConfig,
    SourceLevelingConfig,
    level_active_source,
    suppress_reference_echo,
)
import denoise
from denoise import NoiseReductionConfig

FORMAT_CONFIG = {
    "wav": {
        "label": "WAV",
        "extension": ".wav",
        "encoder": "soundfile",
        "format": "WAV",
    },
    "flac": {
        "label": "FLAC",
        "extension": ".flac",
        "encoder": "soundfile",
        "format": "FLAC",
    },
    "mp3": {
        "label": "MP3",
        "extension": ".mp3",
        "encoder": "lameenc",
    },
}

QUALITY_CONFIG = {
    "balanced": {
        "label": "Balanced",
        "sample_rate": 16000,
        "subtype": "PCM_16",
        "mp3_bitrate_kbps": 64,
    },
    "high": {
        "label": "High Quality",
        "sample_rate": 48000,
        "subtype": "PCM_24",
        "mp3_bitrate_kbps": 128,
    },
}

NORMALIZE_ACTIVE_FLOOR = 0.001
NORMALIZE_TARGET_LEVEL = 0.12
NORMALIZE_MAX_GAIN = 8.0
NORMALIZE_REFERENCE_PERCENTILE = 95
NORMALIZE_LIMIT = 0.98
ECHO_SUPPRESSION_DEFAULT_ENABLED = False
NOISE_REDUCTION_DEFAULT_ENABLED = False
NOISE_REDUCTION_MIX = 0.35
NOISE_REDUCTION_LATENCY_MS = 20.0
SOURCE_LEVELING_MIN_ACTIVE_SECONDS = 0.5
DEBUG_AUDIO_PIPELINE_DEFAULT_ENABLED = False
RAW_RECORDER_CHUNK_FRAMES = 2048

AUTO_STOP_OFF = None
AUTO_STOP_5_MINUTES = 300
AUTO_STOP_10_MINUTES = 600
AUTO_STOP_20_MINUTES = 1200
AUTO_STOP_DEFAULT_SECONDS = AUTO_STOP_10_MINUTES
AUTO_STOP_MIN_RECORD_SECONDS = 60.0

AUTO_STOP_OPTIONS = [
    (AUTO_STOP_OFF, "Off"),
    (AUTO_STOP_5_MINUTES, "5 minutes"),
    (AUTO_STOP_10_MINUTES, "10 minutes"),
    (AUTO_STOP_20_MINUTES, "20 minutes"),
]

TRIM_SILENCE_DEFAULT_ENABLED = False
TRIM_EDGE_SILENCE_SECONDS = 5.0

MIC_ACTIVITY_DETECTOR_CONFIG = {
    "margin_db": 9.0,
    "min_threshold_db": -55.0,
    "max_threshold_db": -30.0,
}

LOOPBACK_ACTIVITY_DETECTOR_CONFIG = {
    "margin_db": 12.0,
    "min_threshold_db": -60.0,
    "max_threshold_db": -28.0,
}


def build_output_profile(fmt, quality, stereo):
    fmt_key = str(fmt or "").strip().lower()
    quality_key = str(quality or "").strip().lower()

    if fmt_key not in FORMAT_CONFIG:
        raise ValueError(f"Unsupported output format: {fmt}")
    if quality_key not in QUALITY_CONFIG:
        raise ValueError(f"Unsupported output quality: {quality}")

    format_config = FORMAT_CONFIG[fmt_key]
    quality_config = QUALITY_CONFIG[quality_key]
    channels = 2 if stereo else 1
    return {
        **quality_config,
        **format_config,
        "label": format_config["label"],
        "format_label": format_config["label"],
        "quality_label": quality_config["label"],
        "format_key": fmt_key,
        "quality_key": quality_key,
        "channels": channels,
    }


def describe_output_profile(fmt, quality, stereo):
    profile = build_output_profile(fmt, quality, stereo)
    rate_khz = profile["sample_rate"] // 1000
    channels = "stereo" if profile["channels"] == 2 else "mono"
    encoding = (
        f"{profile['mp3_bitrate_kbps']} kbps"
        if profile["encoder"] == "lameenc"
        else profile["subtype"]
    )
    return f"{profile['format_label']} / {rate_khz} kHz / {channels} / {encoding}"


def normalize_auto_stop_silence_seconds(value):
    if value is None or value is False:
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("", "off", "none", "false", "0"):
            return None
        try:
            value = int(normalized)
        except ValueError:
            return AUTO_STOP_DEFAULT_SECONDS

    try:
        seconds = int(value)
    except (TypeError, ValueError):
        return AUTO_STOP_DEFAULT_SECONDS

    supported = {seconds for seconds, _label in AUTO_STOP_OPTIONS if seconds is not None}
    if seconds in supported:
        return seconds
    return AUTO_STOP_DEFAULT_SECONDS


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


def normalize_trim_silence_enabled(value):
    return _normalize_bool_setting(value, TRIM_SILENCE_DEFAULT_ENABLED)


def normalize_echo_suppression_enabled(value):
    return _normalize_bool_setting(value, ECHO_SUPPRESSION_DEFAULT_ENABLED)


def normalize_noise_reduction_enabled(value):
    return _normalize_bool_setting(value, NOISE_REDUCTION_DEFAULT_ENABLED)


def normalize_debug_audio_pipeline_enabled(value):
    return _normalize_bool_setting(value, DEBUG_AUDIO_PIPELINE_DEFAULT_ENABLED)


def build_recording_filename(timestamp, extension):
    clean_extension = str(extension or "").strip()
    if clean_extension and not clean_extension.startswith("."):
        clean_extension = f".{clean_extension}"
    return f"{RECORDING_FILENAME_PREFIX}_{timestamp}{clean_extension}"


class SourceActivityDetector:
    def __init__(
        self,
        margin_db=10.0,
        min_threshold_db=-55.0,
        max_threshold_db=-28.0,
        history_seconds=30.0,
        active_window_seconds=1.0,
        active_ratio=0.2,
    ):
        self.margin_db = float(margin_db)
        self.min_threshold_db = float(min_threshold_db)
        self.max_threshold_db = float(max_threshold_db)
        self.history_seconds = float(history_seconds)
        self.active_window_seconds = float(active_window_seconds)
        self.active_ratio = float(active_ratio)
        self.recent_db_values = deque()
        self.recent_activity = deque()
        self.current_db = -120.0
        self.noise_floor_db = -120.0
        self.threshold_db = self.min_threshold_db
        self.active = False

    def update(self, data, samplerate, now=None):
        if now is None:
            now = time.monotonic()

        audio = np.asarray(data, dtype=np.float32)
        if audio.size == 0:
            return self.active

        mean_square = float(np.mean(audio ** 2))
        if mean_square <= 0:
            self.current_db = -100.0
        else:
            rms = float(np.sqrt(mean_square))
            self.current_db = max(20.0 * np.log10(rms), -100.0)

        self.recent_db_values.append((now, self.current_db))
        history_cutoff = now - self.history_seconds
        while self.recent_db_values and self.recent_db_values[0][0] < history_cutoff:
            self.recent_db_values.popleft()

        db_values = np.array([value for _timestamp, value in self.recent_db_values], dtype=np.float32)
        self.noise_floor_db = float(np.percentile(db_values, 20))
        dynamic_threshold = self.noise_floor_db + self.margin_db
        self.threshold_db = min(
            max(dynamic_threshold, self.min_threshold_db),
            self.max_threshold_db,
        )

        block_active = self.current_db >= self.threshold_db
        self.recent_activity.append((now, block_active))
        active_cutoff = now - self.active_window_seconds
        while self.recent_activity and self.recent_activity[0][0] < active_cutoff:
            self.recent_activity.popleft()

        total_blocks = len(self.recent_activity)
        active_blocks = sum(1 for _timestamp, is_active in self.recent_activity if is_active)
        self.active = total_blocks > 0 and (active_blocks / total_blocks) >= self.active_ratio
        return self.active


def build_activity_detector(source_name):
    config = (
        LOOPBACK_ACTIVITY_DETECTOR_CONFIG
        if source_name == "loopback"
        else MIC_ACTIVITY_DETECTOR_CONFIG
    )
    return SourceActivityDetector(**config)


def _as_2d_audio(data):
    audio = np.asarray(data)
    if audio.ndim == 1:
        audio = audio.reshape(-1, 1)
    return audio


def build_source_activity_timeline(
    data,
    samplerate,
    source_name,
    chunk_frames=RAW_RECORDER_CHUNK_FRAMES,
):
    audio = _as_2d_audio(data).astype(np.float32, copy=False)
    detector = build_activity_detector(source_name)
    timeline = []
    chunk_frames = int(chunk_frames)

    for start_frame in range(0, len(audio), chunk_frames):
        end_frame = min(len(audio), start_frame + chunk_frames)
        now = start_frame / float(samplerate)
        active = detector.update(audio[start_frame:end_frame], samplerate, now=now)
        timeline.append((start_frame, end_frame, bool(active)))

    return timeline


def _combine_activity_timelines(source_timelines, total_frames, chunk_frames):
    combined = []
    chunk_frames = int(chunk_frames)
    chunk_count = int(np.ceil(total_frames / float(chunk_frames))) if total_frames else 0

    for chunk_index in range(chunk_count):
        start_frame = chunk_index * chunk_frames
        end_frame = min(total_frames, start_frame + chunk_frames)
        active = any(
            chunk_index < len(timeline) and bool(timeline[chunk_index][2])
            for timeline in source_timelines
        )
        combined.append((start_frame, end_frame, active))

    return combined


def _bounds_from_activity_timeline(
    timeline,
    total_frames,
    samplerate,
    keep_silence_seconds=TRIM_EDGE_SILENCE_SECONDS,
):
    keep_frames = max(1, int(round(float(samplerate) * keep_silence_seconds)))
    active_entries = [entry for entry in timeline if entry[2]]

    if total_frames <= 0:
        return 0, 0, {
            "applied": False,
            "start_removed_seconds": 0.0,
            "end_removed_seconds": 0.0,
        }

    if not active_entries:
        end_frame = min(total_frames, keep_frames)
        return 0, end_frame, {
            "applied": end_frame < total_frames,
            "start_removed_seconds": 0.0,
            "end_removed_seconds": (total_frames - end_frame) / float(samplerate),
        }

    first_active_frame = active_entries[0][0]
    last_active_frame = active_entries[-1][1]
    start_frame = max(0, first_active_frame - keep_frames)
    end_frame = min(total_frames, last_active_frame + keep_frames)

    return start_frame, end_frame, {
        "applied": start_frame > 0 or end_frame < total_frames,
        "start_removed_seconds": start_frame / float(samplerate),
        "end_removed_seconds": (total_frames - end_frame) / float(samplerate),
    }


def calculate_trim_bounds_for_sources(
    sources,
    samplerate,
    keep_silence_seconds=TRIM_EDGE_SILENCE_SECONDS,
    chunk_frames=RAW_RECORDER_CHUNK_FRAMES,
):
    normalized_sources = [
        (source_name, _as_2d_audio(data))
        for source_name, data in sources
    ]
    total_frames = max((len(data) for _source_name, data in normalized_sources), default=0)
    source_timelines = [
        build_source_activity_timeline(
            data,
            samplerate,
            source_name,
            chunk_frames=chunk_frames,
        )
        for source_name, data in normalized_sources
    ]
    combined_timeline = _combine_activity_timelines(
        source_timelines,
        total_frames,
        chunk_frames,
    )
    start_frame, end_frame, stats = _bounds_from_activity_timeline(
        combined_timeline,
        total_frames,
        samplerate,
        keep_silence_seconds=keep_silence_seconds,
    )
    stats["source_names"] = [source_name for source_name, _data in normalized_sources]
    return start_frame, end_frame, stats


def trim_edge_silence_data(
    data,
    samplerate,
    source_name="mic",
    keep_silence_seconds=TRIM_EDGE_SILENCE_SECONDS,
    chunk_frames=RAW_RECORDER_CHUNK_FRAMES,
):
    audio = _as_2d_audio(data)
    start_frame, end_frame, stats = calculate_trim_bounds_for_sources(
        [(source_name, audio)],
        samplerate,
        keep_silence_seconds=keep_silence_seconds,
        chunk_frames=chunk_frames,
    )
    return audio[start_frame:end_frame].copy(), stats


class AutoStopController:
    def __init__(
        self,
        silence_seconds=None,
        min_record_seconds=AUTO_STOP_MIN_RECORD_SECONDS,
        start_ts=None,
    ):
        self.silence_seconds = normalize_auto_stop_silence_seconds(silence_seconds)
        self.min_record_seconds = float(min_record_seconds)
        self.start_ts = time.monotonic() if start_ts is None else float(start_ts)
        self.last_active_ts = self.start_ts

    def update(self, source_active, now=None):
        if now is None:
            now = time.monotonic()

        if self.silence_seconds is None:
            return False

        any_active = any(bool(active) for active in source_active.values())
        if any_active:
            self.last_active_ts = now
            return False

        if now - self.start_ts < self.min_record_seconds:
            return False

        return now - self.last_active_ts >= self.silence_seconds

    def reason(self):
        if self.silence_seconds is None:
            return None
        minutes = int(self.silence_seconds // 60)
        return f"silence_timeout_{minutes}min"


class RawRecorder(threading.Thread):
    """
    Helper thread to record a single device to a WAV file.
    """
    def __init__(
        self,
        device,
        filepath,
        samplerate=44100,
        channels=2,
        subtype="PCM_16",
        source_name=None,
        on_audio_data=None,
    ):
        super().__init__()
        self.device = device
        self.filepath = filepath
        self.samplerate = samplerate
        self.channels = channels
        self.subtype = subtype
        self.source_name = source_name
        self.on_audio_data = on_audio_data
        self.stop_event = threading.Event()
        self.error = None

    def run(self):
        try:
            with sf.SoundFile(
                self.filepath,
                mode="w",
                samplerate=self.samplerate,
                channels=self.channels,
                format="WAV",
                subtype=self.subtype,
            ) as f_wav:
                with self.device.recorder(samplerate=self.samplerate, channels=self.channels) as mic:
                    while not self.stop_event.is_set():
                        data = mic.record(numframes=RAW_RECORDER_CHUNK_FRAMES)
                        if self.on_audio_data and self.source_name:
                            self.on_audio_data(self.source_name, data)
                        f_wav.write(data)
        except Exception as e:
            self.error = str(e)

    def stop(self):
        self.stop_event.set()
        self.join()

class AudioRecorder(threading.Thread):
    """
    Orchestrates recording from Microphone, Loopback, or Both.
    """
    def __init__(
        self,
        mic_id,
        source_mode,
        output_folder,
        output_format="flac",
        quality="balanced",
        stereo=False,
        normalize=False,
        echo_suppression=ECHO_SUPPRESSION_DEFAULT_ENABLED,
        noise_reduction=NOISE_REDUCTION_DEFAULT_ENABLED,
        debug_audio_pipeline=DEBUG_AUDIO_PIPELINE_DEFAULT_ENABLED,
        trim_silence=TRIM_SILENCE_DEFAULT_ENABLED,
        auto_stop_silence_seconds=AUTO_STOP_DEFAULT_SECONDS,
        on_finish_callback=None,
    ):
        super().__init__()
        self.mic_id = mic_id
        self.source_mode = source_mode # "mic", "loopback", "both"
        self.output_folder = output_folder
        self.output_format = str(output_format or "flac").strip().lower()
        self.quality = str(quality or "balanced").strip().lower()
        self.stereo = bool(stereo)
        self.profile = build_output_profile(self.output_format, self.quality, self.stereo)
        self.normalize = normalize
        self.echo_suppression = normalize_echo_suppression_enabled(echo_suppression)
        self.noise_reduction = normalize_noise_reduction_enabled(noise_reduction)
        self.debug_audio_pipeline = normalize_debug_audio_pipeline_enabled(debug_audio_pipeline)
        self.trim_silence = normalize_trim_silence_enabled(trim_silence)
        self.auto_stop_silence_seconds = normalize_auto_stop_silence_seconds(auto_stop_silence_seconds)
        self.callback = on_finish_callback
        
        self.recording = False
        self.stop_event = threading.Event()
        self.error_message = None
        self.final_filepath = None
        
        # Temp files
        self.temp_files = []
        self.recorders = []
        self.activity_lock = threading.Lock()
        self.activity_detectors = {}
        self.source_active = {}
        self.auto_stop_controller = None
        self.auto_stop_triggered = False
        self.auto_stop_reason = None
        self.trim_silence_applied = False
        self.trim_silence_removed_seconds = 0.0
        self.echo_suppression_applied = False
        self.echo_suppression_reason = "not_run"
        self.echo_suppression_stats = {}
        self.noise_reduction_applied = False
        self.noise_reduction_reason = "not_run"
        self.source_leveling_stats = {}
        self.debug_audio_artifacts = []
        self.debug_audio_dir = None
        self.finish_metadata = self.build_finish_metadata()

    def _active_source_names(self):
        if self.source_mode in ("both", "mic_reference"):
            return ["mic", "loopback"]
        if self.source_mode == "loopback":
            return ["loopback"]
        return ["mic"]

    def _build_activity_detector(self, source_name):
        return build_activity_detector(source_name)

    def _setup_auto_stop(self):
        source_names = self._active_source_names()
        self.activity_detectors = {
            source_name: self._build_activity_detector(source_name)
            for source_name in source_names
        }
        self.source_active = {source_name: False for source_name in source_names}
        self.auto_stop_controller = AutoStopController(
            silence_seconds=self.auto_stop_silence_seconds,
            min_record_seconds=AUTO_STOP_MIN_RECORD_SECONDS,
        )
        self.auto_stop_triggered = False
        self.auto_stop_reason = None

    def report_activity(self, source_name, data, now=None):
        with self.activity_lock:
            detector = self.activity_detectors.get(source_name)
            if detector is None:
                return

            self.source_active[source_name] = detector.update(
                data,
                self.profile["sample_rate"],
                now=now,
            )

            if self.auto_stop_controller and self.auto_stop_controller.update(self.source_active, now=now):
                self.request_auto_stop(self.auto_stop_controller.reason())

    def request_auto_stop(self, reason):
        if self.auto_stop_triggered:
            return
        self.auto_stop_triggered = True
        self.auto_stop_reason = reason
        self.stop_event.set()

    def build_finish_metadata(self):
        return {
            "auto_stop_enabled": self.auto_stop_silence_seconds is not None,
            "auto_stop_silence_seconds": self.auto_stop_silence_seconds,
            "auto_stop_triggered": self.auto_stop_triggered,
            "auto_stop_reason": self.auto_stop_reason,
            "trim_silence_enabled": self.trim_silence,
            "trim_silence_keep_seconds": TRIM_EDGE_SILENCE_SECONDS,
            "trim_silence_applied": self.trim_silence_applied,
            "trim_silence_removed_seconds": round(self.trim_silence_removed_seconds, 3),
            "echo_suppression_enabled": self.echo_suppression,
            "echo_suppression_applied": self.echo_suppression_applied,
            "echo_suppression_reason": self.echo_suppression_reason,
            "echo_suppression_stats": self.echo_suppression_stats,
            "noise_reduction_enabled": self.noise_reduction,
            "noise_reduction_available": denoise.is_available(),
            "noise_reduction_applied": self.noise_reduction_applied,
            "noise_reduction_reason": self.noise_reduction_reason,
            "source_leveling_enabled": bool(self.normalize),
            "source_leveling_stats": self.source_leveling_stats,
            "debug_audio_pipeline_enabled": self.debug_audio_pipeline,
            "debug_audio_dir": self.debug_audio_dir,
        }

    def _metadata_json_safe(self, value):
        if isinstance(value, dict):
            return {str(k): self._metadata_json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._metadata_json_safe(v) for v in value]
        if isinstance(value, np.generic):
            return value.item()
        return value

    def _sidecar_metadata(self, final_filepath):
        metadata = self.build_finish_metadata()
        metadata.update({
            "final_filepath": final_filepath,
            "source_mode": self.source_mode,
            "output_format": self.output_format,
            "quality": self.quality,
            "stereo": self.stereo,
            "normalize": bool(self.normalize),
            "profile": self.profile,
            "debug_audio_artifacts": [
                {"label": item["label"], "path": item.get("exported_path")}
                for item in self.debug_audio_artifacts
                if item.get("exported_path")
            ],
        })
        return self._metadata_json_safe(metadata)

    def _write_metadata_sidecar(self, final_filepath):
        sidecar_path = os.path.splitext(final_filepath)[0] + ".json"
        metadata = self._sidecar_metadata(final_filepath)
        with open(sidecar_path, "w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2, ensure_ascii=False)
        return sidecar_path

    def _record_debug_audio(self, label, path=None, data=None, samplerate=None, subtype=None):
        if not self.debug_audio_pipeline:
            return

        artifact_path = path
        if data is not None:
            artifact_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
            sf.write(
                artifact_path,
                data,
                samplerate,
                format="WAV",
                subtype=subtype or self.profile["subtype"],
            )
            self.temp_files.append(artifact_path)

        if artifact_path:
            self.debug_audio_artifacts.append({
                "label": label,
                "path": artifact_path,
            })

    def _export_debug_audio_artifacts(self, final_filepath):
        if not self.debug_audio_pipeline or not self.debug_audio_artifacts:
            self.debug_audio_dir = None
            return None

        base, _ext = os.path.splitext(final_filepath)
        debug_dir = f"{base}_debug"
        os.makedirs(debug_dir, exist_ok=True)
        for index, item in enumerate(self.debug_audio_artifacts, start=1):
            src = item.get("path")
            if not src or not os.path.exists(src):
                continue
            filename = f"{index:02d}_{item['label']}.wav"
            dst = os.path.join(debug_dir, filename)
            shutil.copy2(src, dst)
            item["exported_path"] = dst
        self.debug_audio_dir = debug_dir
        return debug_dir

    def _get_device(self, is_loopback):
        if is_loopback:
            # For loopback, we try to find the default speaker's loopback
            default_speaker = sc.default_speaker()
            mics = sc.all_microphones(include_loopback=True)
            # Try exact name match
            loopback_mic = next((m for m in mics if m.name == default_speaker.name), None)
            # Try fuzzy match
            if not loopback_mic:
                loopback_mic = next((m for m in mics if default_speaker.name in m.name), None)
            
            if not loopback_mic:
                raise Exception("Could not detect System Audio loopback device.")
            return loopback_mic
        else:
            return sc.get_microphone(self.mic_id, include_loopback=False)

    def run(self):
        self.recording = True
        self.error_message = None
        self.temp_files = []
        self.recorders = []
        
        try:
            samplerate = self.profile["sample_rate"]
            channels = self.profile["channels"]
            subtype = self.profile["subtype"]
            self._setup_auto_stop()

            # 1. Setup Recorders
            if self.source_mode in ("both", "mic_reference"):
                # Need two recorders
                dev_mic = self._get_device(is_loopback=False)
                dev_loop = self._get_device(is_loopback=True)
                
                t1 = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
                t2 = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
                self.temp_files = [t1, t2]
                
                self.recorders.append(
                    RawRecorder(
                        dev_mic,
                        t1,
                        samplerate=samplerate,
                        channels=channels,
                        subtype=subtype,
                        source_name="mic",
                        on_audio_data=self.report_activity,
                    )
                )
                self.recorders.append(
                    RawRecorder(
                        dev_loop,
                        t2,
                        samplerate=samplerate,
                        channels=channels,
                        subtype=subtype,
                        source_name="loopback",
                        on_audio_data=self.report_activity,
                    )
                )
                
            elif self.source_mode == "loopback":
                dev = self._get_device(is_loopback=True)
                t1 = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
                self.temp_files = [t1]
                self.recorders.append(
                    RawRecorder(
                        dev,
                        t1,
                        samplerate=samplerate,
                        channels=channels,
                        subtype=subtype,
                        source_name="loopback",
                        on_audio_data=self.report_activity,
                    )
                )
                
            else: # mic
                dev = self._get_device(is_loopback=False)
                t1 = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
                self.temp_files = [t1]
                self.recorders.append(
                    RawRecorder(
                        dev,
                        t1,
                        samplerate=samplerate,
                        channels=channels,
                        subtype=subtype,
                        source_name="mic",
                        on_audio_data=self.report_activity,
                    )
                )

            print(f"Starting recording mode: {self.source_mode}")

            # 2. Start Recording
            for r in self.recorders:
                r.start()
            
            # Wait for stop signal
            self.stop_event.wait()
            
            # 3. Stop Recording
            for r in self.recorders:
                r.stop()
                if r.error:
                    raise Exception(f"Recorder error: {r.error}")

            # 4. Mix/Process
            source_wav = self._prepare_source_wav(subtype)
            
            # 6. Finalize
            if not os.path.exists(self.output_folder):
                os.makedirs(self.output_folder)

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = build_recording_filename(timestamp, self.profile["extension"])
            self.final_filepath = os.path.join(self.output_folder, filename)
            self._write_final_output(source_wav, self.final_filepath)
            self._export_debug_audio_artifacts(self.final_filepath)
            self.finish_metadata = self.build_finish_metadata()
            self._write_metadata_sidecar(self.final_filepath)

        except Exception as e:
            self.error_message = str(e)
            print(f"Error during recording process: {e}")
        finally:
            self.recording = False
            # Clean up all temp files
            for t in self.temp_files:
                if os.path.exists(t):
                    try:
                        os.remove(t)
                    except: pass

            self.finish_metadata = self.build_finish_metadata()

            if self.callback:
                self.callback(self.final_filepath, self.error_message)

    def stop(self):
        self.stop_event.set()

    def _maybe_trim_temp_sources(self):
        if not self.trim_silence:
            self.trim_silence_applied = False
            self.trim_silence_removed_seconds = 0.0
            return

        source_names = self._active_source_names()
        source_paths = self.temp_files[:len(source_names)]
        if not source_paths:
            return

        try:
            loaded_sources = []
            file_infos = []
            samplerate = None

            for source_name, path in zip(source_names, source_paths):
                info = sf.info(path)
                data, sr = sf.read(path, always_2d=True)
                if samplerate is None:
                    samplerate = sr
                elif sr != samplerate:
                    raise ValueError("Cannot trim audio with different sample rates.")
                loaded_sources.append((source_name, data))
                file_infos.append((path, info, data))

            start_frame, end_frame, stats = calculate_trim_bounds_for_sources(
                loaded_sources,
                samplerate,
            )

            if stats["applied"]:
                target_frames = max(0, end_frame - start_frame)
                for path, info, data in file_infos:
                    trimmed = np.zeros((target_frames, data.shape[1]), dtype=data.dtype)
                    clip_start = min(max(start_frame, 0), len(data))
                    clip_end = min(max(end_frame, 0), len(data))
                    if clip_end > clip_start:
                        segment = data[clip_start:clip_end]
                        trimmed[:len(segment)] = segment
                    sf.write(path, trimmed, samplerate, format=info.format, subtype=info.subtype)

            self.trim_silence_applied = bool(stats["applied"])
            self.trim_silence_removed_seconds = (
                float(stats["start_removed_seconds"]) + float(stats["end_removed_seconds"])
            )
        except Exception as e:
            print(f"Silence trim failed: {e}")
            self.trim_silence_applied = False
            self.trim_silence_removed_seconds = 0.0

    def _build_leveling_mask(self, data):
        audio = np.asarray(data, dtype=np.float32)
        if audio.ndim == 1:
            audio = audio.reshape(-1, 1)
        return np.max(np.abs(audio), axis=1) >= NORMALIZE_ACTIVE_FLOOR

    def _level_source_data(self, data, samplerate, source_name):
        mask = self._build_leveling_mask(data)
        leveled, stats = level_active_source(
            data,
            mask,
            SourceLevelingConfig(
                enabled=bool(self.normalize),
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

    def _write_leveled_source_wav(self, source_wav, source_name, subtype):
        info = sf.info(source_wav)
        data, sr = sf.read(source_wav, always_2d=True)
        leveled = self._level_source_data(data, sr, source_name)
        label = f"leveled_{source_name}"
        self._record_debug_audio(label, data=leveled, samplerate=sr, subtype=info.subtype)
        leveled_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        sf.write(leveled_wav, leveled, sr, format=info.format, subtype=subtype or info.subtype)
        self.temp_files.append(leveled_wav)
        return leveled_wav

    def _denoise_mic_data(self, data, samplerate):
        cleaned, stats = denoise.reduce_noise(
            data,
            samplerate,
            NoiseReductionConfig(
                enabled=self.noise_reduction,
                mix=NOISE_REDUCTION_MIX,
                latency_ms=NOISE_REDUCTION_LATENCY_MS,
            ),
        )
        self.noise_reduction_applied = bool(stats.get("applied"))
        self.noise_reduction_reason = str(stats.get("reason", "unknown"))
        return cleaned

    def _mix_audio_data(self, d1, d2, out_file, samplerate, subtype, limit_output=False):
        if d1.shape[1] != d2.shape[1]:
            raise ValueError("Cannot mix audio with different channel counts.")

        max_len = max(len(d1), len(d2))
        if len(d1) < max_len:
            d1 = np.concatenate(
                (d1, np.zeros((max_len - len(d1), d1.shape[1]), dtype=d1.dtype))
            )
        if len(d2) < max_len:
            d2 = np.concatenate(
                (d2, np.zeros((max_len - len(d2), d2.shape[1]), dtype=d2.dtype))
            )

        mixed = d1 + d2
        if limit_output:
            mixed = self._apply_limiter(mixed)
        else:
            mixed = np.clip(mixed, -1.0, 1.0)
        sf.write(out_file, mixed, samplerate, format="WAV", subtype=subtype)

    def _prepare_both_source_wav(self, subtype):
        mic_wav, loopback_wav = self.temp_files[:2]
        mic_info = sf.info(mic_wav)
        loopback_info = sf.info(loopback_wav)
        mic_data, mic_sr = sf.read(mic_wav, always_2d=True)
        loopback_data, loopback_sr = sf.read(loopback_wav, always_2d=True)

        if mic_sr != loopback_sr:
            raise ValueError("Cannot mix audio with different sample rates.")
        if mic_data.shape[1] != loopback_data.shape[1]:
            raise ValueError("Cannot mix audio with different channel counts.")

        self.source_leveling_stats = {}
        self._record_debug_audio("raw_mic", path=mic_wav)
        self._record_debug_audio("raw_loopback", path=loopback_wav)
        if self.echo_suppression:
            mic_data, echo_stats = suppress_reference_echo(
                mic_data,
                loopback_data,
                mic_sr,
                EchoSuppressionConfig(enabled=True),
            )
            self.echo_suppression_applied = bool(echo_stats.get("applied"))
            self.echo_suppression_reason = str(echo_stats.get("reason", "unknown"))
            self.echo_suppression_stats = echo_stats
        else:
            self.echo_suppression_applied = False
            self.echo_suppression_reason = "disabled"
            self.echo_suppression_stats = {}

        self._record_debug_audio("aec_mic", data=mic_data, samplerate=mic_sr, subtype=subtype)
        mic_data = self._denoise_mic_data(mic_data, mic_sr)
        self._record_debug_audio("denoised_mic", data=mic_data, samplerate=mic_sr, subtype=subtype)
        mic_data = self._level_source_data(mic_data, mic_sr, "mic")
        self._record_debug_audio("leveled_mic", data=mic_data, samplerate=mic_sr, subtype=subtype)

        output_subtype = subtype or mic_info.subtype or loopback_info.subtype
        if self.source_mode == "mic_reference":
            mic_output_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
            sf.write(mic_output_wav, mic_data, mic_sr, format="WAV", subtype=output_subtype)
            self.temp_files.append(mic_output_wav)
            self._record_debug_audio("mic_reference_output", path=mic_output_wav)
            return mic_output_wav

        loopback_data = self._level_source_data(loopback_data, loopback_sr, "loopback")
        self._record_debug_audio(
            "leveled_loopback",
            data=loopback_data,
            samplerate=loopback_sr,
            subtype=subtype,
        )
        mixed_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        self._mix_audio_data(
            mic_data,
            loopback_data,
            mixed_wav,
            mic_sr,
            output_subtype,
            limit_output=bool(self.normalize),
        )
        self.temp_files.append(mixed_wav)
        self._record_debug_audio("mixed_pre_trim", path=mixed_wav)
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

    def _prepare_source_wav(self, subtype):
        if len(self.temp_files) == 2:
            mixed = self._prepare_both_source_wav(subtype)
            return self._maybe_trim_final_wav(mixed)

        self.echo_suppression_applied = False
        self.echo_suppression_reason = "single_source"
        self.echo_suppression_stats = {}
        self.source_leveling_stats = {}

        source_wav = self.temp_files[0]
        self._record_debug_audio(f"raw_{self.source_mode}", path=source_wav)
        if self.source_mode == "mic" and self.noise_reduction:
            data, sr = sf.read(source_wav, always_2d=True)
            info = sf.info(source_wav)
            cleaned = self._denoise_mic_data(data, sr)
            self._record_debug_audio("denoised_mic", data=cleaned, samplerate=sr, subtype=info.subtype)
            denoised_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
            sf.write(denoised_wav, cleaned, sr, format=info.format, subtype=info.subtype)
            self.temp_files.append(denoised_wav)
            source_wav = denoised_wav
        else:
            self.noise_reduction_applied = False
            self.noise_reduction_reason = "disabled" if not self.noise_reduction else "not_mic_source"

        if self.normalize:
            source_name = "loopback" if self.source_mode == "loopback" else "mic"
            source_wav = self._write_leveled_source_wav(source_wav, source_name, subtype)
        return self._maybe_trim_final_wav(source_wav)

    def _mix_audio(self, file1, file2, out_file, subtype, limit_output=False):
        d1, sr1 = sf.read(file1, always_2d=True)
        d2, sr2 = sf.read(file2, always_2d=True)

        if sr1 != sr2:
            raise ValueError("Cannot mix audio with different sample rates.")
        if d1.shape[1] != d2.shape[1]:
            raise ValueError("Cannot mix audio with different channel counts.")
        
        # Ensure same length
        max_len = max(len(d1), len(d2))
        
        # Pad d1
        if len(d1) < max_len:
            pad_width = max_len - len(d1)
            # handle mono/stereo padding
            shape = (pad_width, d1.shape[1])
            d1 = np.concatenate((d1, np.zeros(shape, dtype=d1.dtype)))
            
        # Pad d2
        if len(d2) < max_len:
            pad_width = max_len - len(d2)
            shape = (pad_width, d2.shape[1])
            d2 = np.concatenate((d2, np.zeros(shape, dtype=d2.dtype)))
            
        mixed = d1 + d2
        if limit_output:
            mixed = self._apply_limiter(mixed)
        else:
            mixed = np.clip(mixed, -1.0, 1.0)
        
        sf.write(out_file, mixed, sr1, format="WAV", subtype=subtype)

    def _normalize_audio(self, filepath):
        try:
            info = sf.info(filepath)
            data, sr = sf.read(filepath, always_2d=True)
            data = self._normalize_audio_data(data)
            sf.write(filepath, data, sr, format=info.format, subtype=info.subtype)
        except Exception as e:
            print(f"Normalization failed: {e}")

    def _normalize_audio_data(self, data):
        active = np.abs(data)
        active = active[active >= NORMALIZE_ACTIVE_FLOOR]
        if active.size == 0:
            return self._apply_limiter(data)

        rms = float(np.sqrt(np.mean(active ** 2)))
        percentile = float(np.percentile(active, NORMALIZE_REFERENCE_PERCENTILE))
        reference_level = max(rms, percentile)
        if not np.isfinite(reference_level) or reference_level <= 0:
            return self._apply_limiter(data)

        gain = min(NORMALIZE_TARGET_LEVEL / reference_level, NORMALIZE_MAX_GAIN)
        return self._apply_limiter(data * gain)

    def _limit_audio(self, filepath):
        info = sf.info(filepath)
        data, sr = sf.read(filepath, always_2d=True)
        data = self._apply_limiter(data)
        sf.write(filepath, data, sr, format=info.format, subtype=info.subtype)

    def _apply_limiter(self, data):
        return np.clip(data, -NORMALIZE_LIMIT, NORMALIZE_LIMIT)

    def _write_final_output(self, source_wav, final_filepath):
        if self.profile["encoder"] == "lameenc":
            self._convert_to_mp3(source_wav, final_filepath, self.profile["mp3_bitrate_kbps"])
            return

        data, sr = sf.read(source_wav, always_2d=True)
        if self.profile["channels"] == 1 and data.shape[1] > 1:
            data = np.mean(data, axis=1, keepdims=True)

        sf.write(
            final_filepath,
            data,
            sr,
            format=self.profile["format"],
            subtype=self.profile["subtype"],
        )

    def _convert_to_mp3(self, src_wav, dst_mp3, bitrate_kbps):
        data, sr = sf.read(src_wav, always_2d=True)
        if self.profile["channels"] == 1 and data.shape[1] > 1:
            data = np.mean(data, axis=1, keepdims=True)
        channels = data.shape[1]
        
        pcm_data = (data * 32767).clip(-32768, 32767).astype(np.int16)
        
        encoder = lameenc.Encoder()
        encoder.set_bit_rate(bitrate_kbps)
        encoder.set_in_sample_rate(sr)
        encoder.set_channels(channels)
        encoder.set_quality(2)
        
        mp3_data = encoder.encode(pcm_data.tobytes())
        mp3_data += encoder.flush()
        
        with open(dst_mp3, "wb") as f_mp3:
            f_mp3.write(mp3_data)

def get_devices(include_loopback=False):
    try:
        devices = sc.all_microphones(include_loopback=include_loopback)
        return [{"id": d.id, "name": d.name} for d in devices]
    except Exception as e:
        print(f"Error fetching devices: {e}")
        return []
