import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QObject, Qt
from PyQt6.QtWidgets import QApplication, QMenu

from app_metadata import SETTINGS_WINDOW_TITLE, TRAY_IDLE_TOOLTIP
from gui import RecordingIndicator, SettingsWindow, TrayApplication


class FakeSettingsWindow:
    def __init__(self, settings):
        self.settings = settings

    def get_settings(self):
        return self.settings


class FakeRecordingIndicator:
    def __init__(self):
        self.show_count = 0
        self.hide_count = 0
        self.finished_count = 0
        self.finished_hide_delay_ms = None
        self.is_finishing = False
        self.context_menu = None
        self.visible = True

    def show_recording(self):
        self.show_count += 1
        self.is_finishing = False

    def hide_recording(self):
        self.hide_count += 1
        self.is_finishing = False
        self.visible = False

    def show_finished(self, hide_after_ms=None):
        self.finished_count += 1
        self.finished_hide_delay_ms = hide_after_ms
        self.is_finishing = True

    def setContextMenu(self, menu):
        self.context_menu = menu

    def isVisible(self):
        return self.visible


class FakeContextMenu:
    def __init__(self):
        self.exec_count = 0

    def exec(self, position):
        self.exec_count += 1


class FakeContextMenuEvent:
    def __init__(self):
        self.accepted = False

    def globalPos(self):
        return None

    def accept(self):
        self.accepted = True


class FakeMouseEvent:
    def __init__(self, button=Qt.MouseButton.LeftButton):
        self._button = button
        self.accepted = False

    def button(self):
        return self._button

    def accept(self):
        self.accepted = True


class FakeTrayIcon:
    def __init__(self):
        self.messages = []
        self.context_menu = None
        self.tooltip = None

    def setContextMenu(self, menu):
        self.context_menu = menu

    def setIcon(self, icon):
        self.icon = icon

    def setToolTip(self, text):
        self.tooltip = text

    def showMessage(self, title, message, icon=None, duration=0):
        self.messages.append((title, message, icon, duration))


class SettingsWindowRecordingIndicatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def make_window(self, data):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        config_path = os.path.join(temp_dir.name, "settings.json")
        with open(config_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh)

        patcher = patch("gui.CONFIG_FILE", config_path)
        patcher.start()
        self.addCleanup(patcher.stop)

        window = SettingsWindow()
        self.addCleanup(window.close)
        return window

    def test_recording_indicator_setting_defaults_on(self):
        window = self.make_window({})

        settings = window.get_settings()

        self.assertIs(settings["show_recording_indicator"], True)

    def test_recording_indicator_setting_can_be_disabled(self):
        window = self.make_window({"show_recording_indicator": False})

        settings = window.get_settings()

        self.assertIs(settings["show_recording_indicator"], False)

    def test_settings_window_title_uses_meetrec_name(self):
        window = self.make_window({})

        self.assertEqual(window.windowTitle(), SETTINGS_WINDOW_TITLE)


class TrayApplicationMenuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def make_subject(self):
        subject = QObject()
        subject.tray_icon = FakeTrayIcon()
        subject.recording_indicator = FakeRecordingIndicator()
        subject.start_recording = lambda mode: None
        subject.stop_recording = lambda: None
        subject.open_settings = lambda: None
        subject.exit_app = lambda: None
        subject.open_recordings_folder = lambda: None
        return subject

    def test_build_menu_places_open_folder_before_settings(self):
        subject = self.make_subject()

        TrayApplication.build_menu(subject)

        action_texts = [
            action.text()
            for action in subject.menu.actions()
            if not action.isSeparator()
        ]
        self.assertLess(
            action_texts.index("Open Recordings Folder"),
            action_texts.index("Settings"),
        )

    def test_build_menu_shares_context_menu_with_recording_indicator(self):
        subject = self.make_subject()

        TrayApplication.build_menu(subject)

        self.assertIs(subject.recording_indicator.context_menu, subject.menu)


class RecordingIndicatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_formats_elapsed_time_as_mm_ss(self):
        self.assertEqual(RecordingIndicator.format_elapsed(0), "00:00")
        self.assertEqual(RecordingIndicator.format_elapsed(59), "00:59")
        self.assertEqual(RecordingIndicator.format_elapsed(60), "01:00")

    def test_click_request_emits_stop_signal(self):
        indicator = RecordingIndicator()
        self.addCleanup(indicator.close)
        calls = []
        indicator.stop_requested.connect(lambda: calls.append("stop"))

        indicator.request_stop()

        self.assertEqual(calls, ["stop"])

    def test_finished_state_left_click_opens_recordings_folder(self):
        indicator = RecordingIndicator()
        self.addCleanup(indicator.close)
        stop_calls = []
        open_calls = []
        indicator.stop_requested.connect(lambda: stop_calls.append("stop"))
        indicator.open_folder_requested.connect(lambda: open_calls.append("open"))
        event = FakeMouseEvent()

        indicator.show_finished()
        indicator.mouseReleaseEvent(event)

        self.assertEqual(stop_calls, [])
        self.assertEqual(open_calls, ["open"])
        self.assertTrue(event.accepted)

    def test_finished_state_keeps_context_menu_available(self):
        indicator = RecordingIndicator()
        self.addCleanup(indicator.close)
        menu = FakeContextMenu()
        event = FakeContextMenuEvent()

        indicator.setContextMenu(menu)
        indicator.show_finished()
        indicator.contextMenuEvent(event)

        self.assertEqual(menu.exec_count, 1)
        self.assertTrue(event.accepted)

    def test_finished_state_hides_after_five_seconds(self):
        indicator = RecordingIndicator()
        self.addCleanup(indicator.close)

        indicator.show_recording()
        indicator.show_finished()

        self.assertTrue(indicator.is_finishing)
        self.assertTrue(indicator.finished_hide_timer.isActive())
        self.assertEqual(indicator.finished_hide_timer.interval(), 5000)

    def test_new_recording_cancels_pending_finished_hide(self):
        indicator = RecordingIndicator()
        self.addCleanup(indicator.close)

        indicator.show_finished()
        indicator.show_recording()

        self.assertFalse(indicator.is_finishing)
        self.assertFalse(indicator.finished_hide_timer.isActive())


class TrayApplicationRecordingIndicatorTests(unittest.TestCase):
    def make_subject(self, show_indicator=True):
        indicator = FakeRecordingIndicator()
        subject = SimpleNamespace(
            recorder=None,
            last_mode=None,
            settings_window=FakeSettingsWindow(
                {
                    "device_id": "mic1",
                    "output_folder": "D:/recordings",
                    "format": "flac",
                    "quality": "balanced",
                    "stereo": False,
                    "normalize": True,
                    "show_recording_indicator": show_indicator,
                    "clipboard": False,
                }
            ),
            signals=SimpleNamespace(
                recording_finished=SimpleNamespace(emit=lambda path, error: None)
            ),
            action_record_mic=SimpleNamespace(setEnabled=lambda enabled: None),
            action_record_loop=SimpleNamespace(setEnabled=lambda enabled: None),
            action_record_both=SimpleNamespace(setEnabled=lambda enabled: None),
            action_stop=SimpleNamespace(setEnabled=lambda enabled: None),
            tray_icon=FakeTrayIcon(),
            icon_rec_path="recording.ico",
            icon_idle_path="idle.ico",
            recording_indicator=indicator,
        )
        return subject, indicator

    def test_start_recording_shows_indicator_when_enabled(self):
        subject, indicator = self.make_subject(show_indicator=True)

        with patch("gui.QIcon"), patch("gui.AudioRecorder") as AudioRecorder:
            TrayApplication.start_recording(subject, "mic")

        AudioRecorder.return_value.start.assert_called_once_with()
        self.assertEqual(indicator.show_count, 1)

    def test_start_recording_skips_indicator_when_disabled(self):
        subject, indicator = self.make_subject(show_indicator=False)

        with patch("gui.QIcon"), patch("gui.AudioRecorder"):
            TrayApplication.start_recording(subject, "mic")

        self.assertEqual(indicator.show_count, 0)

    def test_start_recording_uses_meetrec_recording_tooltip(self):
        subject, indicator = self.make_subject(show_indicator=True)

        with patch("gui.QIcon"), patch("gui.AudioRecorder"):
            TrayApplication.start_recording(subject, "mic")

        self.assertEqual(subject.tray_icon.tooltip, "MeetRec Recording (mic)")

    def test_recording_finished_hides_indicator(self):
        subject, indicator = self.make_subject(show_indicator=True)

        with patch("gui.QIcon"):
            TrayApplication.on_recording_finished(subject, "D:/recordings/test.flac", "")

        self.assertEqual(indicator.hide_count, 1)

    def test_recording_finished_restores_meetrec_idle_tooltip(self):
        subject, indicator = self.make_subject(show_indicator=True)

        with patch("gui.QIcon"):
            TrayApplication.on_recording_finished(subject, "D:/recordings/test.flac", "")

        self.assertEqual(subject.tray_icon.tooltip, TRAY_IDLE_TOOLTIP)

    def test_recording_finished_keeps_finished_indicator_until_delay_expires(self):
        subject, indicator = self.make_subject(show_indicator=True)
        indicator.is_finishing = True

        with patch("gui.QIcon"):
            TrayApplication.on_recording_finished(subject, "D:/recordings/test.flac", "")

        self.assertEqual(indicator.hide_count, 0)

    def test_stop_recording_marks_indicator_finished_immediately(self):
        subject, indicator = self.make_subject(show_indicator=True)
        subject.recorder = SimpleNamespace(stop=lambda: setattr(subject, "stopped", True))
        subject.stopped = False

        TrayApplication.stop_recording(subject)

        self.assertTrue(subject.stopped)
        self.assertEqual(indicator.finished_count, 1)
        self.assertEqual(indicator.finished_hide_delay_ms, 5000)
        self.assertEqual(indicator.hide_count, 0)


if __name__ == "__main__":
    unittest.main()
