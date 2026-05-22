import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QObject, Qt
from PyQt6.QtWidgets import QApplication, QGroupBox, QMenu

from app_metadata import SETTINGS_WINDOW_TITLE, TRAY_IDLE_TOOLTIP
from gui import (
    KeyboardHotkeyManager,
    RecordingIndicator,
    SettingsWindow,
    TrayApplication,
    WindowsLowLevelHotkeyManager,
    parse_windows_hotkey,
)


class FakeSettingsWindow:
    def __init__(self, settings):
        self.settings = settings

    def get_settings(self):
        return self.settings


class FakeHotkeyManager:
    def __init__(self):
        self.cleared = False
        self.registrations = []

    def clear(self):
        self.cleared = True

    def register(self, hotkey, callback):
        self.registrations.append((hotkey, callback))
        return True


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


class WindowsHotkeyParserTests(unittest.TestCase):
    def test_parse_alt_shift_letter_for_low_level_manager(self):
        self.assertEqual(parse_windows_hotkey("alt+shift+r"), (0x0001 | 0x0004, 0x52))

    def test_parse_common_keys_for_low_level_manager(self):
        self.assertEqual(parse_windows_hotkey("ctrl+alt+s"), (0x0002 | 0x0001, 0x53))
        self.assertEqual(parse_windows_hotkey("win+space"), (0x0008, 0x20))
        self.assertEqual(parse_windows_hotkey("shift+f12"), (0x0004, 0x7B))

    def test_parse_unknown_or_ambiguous_hotkey_returns_none(self):
        self.assertIsNone(parse_windows_hotkey("alt+shift+unknown-key"))
        self.assertIsNone(parse_windows_hotkey("ctrl+alt+r+s"))


class WindowsLowLevelHotkeyManagerTests(unittest.TestCase):
    def test_alt_shift_letter_triggers_once_until_keyup(self):
        manager = WindowsLowLevelHotkeyManager(install_hook=False)
        calls = []
        manager.register("alt+shift+r", lambda: calls.append("mic"))

        manager.handle_key_down(0xA4)
        manager.handle_key_down(0xA0)
        manager.handle_key_down(0x52)
        manager.handle_key_down(0x52)
        manager.handle_key_up(0x52)
        manager.handle_key_down(0x52)

        self.assertEqual(calls, ["mic", "mic"])

    def test_clear_removes_low_level_registrations(self):
        manager = WindowsLowLevelHotkeyManager(install_hook=False)
        calls = []
        manager.register("alt+shift+r", lambda: calls.append("mic"))
        manager.clear()

        manager.handle_key_down(0xA4)
        manager.handle_key_down(0xA0)
        manager.handle_key_down(0x52)

        self.assertEqual(calls, [])


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

    def test_notification_setting_defaults_on(self):
        window = self.make_window({})

        settings = window.get_settings()

        self.assertIs(settings["show_notifications"], True)

    def test_notification_setting_can_be_disabled(self):
        window = self.make_window({"show_notifications": False})

        settings = window.get_settings()

        self.assertIs(settings["show_notifications"], False)

    def test_settings_window_title_uses_meetrec_name(self):
        window = self.make_window({})

        self.assertEqual(window.windowTitle(), SETTINGS_WINDOW_TITLE)

    def test_settings_window_uses_app_icon(self):
        window = self.make_window({})

        self.assertFalse(window.windowIcon().isNull())

    def test_auto_stop_setting_defaults_to_ten_minutes(self):
        window = self.make_window({})

        settings = window.get_settings()

        self.assertEqual(settings["auto_stop_silence_seconds"], 600)

    def test_auto_stop_setting_can_be_disabled(self):
        window = self.make_window({"auto_stop_silence_seconds": None})

        settings = window.get_settings()

        self.assertIsNone(settings["auto_stop_silence_seconds"])

    def test_auto_stop_setting_loads_five_minutes(self):
        window = self.make_window({"auto_stop_silence_seconds": 300})

        settings = window.get_settings()

        self.assertEqual(settings["auto_stop_silence_seconds"], 300)

    def test_trim_silence_setting_defaults_off(self):
        window = self.make_window({})

        settings = window.get_settings()

        self.assertIs(settings["trim_silence"], False)

    def test_trim_silence_setting_can_be_enabled(self):
        window = self.make_window({"trim_silence": True})

        settings = window.get_settings()

        self.assertIs(settings["trim_silence"], True)

    def test_trim_silence_setting_is_in_post_processing_group(self):
        window = self.make_window({})

        post_group = window.findChild(QGroupBox, "postProcessingSettingsGroup")

        self.assertIsNotNone(post_group)
        self.assertIs(window.chk_trim_silence.parentWidget(), post_group)

    def test_launch_at_startup_setting_defaults_off(self):
        window = self.make_window({})

        settings = window.get_settings()

        self.assertIs(settings["launch_at_startup"], False)

    def test_launch_at_startup_setting_can_be_enabled(self):
        window = self.make_window({"launch_at_startup": True})

        settings = window.get_settings()

        self.assertIs(settings["launch_at_startup"], True)

    def test_general_group_contains_startup_auto_stop_and_indicator_settings(self):
        window = self.make_window({})

        general_group = window.findChild(QGroupBox, "generalSettingsGroup")
        notifications_group = window.findChild(QGroupBox, "notificationsSettingsGroup")

        self.assertIsNotNone(general_group)
        self.assertIsNotNone(notifications_group)
        self.assertIs(window.chk_launch_at_startup.parentWidget(), general_group)
        self.assertIs(window.combo_auto_stop.parentWidget(), general_group)
        self.assertIs(window.chk_recording_indicator.parentWidget(), general_group)
        self.assertIs(window.chk_notifications.parentWidget(), notifications_group)

    def test_general_group_places_floating_timer_before_auto_stop(self):
        window = self.make_window({})

        general_layout = window.findChild(QGroupBox, "generalSettingsGroup").layout()
        startup_row, _startup_role = general_layout.getWidgetPosition(
            window.chk_launch_at_startup
        )
        indicator_row, _indicator_role = general_layout.getWidgetPosition(
            window.chk_recording_indicator
        )
        auto_stop_row, _auto_stop_role = general_layout.getWidgetPosition(
            window.combo_auto_stop
        )

        self.assertLess(startup_row, indicator_row)
        self.assertLess(indicator_row, auto_stop_row)

    def test_legacy_stop_hotkey_keeps_dedicated_stop_enabled(self):
        window = self.make_window({"hk_stop": "ctrl+alt+s"})

        self.assertFalse(window.chk_stop_with_record_hotkeys.isChecked())
        self.assertTrue(window.hk_stop.isEnabled())
        self.assertEqual(window.hk_stop.text(), "ctrl+alt+s")

    def test_legacy_settings_without_stop_hotkey_use_record_hotkeys_to_stop(self):
        window = self.make_window({})

        self.assertTrue(window.chk_stop_with_record_hotkeys.isChecked())
        self.assertFalse(window.hk_stop.isEnabled())

    def test_explicit_stop_with_record_hotkeys_false_keeps_dedicated_stop_enabled(self):
        window = self.make_window(
            {"hk_stop": "ctrl+alt+s", "stop_with_record_hotkeys": False}
        )

        self.assertFalse(window.chk_stop_with_record_hotkeys.isChecked())
        self.assertTrue(window.hk_stop.isEnabled())

    def test_explicit_stop_with_record_hotkeys_true_overrides_legacy_stop_hotkey(self):
        window = self.make_window(
            {"hk_stop": "ctrl+alt+s", "stop_with_record_hotkeys": True}
        )

        self.assertTrue(window.chk_stop_with_record_hotkeys.isChecked())
        self.assertFalse(window.hk_stop.isEnabled())

    def test_save_settings_applies_startup_preference(self):
        window = self.make_window({"launch_at_startup": True})

        with patch("gui.set_launch_at_startup_enabled") as set_startup, patch("gui.QMessageBox.information"):
            window.save_settings()

        set_startup.assert_called_once_with(True)


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


class TrayApplicationHotkeyTests(unittest.TestCase):
    def test_register_hotkeys_uses_app_hotkey_manager(self):
        hotkey_manager = FakeHotkeyManager()
        subject = SimpleNamespace(
            hotkey_manager=hotkey_manager,
            settings_window=FakeSettingsWindow(
                {
                    "hk_mic": "alt+shift+r",
                    "hk_loop": "ctrl+shift+l",
                    "hk_both": "",
                    "hk_stop": "ctrl+shift+s",
                    "stop_with_record_hotkeys": False,
                }
            ),
            toggled=[],
            stopped=False,
        )
        subject.toggle_recording = lambda mode: subject.toggled.append(mode)
        subject.stop_recording = lambda: setattr(subject, "stopped", True)

        with patch("gui.keyboard.add_hotkey") as add_hotkey, patch(
            "gui.keyboard.unhook_all_hotkeys"
        ) as unhook_all_hotkeys:
            TrayApplication.register_hotkeys(subject)

        self.assertTrue(hotkey_manager.cleared)
        self.assertEqual(
            [item[0] for item in hotkey_manager.registrations],
            ["alt+shift+r", "ctrl+shift+l", "ctrl+shift+s"],
        )
        self.assertFalse(add_hotkey.called)
        self.assertFalse(unhook_all_hotkeys.called)

        hotkey_manager.registrations[0][1]()
        hotkey_manager.registrations[2][1]()

        self.assertEqual(subject.toggled, ["mic"])
        self.assertTrue(subject.stopped)

    def test_record_hotkey_stops_active_recording_when_option_enabled(self):
        subject = SimpleNamespace(
            recorder=SimpleNamespace(is_alive=lambda: True),
            settings_window=FakeSettingsWindow({"stop_with_record_hotkeys": True}),
            stopped=False,
            started=None,
        )
        subject.stop_recording = lambda: setattr(subject, "stopped", True)
        subject.start_recording = lambda mode: setattr(subject, "started", mode)

        TrayApplication.toggle_recording(subject, "mic")

        self.assertTrue(subject.stopped)
        self.assertIsNone(subject.started)

    def test_record_hotkey_does_not_switch_mode_when_option_disabled(self):
        subject = SimpleNamespace(
            recorder=SimpleNamespace(is_alive=lambda: True),
            settings_window=FakeSettingsWindow({"stop_with_record_hotkeys": False}),
            stopped=False,
            started=None,
        )
        subject.stop_recording = lambda: setattr(subject, "stopped", True)
        subject.start_recording = lambda mode: setattr(subject, "started", mode)

        TrayApplication.toggle_recording(subject, "loopback")

        self.assertFalse(subject.stopped)
        self.assertIsNone(subject.started)

    def test_record_hotkey_starts_recording_when_idle(self):
        subject = SimpleNamespace(
            recorder=None,
            settings_window=FakeSettingsWindow({"stop_with_record_hotkeys": True}),
            stopped=False,
            started=None,
        )
        subject.stop_recording = lambda: setattr(subject, "stopped", True)
        subject.start_recording = lambda mode: setattr(subject, "started", mode)

        TrayApplication.toggle_recording(subject, "both")

        self.assertFalse(subject.stopped)
        self.assertEqual(subject.started, "both")


class TrayApplicationNotificationTests(unittest.TestCase):
    def test_notification_is_skipped_when_disabled(self):
        subject = SimpleNamespace(
            tray_icon=FakeTrayIcon(),
            settings_window=FakeSettingsWindow({"show_notifications": False}),
        )

        TrayApplication.show_tray_notification(subject, "Started", "Recording mic")

        self.assertEqual(subject.tray_icon.messages, [])

    def test_notification_is_sent_when_enabled(self):
        subject = SimpleNamespace(
            tray_icon=FakeTrayIcon(),
            settings_window=FakeSettingsWindow({"show_notifications": True}),
        )

        TrayApplication.show_tray_notification(
            subject, "Started", "Recording mic", duration=1234
        )

        self.assertEqual(len(subject.tray_icon.messages), 1)
        self.assertEqual(subject.tray_icon.messages[0][0], "Started")
        self.assertEqual(subject.tray_icon.messages[0][1], "Recording mic")
        self.assertEqual(subject.tray_icon.messages[0][3], 1234)


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
                    "trim_silence": True,
                    "auto_stop_silence_seconds": 600,
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
            show_tray_notification=lambda *args, **kwargs: None,
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

    def test_start_recording_passes_auto_stop_setting(self):
        subject, _indicator = self.make_subject(show_indicator=True)

        with patch("gui.QIcon"), patch("gui.AudioRecorder") as AudioRecorder:
            TrayApplication.start_recording(subject, "both")

        self.assertEqual(AudioRecorder.call_args.kwargs["auto_stop_silence_seconds"], 600)

    def test_start_recording_passes_trim_silence_setting(self):
        subject, _indicator = self.make_subject(show_indicator=True)

        with patch("gui.QIcon"), patch("gui.AudioRecorder") as AudioRecorder:
            TrayApplication.start_recording(subject, "both")

        self.assertEqual(AudioRecorder.call_args.kwargs["trim_silence"], True)

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
