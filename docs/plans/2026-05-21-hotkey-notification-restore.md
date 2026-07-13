# Hotkey Notification Restore Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore the missing PR #3 hotkey and tray-notification behavior in the current MeetRec codebase.

**Architecture:** Keep the existing single-file GUI structure and add the missing hotkey abstraction back into `gui.py`. Settings continue to flow through `SettingsWindow.get_settings()`, while `TrayApplication` consumes those settings through a hotkey manager and a notification wrapper.

**Tech Stack:** Python, PyQt6, `keyboard`, Windows low-level keyboard hook via `ctypes`, `unittest`.

---

## File Structure

- `gui.py`: add `parse_windows_hotkey`, `KeyboardHotkeyManager`, `WindowsLowLevelHotkeyManager`, `create_hotkey_manager`; restore `show_notifications`, `stop_with_record_hotkeys`, `toggle_recording`, and notification wrapper behavior.
- `tests/test_gui_hotkeys.py`: add regression tests for parser, low-level manager, settings migration, registration routing, toggle behavior, and notification suppression.
- `README.md`: document notification toggle and record-hotkeys-to-stop behavior.

### Task 1: Regression Tests For Missing PR #3 Behavior

**Files:**
- Modify: `tests/test_gui_hotkeys.py`

- [ ] **Step 1: Add imports and fake manager support**

```python
from gui import (
    KeyboardHotkeyManager,
    RecordingIndicator,
    SettingsWindow,
    TrayApplication,
    WindowsLowLevelHotkeyManager,
    parse_windows_hotkey,
)

class FakeHotkeyManager:
    def __init__(self):
        self.cleared = False
        self.registrations = []

    def clear(self):
        self.cleared = True

    def register(self, hotkey, callback):
        self.registrations.append((hotkey, callback))
        return True
```

- [ ] **Step 2: Add parser and low-level manager tests**

```python
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
```

- [ ] **Step 3: Add settings, registration, toggle, and notification tests**

```python
def test_notification_setting_defaults_on(self):
    window = self.make_window({})
    self.assertIs(window.get_settings()["show_notifications"], True)

def test_legacy_stop_hotkey_keeps_dedicated_stop_enabled(self):
    window = self.make_window({"hk_stop": "ctrl+alt+s"})
    self.assertFalse(window.chk_stop_with_record_hotkeys.isChecked())
    self.assertTrue(window.hk_stop.isEnabled())

def test_legacy_settings_without_stop_hotkey_use_record_hotkeys_to_stop(self):
    window = self.make_window({})
    self.assertTrue(window.chk_stop_with_record_hotkeys.isChecked())
    self.assertFalse(window.hk_stop.isEnabled())
```

```python
class TrayApplicationHotkeyTests(unittest.TestCase):
    def test_register_hotkeys_uses_app_hotkey_manager(self):
        hotkey_manager = FakeHotkeyManager()
        subject = SimpleNamespace(
            hotkey_manager=hotkey_manager,
            settings_window=FakeSettingsWindow({
                "hk_mic": "alt+shift+r",
                "hk_loop": "ctrl+shift+l",
                "hk_both": "",
                "hk_stop": "ctrl+shift+s",
                "stop_with_record_hotkeys": False,
            }),
            toggled=[],
            stopped=False,
        )
        subject.toggle_recording = lambda mode: subject.toggled.append(mode)
        subject.stop_recording = lambda: setattr(subject, "stopped", True)

        with patch("gui.keyboard.add_hotkey") as add_hotkey, patch("gui.keyboard.unhook_all_hotkeys") as unhook_all_hotkeys:
            TrayApplication.register_hotkeys(subject)

        self.assertTrue(hotkey_manager.cleared)
        self.assertEqual([item[0] for item in hotkey_manager.registrations], ["alt+shift+r", "ctrl+shift+l", "ctrl+shift+s"])
        self.assertFalse(add_hotkey.called)
        self.assertFalse(unhook_all_hotkeys.called)

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
```

```python
class TrayApplicationNotificationTests(unittest.TestCase):
    def test_notification_is_skipped_when_disabled(self):
        subject = SimpleNamespace(
            tray_icon=FakeTrayIcon(),
            settings_window=FakeSettingsWindow({"show_notifications": False}),
        )

        TrayApplication.show_tray_notification(subject, "Started", "Recording mic")

        self.assertEqual(subject.tray_icon.messages, [])
```

- [ ] **Step 4: Run tests to verify RED**

Run: `python -m unittest tests.test_gui_hotkeys`

Expected: import errors or assertion failures for missing `parse_windows_hotkey`, `WindowsLowLevelHotkeyManager`, `chk_stop_with_record_hotkeys`, and `show_tray_notification`.

### Task 2: Restore Hotkey Manager And Settings Behavior

**Files:**
- Modify: `gui.py`

- [ ] **Step 1: Add Windows hotkey parser and managers**

```python
WINDOWS_MODIFIER_KEYS = {"alt": 0x0001, "ctrl": 0x0002, "control": 0x0002, "shift": 0x0004, "windows": 0x0008, "win": 0x0008}
WINDOWS_SPECIAL_KEYS = {"space": 0x20, "esc": 0x1B, "escape": 0x1B, "enter": 0x0D, "return": 0x0D}

def parse_windows_hotkey(hotkey):
    ...

class KeyboardHotkeyManager:
    def clear(self):
        keyboard.unhook_all_hotkeys()

    def register(self, hotkey, callback):
        keyboard.add_hotkey(hotkey, callback)
        return True

class WindowsLowLevelHotkeyManager:
    def register(self, hotkey, callback):
        parsed = parse_windows_hotkey(hotkey)
        if parsed is None:
            return self.fallback.register(hotkey, callback)
        self.callbacks.setdefault(parsed, []).append(callback)
        return True
```

- [ ] **Step 2: Add notification and stop-hotkey settings**

```python
self.chk_notifications = QCheckBox("Show tray notifications")
self.chk_notifications.setChecked(True)
self.chk_stop_with_record_hotkeys = QCheckBox("Use record hotkeys to stop recording")
self.chk_stop_with_record_hotkeys.toggled.connect(self.update_stop_hotkey_state)
```

Persist keys:

```python
"show_notifications": self.chk_notifications.isChecked(),
"stop_with_record_hotkeys": self.chk_stop_with_record_hotkeys.isChecked(),
```

- [ ] **Step 3: Route hotkeys through manager and toggle behavior**

```python
if hk_mic:
    hotkey_manager.register(hk_mic, lambda: self.toggle_recording("mic"))
if hk_stop and not settings.get("stop_with_record_hotkeys", True):
    hotkey_manager.register(hk_stop, self.stop_recording)
```

- [ ] **Step 4: Route tray messages through `show_tray_notification`**

```python
def show_tray_notification(self, title, message, icon=QSystemTrayIcon.MessageIcon.Information, duration=2000):
    if self.notifications_enabled():
        self.tray_icon.showMessage(title, message, icon, duration)
```

- [ ] **Step 5: Run tests to verify GREEN**

Run: `python -m unittest tests.test_gui_hotkeys`

Expected: all tests in `tests.test_gui_hotkeys` pass.

### Task 3: Documentation And Full Verification

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Document restored behavior**

Add concise bullets under control/settings usage:

```markdown
- Global Hotkeys: Start or stop recording from anywhere, including Alt+Shift combinations.
- Notification Toggle: Turn tray balloon notifications on or off from Settings.
- Enable "Use record hotkeys to stop recording" if you want any record hotkey to stop the active recording.
```

- [ ] **Step 2: Run focused and full tests**

Run: `python -m unittest tests.test_gui_hotkeys tests.test_audio_output_profile tests.test_silence_auto_stop tests.test_app_metadata tests.test_product_identity_text`

Expected: all listed test modules pass.

- [ ] **Step 3: Inspect git diff**

Run: `git diff --stat`

Expected: changes limited to `gui.py`, `tests/test_gui_hotkeys.py`, `README.md`, this plan, plus pre-existing user changes.
