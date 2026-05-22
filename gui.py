import sys
import os
import json
import shutil
import tempfile
import subprocess
import ctypes
from ctypes import wintypes
from PyQt6.QtWidgets import (QApplication, QSystemTrayIcon, QMenu, QMainWindow, 
                             QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, 
                             QPushButton, QFileDialog, QMessageBox, QGroupBox, 
                             QLineEdit, QFormLayout, QCheckBox)
from PyQt6.QtGui import QIcon, QAction, QKeySequence
from PyQt6.QtCore import pyqtSignal, QObject, Qt, QUrl, QMimeData, QDir, QTimer, QEvent
import soundcard as sc
import keyboard
from app_metadata import (
    APP_NAME,
    APP_ICON_FILENAME,
    ICON_IDLE_FILENAME,
    ICON_RECORDING_FILENAME,
    READY_MESSAGE_BODY,
    READY_MESSAGE_TITLE,
    SETTINGS_WINDOW_TITLE,
    TRAY_IDLE_TOOLTIP,
)
from audio_recorder import (
    AUTO_STOP_DEFAULT_SECONDS,
    AUTO_STOP_OPTIONS,
    AudioRecorder,
    FORMAT_CONFIG,
    QUALITY_CONFIG,
    describe_output_profile,
    get_devices,
    normalize_auto_stop_silence_seconds,
)
from clipboard_utils import copy_file_to_clipboard

CONFIG_FILE = "settings.json"

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


STARTUP_REGISTRY_PATH = r"Software\Microsoft\Windows\CurrentVersion\Run"


def _quote_command_part(value):
    return f'"{value}"'


def build_startup_command(executable=None, script_path=None, frozen=None):
    is_frozen = getattr(sys, "frozen", False) if frozen is None else frozen
    executable_path = os.path.abspath(executable or sys.executable)
    if is_frozen:
        return _quote_command_part(executable_path)

    script = script_path or (sys.argv[0] if sys.argv else "main.py")
    script = os.path.abspath(script)
    return f"{_quote_command_part(executable_path)} {_quote_command_part(script)}"


def set_launch_at_startup_enabled(enabled):
    if sys.platform != "win32":
        return True

    import winreg

    try:
        with winreg.CreateKeyEx(
            winreg.HKEY_CURRENT_USER,
            STARTUP_REGISTRY_PATH,
            0,
            winreg.KEY_SET_VALUE,
        ) as key:
            if enabled:
                winreg.SetValueEx(key, APP_NAME, 0, winreg.REG_SZ, build_startup_command())
            else:
                try:
                    winreg.DeleteValue(key, APP_NAME)
                except FileNotFoundError:
                    pass
        return True
    except OSError as e:
        print(f"Failed to update startup setting: {e}")
        return False


WINDOWS_MODIFIER_KEYS = {
    "alt": 0x0001,
    "ctrl": 0x0002,
    "control": 0x0002,
    "shift": 0x0004,
    "windows": 0x0008,
    "win": 0x0008,
}

WINDOWS_SPECIAL_KEYS = {
    "backspace": 0x08,
    "tab": 0x09,
    "enter": 0x0D,
    "return": 0x0D,
    "esc": 0x1B,
    "escape": 0x1B,
    "space": 0x20,
    "left": 0x25,
    "up": 0x26,
    "right": 0x27,
    "down": 0x28,
    "delete": 0x2E,
    "plus": 0xBB,
    "comma": 0xBC,
    "-": 0xBD,
    "minus": 0xBD,
    ".": 0xBE,
    "period": 0xBE,
    "/": 0xBF,
    "slash": 0xBF,
}

for number in range(1, 13):
    WINDOWS_SPECIAL_KEYS[f"f{number}"] = 0x70 + number - 1


def parse_windows_hotkey(hotkey):
    parts = [part.strip().lower() for part in (hotkey or "").split("+") if part.strip()]
    if not parts:
        return None

    modifiers = 0
    keys = []
    for part in parts:
        modifier = WINDOWS_MODIFIER_KEYS.get(part)
        if modifier:
            modifiers |= modifier
        else:
            keys.append(part)

    if len(keys) != 1:
        return None

    key = keys[0]
    if len(key) == 1 and "a" <= key <= "z":
        virtual_key = ord(key.upper())
    elif len(key) == 1 and "0" <= key <= "9":
        virtual_key = ord(key)
    else:
        virtual_key = WINDOWS_SPECIAL_KEYS.get(key)

    if not virtual_key:
        return None

    return modifiers, virtual_key


class KBDLLHOOKSTRUCT(ctypes.Structure):
    _fields_ = [
        ("vkCode", wintypes.DWORD),
        ("scanCode", wintypes.DWORD),
        ("flags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.c_void_p),
    ]


LowLevelKeyboardProcFactory = getattr(ctypes, "WINFUNCTYPE", ctypes.CFUNCTYPE)
LowLevelKeyboardProc = LowLevelKeyboardProcFactory(
    wintypes.LPARAM, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM
)


class KeyboardHotkeyManager:
    def clear(self):
        try:
            keyboard.unhook_all_hotkeys()
        except Exception:
            pass

    def register(self, hotkey, callback):
        keyboard.add_hotkey(hotkey, callback)
        return True


class WindowsLowLevelHotkeyManager:
    WH_KEYBOARD_LL = 13
    WM_KEYDOWN = 0x0100
    WM_KEYUP = 0x0101
    WM_SYSKEYDOWN = 0x0104
    WM_SYSKEYUP = 0x0105
    KEY_DOWN_MESSAGES = {WM_KEYDOWN, WM_SYSKEYDOWN}
    KEY_UP_MESSAGES = {WM_KEYUP, WM_SYSKEYUP}
    VK_TO_MODIFIER = {
        0x10: WINDOWS_MODIFIER_KEYS["shift"],
        0xA0: WINDOWS_MODIFIER_KEYS["shift"],
        0xA1: WINDOWS_MODIFIER_KEYS["shift"],
        0x11: WINDOWS_MODIFIER_KEYS["ctrl"],
        0xA2: WINDOWS_MODIFIER_KEYS["ctrl"],
        0xA3: WINDOWS_MODIFIER_KEYS["ctrl"],
        0x12: WINDOWS_MODIFIER_KEYS["alt"],
        0xA4: WINDOWS_MODIFIER_KEYS["alt"],
        0xA5: WINDOWS_MODIFIER_KEYS["alt"],
        0x5B: WINDOWS_MODIFIER_KEYS["windows"],
        0x5C: WINDOWS_MODIFIER_KEYS["windows"],
    }

    def __init__(self, install_hook=True, fallback=None):
        self.fallback = fallback or KeyboardHotkeyManager()
        self.callbacks = {}
        self.active_modifiers = 0
        self.active_hotkeys = set()
        self.hook = None
        self.user32 = None
        self.kernel32 = None
        self.hook_callback = None
        if sys.platform == "win32":
            self.user32 = ctypes.WinDLL("user32", use_last_error=True)
            self.kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            self.configure_api()
            self.hook_callback = LowLevelKeyboardProc(self.low_level_keyboard_proc)
            if install_hook:
                self.install_hook()

    def configure_api(self):
        self.user32.SetWindowsHookExW.argtypes = [
            ctypes.c_int,
            LowLevelKeyboardProc,
            wintypes.HINSTANCE,
            wintypes.DWORD,
        ]
        self.user32.SetWindowsHookExW.restype = wintypes.HHOOK
        self.user32.UnhookWindowsHookEx.argtypes = [wintypes.HHOOK]
        self.user32.UnhookWindowsHookEx.restype = wintypes.BOOL
        self.user32.CallNextHookEx.argtypes = [
            wintypes.HHOOK,
            ctypes.c_int,
            wintypes.WPARAM,
            wintypes.LPARAM,
        ]
        self.user32.CallNextHookEx.restype = wintypes.LPARAM
        self.kernel32.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
        self.kernel32.GetModuleHandleW.restype = wintypes.HMODULE

    def install_hook(self):
        if self.user32 is None or self.hook:
            return bool(self.hook)

        self.hook = self.user32.SetWindowsHookExW(
            self.WH_KEYBOARD_LL,
            self.hook_callback,
            self.kernel32.GetModuleHandleW(None),
            0,
        )
        if not self.hook:
            print(f"Failed to install low-level hotkey hook: {ctypes.get_last_error()}")
        return bool(self.hook)

    def clear(self):
        self.callbacks.clear()
        self.active_modifiers = 0
        self.active_hotkeys.clear()
        try:
            self.fallback.clear()
        except Exception as e:
            print(f"Failed to clear fallback hotkeys: {e}")

    def register(self, hotkey, callback):
        parsed = parse_windows_hotkey(hotkey)
        if parsed is None:
            return self.fallback.register(hotkey, callback)

        if self.user32 is not None and not self.install_hook():
            return self.fallback.register(hotkey, callback)

        self.callbacks.setdefault(parsed, []).append(callback)
        return True

    def low_level_keyboard_proc(self, n_code, w_param, l_param):
        try:
            if n_code >= 0:
                event = ctypes.cast(l_param, ctypes.POINTER(KBDLLHOOKSTRUCT)).contents
                self.process_key_event(int(w_param), int(event.vkCode))
        except Exception as e:
            print(f"Failed to handle low-level hotkey event: {e}")
        return self.user32.CallNextHookEx(None, n_code, w_param, l_param)

    def process_key_event(self, message, virtual_key):
        if message in self.KEY_DOWN_MESSAGES:
            self.handle_key_down(virtual_key)
        elif message in self.KEY_UP_MESSAGES:
            self.handle_key_up(virtual_key)

    def handle_key_down(self, virtual_key):
        modifier = self.VK_TO_MODIFIER.get(virtual_key)
        if modifier:
            self.active_modifiers |= modifier
            return

        hotkey = (self.active_modifiers, virtual_key)
        if hotkey in self.callbacks and hotkey not in self.active_hotkeys:
            self.active_hotkeys.add(hotkey)
            for callback in list(self.callbacks[hotkey]):
                callback()

    def handle_key_up(self, virtual_key):
        modifier = self.VK_TO_MODIFIER.get(virtual_key)
        if modifier:
            self.active_modifiers &= ~modifier
            self.active_hotkeys.clear()
            return

        for hotkey in list(self.active_hotkeys):
            if hotkey[1] == virtual_key:
                self.active_hotkeys.discard(hotkey)


def create_hotkey_manager(app):
    if sys.platform == "win32":
        return WindowsLowLevelHotkeyManager()
    return KeyboardHotkeyManager()


class SignalManager(QObject):
    recording_finished = pyqtSignal(str, str)


class RecordingIndicator(QWidget):
    FINISHED_HIDE_DELAY_MS = 5000

    stop_requested = pyqtSignal()
    open_folder_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(
            parent,
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowStaysOnTopHint,
        )
        self.elapsed_seconds = 0
        self._drag_offset = None
        self._press_global_pos = None
        self._dragged = False
        self.is_finishing = False
        self.context_menu = None

        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedSize(92, 34)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setWindowTitle("Recording")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.container = QWidget(self)
        self.container.setObjectName("recordingIndicatorContainer")
        self.container.setStyleSheet(
            """
            QWidget#recordingIndicatorContainer {
                background-color: rgba(24, 24, 24, 220);
                border-radius: 17px;
            }
            """
        )

        inner_layout = QHBoxLayout(self.container)
        inner_layout.setContentsMargins(13, 0, 13, 0)
        inner_layout.setSpacing(8)

        self.dot = QLabel(self.container)
        self.dot.setFixedSize(9, 9)

        self.timer_label = QLabel("00:00", self.container)
        self.timer_label.setStyleSheet(
            "color: white; font-size: 13px; font-weight: 600;"
        )
        self.timer_label.setAlignment(Qt.AlignmentFlag.AlignVCenter)

        inner_layout.addWidget(self.dot)
        inner_layout.addWidget(self.timer_label)
        layout.addWidget(self.container)

        self.timer = QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.update_elapsed)
        self.finished_hide_timer = QTimer(self)
        self.finished_hide_timer.setSingleShot(True)
        self.finished_hide_timer.timeout.connect(self.hide_recording)
        self.set_recording_style()

    def set_recording_style(self):
        self.dot.setStyleSheet("background-color: #ff2d2d; border-radius: 4px;")
        self.container.setStyleSheet(
            """
            QWidget#recordingIndicatorContainer {
                background-color: rgba(24, 24, 24, 220);
                border-radius: 17px;
            }
            """
        )

    def set_finished_style(self):
        self.dot.setStyleSheet("background-color: #8a8a8a; border-radius: 4px;")
        self.container.setStyleSheet(
            """
            QWidget#recordingIndicatorContainer {
                background-color: rgba(24, 24, 24, 190);
                border-radius: 17px;
            }
            """
        )

    @staticmethod
    def format_elapsed(seconds):
        seconds = max(0, int(seconds))
        minutes, remaining_seconds = divmod(seconds, 60)
        return f"{minutes:02d}:{remaining_seconds:02d}"

    def show_recording(self):
        self.finished_hide_timer.stop()
        self.is_finishing = False
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.elapsed_seconds = 0
        self.update_timer_text()
        self.set_recording_style()
        self.position_near_taskbar()
        self.show()
        self.raise_()
        self.timer.start()

    def show_finished(self, hide_after_ms=None):
        self.timer.stop()
        self.is_finishing = True
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.set_finished_style()
        self.raise_()
        delay_ms = self.FINISHED_HIDE_DELAY_MS if hide_after_ms is None else hide_after_ms
        self.finished_hide_timer.start(delay_ms)

    def hide_recording(self):
        self.timer.stop()
        self.finished_hide_timer.stop()
        self.is_finishing = False
        self.hide()

    def update_elapsed(self):
        self.elapsed_seconds += 1
        self.update_timer_text()

    def update_timer_text(self):
        self.timer_label.setText(self.format_elapsed(self.elapsed_seconds))

    def position_near_taskbar(self):
        screen = QApplication.primaryScreen()
        if not screen:
            return
        geometry = screen.availableGeometry()
        x = geometry.right() - self.width() - 16
        y = geometry.bottom() - self.height() - 16
        self.move(x, y)

    def request_stop(self):
        if self.is_finishing:
            return
        self.stop_requested.emit()

    def request_open_folder(self):
        if self.is_finishing:
            self.open_folder_requested.emit()

    def setContextMenu(self, menu):
        self.context_menu = menu

    def contextMenuEvent(self, event):
        if self.context_menu:
            self.context_menu.exec(event.globalPos())
            event.accept()
            return
        super().contextMenuEvent(event)

    def mousePressEvent(self, event):
        if self.is_finishing:
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton:
            global_pos = event.globalPosition().toPoint()
            self._press_global_pos = global_pos
            self._drag_offset = global_pos - self.frameGeometry().topLeft()
            self._dragged = False
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.is_finishing:
            event.accept()
            return
        if self._drag_offset is not None:
            global_pos = event.globalPosition().toPoint()
            if self._press_global_pos is not None:
                delta = global_pos - self._press_global_pos
                self._dragged = self._dragged or abs(delta.x()) > 3 or abs(delta.y()) > 3
            self.move(global_pos - self._drag_offset)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.is_finishing:
            if event.button() == Qt.MouseButton.LeftButton:
                self.request_open_folder()
            self._drag_offset = None
            self._press_global_pos = None
            self._dragged = False
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton:
            if not self._dragged:
                self.request_stop()
            self._drag_offset = None
            self._press_global_pos = None
            self._dragged = False
            event.accept()
            return
        super().mouseReleaseEvent(event)


class HotkeyEdit(QLineEdit):
    """
    Custom widget to capture hotkeys by pressing them.
    Maps Qt events to 'keyboard' library compatible strings.
    """
    CAPTURE_PROMPT = "Press shortcut..."
    sequence_captured = pyqtSignal(str)
    capture_cancelled = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setPlaceholderText("Click to set hotkey...")
        self.setReadOnly(True)
        self.current_sequence = None
        self.is_capturing = False
        self._previous_text = ""
        self._keyboard_hook = None
        self._modifier_scan_codes = {
            "ctrl": set(),
            "alt": set(),
            "shift": set(),
            "windows": set(),
        }
        self._modifier_names_by_scan_code = self.build_modifier_scan_code_lookup()
        self.sequence_captured.connect(self.finish_capture)
        self.capture_cancelled.connect(self.cancel_capture)

    def begin_capture(self):
        if self.is_capturing:
            return
        self.is_capturing = True
        self._previous_text = self.text()
        self.setText(self.CAPTURE_PROMPT)
        self.selectAll()
        self.setStyleSheet("color: #666;")
        self.start_keyboard_capture()

    def finish_capture(self, sequence):
        self.stop_keyboard_capture()
        self.is_capturing = False
        self.current_sequence = sequence or None
        self.setStyleSheet("")
        self.setText(sequence)
        self.clearFocus()

    def cancel_capture(self):
        self.stop_keyboard_capture()
        self.is_capturing = False
        self.setStyleSheet("")
        self.setText(self._previous_text)
        self.clearFocus()

    def mousePressEvent(self, event):
        self.setFocus()
        self.begin_capture()
        super().mousePressEvent(event)

    def focusInEvent(self, event):
        super().focusInEvent(event)
        self.begin_capture()

    def focusOutEvent(self, event):
        if self.is_capturing:
            self.is_capturing = False
            self.stop_keyboard_capture()
            self.setStyleSheet("")
            self.setText(self._previous_text)
        super().focusOutEvent(event)

    def start_keyboard_capture(self):
        if self._keyboard_hook is not None:
            return
        for scan_codes in self._modifier_scan_codes.values():
            scan_codes.clear()
        try:
            self._keyboard_hook = keyboard.hook(self.handle_keyboard_hook)
        except Exception as e:
            print(f"Failed to start hotkey capture hook: {e}")

    def stop_keyboard_capture(self):
        if self._keyboard_hook is None:
            return
        try:
            keyboard.unhook(self._keyboard_hook)
        except Exception as e:
            print(f"Failed to stop hotkey capture hook: {e}")
        finally:
            self._keyboard_hook = None
            for scan_codes in self._modifier_scan_codes.values():
                scan_codes.clear()

    def handle_keyboard_hook(self, event):
        if not self.is_capturing:
            return

        key_name = self.normalize_hook_key_name(event.name)
        modifier = self.modifier_name_for_hook_event(key_name, event.scan_code)
        scan_code = event.scan_code

        if modifier:
            if event.event_type == "down":
                self._modifier_scan_codes[modifier].add(scan_code)
            elif event.event_type == "up":
                self._modifier_scan_codes[modifier].discard(scan_code)
            return

        if event.event_type != "down":
            return

        if key_name in ("esc", "escape"):
            self.capture_cancelled.emit()
            return

        if key_name in ("backspace", "delete"):
            self.sequence_captured.emit("")
            return

        sequence = self.format_hook_hotkey(key_name)
        if sequence:
            self.sequence_captured.emit(sequence)

    def normalize_hook_key_name(self, key_name):
        key_name = (key_name or "").lower()
        aliases = {
            "left windows": "windows",
            "right windows": "windows",
            "win": "windows",
            "cmd": "windows",
            "+": "plus",
            ",": "comma",
            " ": "space",
            "return": "enter",
        }
        return aliases.get(key_name, key_name)

    def modifier_name_for_hook_key(self, key_name):
        aliases = {
            "ctrl": "ctrl",
            "control": "ctrl",
            "left ctrl": "ctrl",
            "right ctrl": "ctrl",
            "alt": "alt",
            "left alt": "alt",
            "right alt": "alt",
            "shift": "shift",
            "left shift": "shift",
            "right shift": "shift",
            "windows": "windows",
            "left windows": "windows",
            "right windows": "windows",
        }
        return aliases.get(key_name)

    def modifier_name_for_hook_event(self, key_name, scan_code):
        if scan_code in self._modifier_names_by_scan_code:
            return self._modifier_names_by_scan_code[scan_code]
        return self.modifier_name_for_hook_key(key_name)

    def build_modifier_scan_code_lookup(self):
        lookup = {}
        modifier_names = {
            "ctrl": ("ctrl", "control", "left ctrl", "right ctrl"),
            "alt": ("alt", "left alt", "right alt"),
            "shift": ("shift", "left shift", "right shift"),
            "windows": ("windows", "left windows", "right windows"),
        }

        for modifier, names in modifier_names.items():
            for name in names:
                try:
                    scan_codes = keyboard.key_to_scan_codes(name, False)
                except Exception:
                    scan_codes = ()
                for scan_code in scan_codes:
                    lookup[scan_code] = modifier

        return lookup

    def format_hook_hotkey(self, key_name):
        parts = []
        for modifier in ("ctrl", "alt", "shift", "windows"):
            if self._modifier_scan_codes[modifier]:
                parts.append(modifier)

        key_text = self.normalize_hook_key_name(key_name)
        if not key_text or self.modifier_name_for_hook_key(key_text):
            return ""

        parts.append(key_text)
        return "+".join(parts)

    def event(self, event):
        if event.type() == QEvent.Type.ShortcutOverride and self.is_capturing:
            self.handle_hotkey_event(event)
            event.accept()
            return True
        return super().event(event)

    def keyPressEvent(self, event):
        self.handle_hotkey_event(event)

    def handle_hotkey_event(self, event):
        key = self.key_from_event(event)
        modifiers = event.modifiers()

        if key in (Qt.Key.Key_Backspace.value, Qt.Key.Key_Delete.value):
            self.finish_capture("")
            return

        if key == Qt.Key.Key_Escape.value:
            self.cancel_capture()
            return

        if key in (
            Qt.Key.Key_Control.value,
            Qt.Key.Key_Shift.value,
            Qt.Key.Key_Alt.value,
            Qt.Key.Key_Meta.value,
        ):
            return

        final_hotkey = self.format_hotkey(key, modifiers)
        if final_hotkey:
            self.finish_capture(final_hotkey)

    def key_from_event(self, event):
        key = event.key()
        if key == Qt.Key.Key_unknown.value and event.nativeVirtualKey():
            return event.nativeVirtualKey()
        return key

    def format_hotkey(self, key, modifiers):
        parts = []
        if modifiers & Qt.KeyboardModifier.ControlModifier:
            parts.append("ctrl")
        if modifiers & Qt.KeyboardModifier.AltModifier:
            parts.append("alt")
        if modifiers & Qt.KeyboardModifier.ShiftModifier:
            parts.append("shift")
        if modifiers & Qt.KeyboardModifier.MetaModifier:
            parts.append("windows")

        key_text = self.key_to_text(key)
        if not key_text:
            return ""

        parts.append(key_text)
        return "+".join(parts)

    def key_to_text(self, key):
        if Qt.Key.Key_A.value <= key <= Qt.Key.Key_Z.value:
            return chr(key).lower()

        if Qt.Key.Key_0.value <= key <= Qt.Key.Key_9.value:
            return chr(key)

        key_map = {
            Qt.Key.Key_F1.value: "f1",
            Qt.Key.Key_F2.value: "f2",
            Qt.Key.Key_F3.value: "f3",
            Qt.Key.Key_F4.value: "f4",
            Qt.Key.Key_F5.value: "f5",
            Qt.Key.Key_F6.value: "f6",
            Qt.Key.Key_F7.value: "f7",
            Qt.Key.Key_F8.value: "f8",
            Qt.Key.Key_F9.value: "f9",
            Qt.Key.Key_F10.value: "f10",
            Qt.Key.Key_F11.value: "f11",
            Qt.Key.Key_F12.value: "f12",
            Qt.Key.Key_Left.value: "left",
            Qt.Key.Key_Right.value: "right",
            Qt.Key.Key_Up.value: "up",
            Qt.Key.Key_Down.value: "down",
            Qt.Key.Key_Space.value: "space",
            Qt.Key.Key_Plus.value: "plus",
            Qt.Key.Key_Comma.value: "comma",
            Qt.Key.Key_Tab.value: "tab",
            Qt.Key.Key_Return.value: "enter",
            Qt.Key.Key_Enter.value: "enter",
            Qt.Key.Key_Insert.value: "insert",
            Qt.Key.Key_Home.value: "home",
            Qt.Key.Key_End.value: "end",
            Qt.Key.Key_PageUp.value: "pageup",
            Qt.Key.Key_PageDown.value: "pagedown",
            Qt.Key.Key_CapsLock.value: "capslock",
            Qt.Key.Key_NumLock.value: "numlock",
            Qt.Key.Key_ScrollLock.value: "scrolllock",
            Qt.Key.Key_Print.value: "print_screen",
            Qt.Key.Key_Pause.value: "pause",
        }
        if key in key_map:
            return key_map[key]

        if 0x20 <= key <= 0x7E:
            return chr(key).lower()

        return ""

class SettingsWindow(QMainWindow):
    settings_saved = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(SETTINGS_WINDOW_TITLE)
        self.setWindowIcon(QIcon(resource_path(APP_ICON_FILENAME)))
        self.setGeometry(100, 100, 500, 600)
        
        self.init_ui()
        self.load_settings()

    def init_ui(self):
        layout = QVBoxLayout()
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # General
        group_general = QGroupBox("General")
        group_general.setObjectName("generalSettingsGroup")
        layout_general = QFormLayout()
        self.chk_launch_at_startup = QCheckBox("Start with Windows")
        self.combo_auto_stop = QComboBox()
        for seconds, label in AUTO_STOP_OPTIONS:
            self.combo_auto_stop.addItem(label, seconds)
        self.chk_recording_indicator = QCheckBox("Show floating recording timer")
        self.chk_recording_indicator.setChecked(True)
        layout_general.addRow(self.chk_launch_at_startup)
        layout_general.addRow(self.chk_recording_indicator)
        layout_general.addRow("Auto-stop after silence:", self.combo_auto_stop)
        group_general.setLayout(layout_general)
        layout.addWidget(group_general)

        # Microphone
        group_mic = QGroupBox("Input Device")
        layout_mic = QVBoxLayout()
        self.combo_mic = QComboBox()
        layout_mic.addWidget(self.combo_mic)
        btn_refresh = QPushButton("Refresh Devices")
        btn_refresh.clicked.connect(self.refresh_devices)
        layout_mic.addWidget(btn_refresh)
        group_mic.setLayout(layout_mic)
        layout.addWidget(group_mic)

        # Output
        group_out = QGroupBox("Output Configuration")
        layout_out = QFormLayout()
        
        layout_folder_inner = QHBoxLayout()
        self.lbl_folder = QLabel(os.getcwd())
        btn_browse = QPushButton("Browse...")
        btn_browse.clicked.connect(self.browse_folder)
        layout_folder_inner.addWidget(self.lbl_folder)
        layout_folder_inner.addWidget(btn_browse)
        
        self.combo_fmt = QComboBox()
        for key, config in FORMAT_CONFIG.items():
            self.combo_fmt.addItem(config["label"], key)

        self.combo_quality = QComboBox()
        for key, config in QUALITY_CONFIG.items():
            self.combo_quality.addItem(config["label"], key)

        self.chk_stereo = QCheckBox("Keep Stereo")
        self.chk_stereo.setChecked(False)
        self.lbl_preview = QLabel()

        self.combo_fmt.currentIndexChanged.connect(self.update_output_preview)
        self.combo_quality.currentIndexChanged.connect(self.update_output_preview)
        self.chk_stereo.toggled.connect(self.update_output_preview)
        
        layout_out.addRow("Folder:", layout_folder_inner)
        layout_out.addRow("Format:", self.combo_fmt)
        layout_out.addRow("Quality:", self.combo_quality)
        layout_out.addRow("Stereo:", self.chk_stereo)
        layout_out.addRow("Preview:", self.lbl_preview)
        group_out.setLayout(layout_out)
        layout.addWidget(group_out)

        # Tray Interaction
        group_tray = QGroupBox("Tray Icon Behavior")
        layout_tray = QFormLayout()
        self.combo_left_click = QComboBox()
        self.combo_left_click.addItems(["Last Used", "Microphone", "Loopback", "Both"])
        layout_tray.addRow("Left Click Action:", self.combo_left_click)
        group_tray.setLayout(layout_tray)
        layout.addWidget(group_tray)

        # Notifications
        group_notifications = QGroupBox("Notifications")
        group_notifications.setObjectName("notificationsSettingsGroup")
        layout_notifications = QVBoxLayout()
        self.chk_notifications = QCheckBox("Show tray notifications")
        self.chk_notifications.setChecked(True)
        layout_notifications.addWidget(self.chk_notifications)
        group_notifications.setLayout(layout_notifications)
        layout.addWidget(group_notifications)

        # Post-Processing
        group_post = QGroupBox("Post-Processing & Clipboard")
        group_post.setObjectName("postProcessingSettingsGroup")
        layout_post = QVBoxLayout()
        self.chk_normalize = QCheckBox("Normalize Audio (Apply first)")
        self.chk_trim_silence = QCheckBox("Trim start/end silence over 5s")
        self.chk_trim_silence.setChecked(False)
        self.chk_clipboard = QCheckBox("Copy File to Clipboard")
        self.chk_delete = QCheckBox("Delete after Copy (Move to Temp)")
        self.chk_delete.setToolTip("Moves the file to the system temp folder before copying, keeping your output folder clean.")
        self.chk_delete.setEnabled(False)
        self.chk_clipboard.toggled.connect(lambda c: self.chk_delete.setEnabled(c))
        
        layout_post.addWidget(self.chk_normalize)
        layout_post.addWidget(self.chk_trim_silence)
        layout_post.addWidget(self.chk_clipboard)
        layout_post.addWidget(self.chk_delete)
        group_post.setLayout(layout_post)
        layout.addWidget(group_post)

        # Hotkeys
        group_hotkeys = QGroupBox("Global Hotkeys")
        layout_hotkeys = QFormLayout()
        self.hk_mic = HotkeyEdit()
        self.hk_loop = HotkeyEdit()
        self.hk_both = HotkeyEdit()
        self.hk_stop = HotkeyEdit()
        self.chk_stop_with_record_hotkeys = QCheckBox("Use record hotkeys to stop recording")
        self.chk_stop_with_record_hotkeys.setToolTip("When enabled, pressing any record hotkey while recording stops the active recording instead of starting another mode.")
        self.chk_stop_with_record_hotkeys.setChecked(True)
        self.chk_stop_with_record_hotkeys.toggled.connect(self.update_stop_hotkey_state)
        layout_hotkeys.addRow("Record Mic:", self.hk_mic)
        layout_hotkeys.addRow("Record Loopback:", self.hk_loop)
        layout_hotkeys.addRow("Record Both:", self.hk_both)
        layout_hotkeys.addRow("", self.chk_stop_with_record_hotkeys)
        layout_hotkeys.addRow("Stop Recording:", self.hk_stop)
        group_hotkeys.setLayout(layout_hotkeys)
        layout.addWidget(group_hotkeys)

        btn_save = QPushButton("Save Settings")
        btn_save.clicked.connect(self.save_settings)
        layout.addWidget(btn_save)

        self.refresh_devices()
        self.update_stop_hotkey_state()

    def update_stop_hotkey_state(self):
        use_record_hotkeys = self.chk_stop_with_record_hotkeys.isChecked()
        self.hk_stop.setEnabled(not use_record_hotkeys)
        if use_record_hotkeys:
            self.hk_stop.setPlaceholderText("Using record hotkeys")
        else:
            self.hk_stop.setPlaceholderText("Click to set hotkey...")

    def refresh_devices(self):
        self.combo_mic.clear()
        try:
            mics = get_devices(include_loopback=False)
            default_mic = sc.default_microphone()
            default_index = 0
            for i, m in enumerate(mics):
                self.combo_mic.addItem(f"{m['name']}", m['id'])
                if m['id'] == default_mic.id:
                    default_index = i
            self.combo_mic.setCurrentIndex(default_index)
        except Exception as e:
            print(f"Error refreshing devices: {e}")

    def browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder", options=QFileDialog.Option.DontUseNativeDialog)
        if folder:
            self.lbl_folder.setText(folder)

    def _set_combo_by_data(self, combo, value, default_value):
        normalized = str(value or default_value).strip().lower()
        default_normalized = str(default_value or "").strip().lower()
        idx = combo.findData(normalized)
        if idx < 0:
            idx = combo.findData(default_normalized)
        if idx >= 0:
            combo.setCurrentIndex(idx)

    def _set_auto_stop_combo(self, value):
        seconds = normalize_auto_stop_silence_seconds(value)
        for idx in range(self.combo_auto_stop.count()):
            if self.combo_auto_stop.itemData(idx) == seconds:
                self.combo_auto_stop.setCurrentIndex(idx)
                return

        default_idx = self.combo_auto_stop.findData(AUTO_STOP_DEFAULT_SECONDS)
        if default_idx >= 0:
            self.combo_auto_stop.setCurrentIndex(default_idx)

    def _parse_bool_setting(self, value):
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in ("true", "1", "yes", "on"):
                return True
            if normalized in ("false", "0", "no", "off", ""):
                return False
        return False

    def update_output_preview(self):
        fmt = self.combo_fmt.currentData() or "flac"
        quality = self.combo_quality.currentData() or "balanced"
        stereo = self.chk_stereo.isChecked()
        self.lbl_preview.setText(describe_output_profile(fmt, quality, stereo))

    def load_settings(self):
        data = {}
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if not isinstance(data, dict):
                    data = {}
            except Exception as e:
                print(f"Error loading settings: {e}")

        self.lbl_folder.setText(data.get("output_folder", os.getcwd()))
        self._set_combo_by_data(self.combo_fmt, data.get("format"), "flac")
        self._set_combo_by_data(self.combo_quality, data.get("quality"), "balanced")
        self.chk_stereo.setChecked(self._parse_bool_setting(data.get("stereo")))
        self.chk_launch_at_startup.setChecked(
            self._parse_bool_setting(data.get("launch_at_startup"))
        )

        saved_id = data.get("device_id")
        if saved_id:
            idx = self.combo_mic.findData(saved_id)
            if idx >= 0: self.combo_mic.setCurrentIndex(idx)

        mode = data.get("tray_click_mode", "Last Used")
        mode_idx = self.combo_left_click.findText(mode)
        if mode_idx >= 0: self.combo_left_click.setCurrentIndex(mode_idx)
        self._set_auto_stop_combo(
            data.get("auto_stop_silence_seconds", AUTO_STOP_DEFAULT_SECONDS)
        )

        self.chk_normalize.setChecked(data.get("normalize", False))
        self.chk_trim_silence.setChecked(
            self._parse_bool_setting(data.get("trim_silence"))
        )
        self.chk_clipboard.setChecked(data.get("clipboard", False))
        self.chk_delete.setChecked(data.get("delete_after", False))
        self.chk_delete.setEnabled(self.chk_clipboard.isChecked())
        self.chk_notifications.setChecked(data.get("show_notifications", True))
        if "show_recording_indicator" in data:
            self.chk_recording_indicator.setChecked(
                self._parse_bool_setting(data.get("show_recording_indicator"))
            )
        else:
            self.chk_recording_indicator.setChecked(True)
        stop_with_record_hotkeys = data.get("stop_with_record_hotkeys")
        if stop_with_record_hotkeys is None:
            stop_with_record_hotkeys = not bool(data.get("hk_stop", ""))
        self.chk_stop_with_record_hotkeys.setChecked(
            self._parse_bool_setting(stop_with_record_hotkeys)
        )

        self.hk_mic.setText(data.get("hk_mic", ""))
        self.hk_loop.setText(data.get("hk_loop", ""))
        self.hk_both.setText(data.get("hk_both", ""))
        self.hk_stop.setText(data.get("hk_stop", ""))
        self.update_stop_hotkey_state()
        self.update_output_preview()

    def save_settings(self):
        data = self.get_settings()
        try:
            startup_updated = set_launch_at_startup_enabled(data.get("launch_at_startup", False))
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            if not startup_updated:
                QMessageBox.warning(
                    self,
                    "Settings",
                    "Settings saved, but the Windows startup setting could not be updated.",
                )
                self.settings_saved.emit()
                return
            QMessageBox.information(self, "Settings", "Settings saved successfully.")
            self.settings_saved.emit()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save settings: {e}")

    def get_settings(self):
        return {
            "device_id": self.combo_mic.currentData(),
            "output_folder": self.lbl_folder.text(),
            "format": self.combo_fmt.currentData(),
            "quality": self.combo_quality.currentData(),
            "stereo": self.chk_stereo.isChecked(),
            "launch_at_startup": self.chk_launch_at_startup.isChecked(),
            "tray_click_mode": self.combo_left_click.currentText(),
            "auto_stop_silence_seconds": self.combo_auto_stop.currentData(),
            "show_notifications": self.chk_notifications.isChecked(),
            "show_recording_indicator": self.chk_recording_indicator.isChecked(),
            "normalize": self.chk_normalize.isChecked(),
            "trim_silence": self.chk_trim_silence.isChecked(),
            "clipboard": self.chk_clipboard.isChecked(),
            "delete_after": self.chk_delete.isChecked(),
            "stop_with_record_hotkeys": self.chk_stop_with_record_hotkeys.isChecked(),
            "hk_mic": self.hk_mic.text(),
            "hk_loop": self.hk_loop.text(),
            "hk_both": self.hk_both.text(),
            "hk_stop": self.hk_stop.text()
        }

class TrayApplication(QObject):
    def __init__(self, app):
        super().__init__()
        self.app = app
        self.recorder = None
        self.last_mode = "mic" 
        
        self.signals = SignalManager()
        self.signals.recording_finished.connect(self.on_recording_finished)

        self.icon_idle_path = resource_path(ICON_IDLE_FILENAME)
        self.icon_rec_path = resource_path(ICON_RECORDING_FILENAME)
        self.generate_icons()
            
        self.tray_icon = QSystemTrayIcon(QIcon(self.icon_idle_path), self.app)
        self.tray_icon.setToolTip(TRAY_IDLE_TOOLTIP)
        self.tray_icon.activated.connect(self.on_tray_activated)
        self.recording_indicator = RecordingIndicator()
        self.recording_indicator.stop_requested.connect(self.stop_recording)
        self.recording_indicator.open_folder_requested.connect(self.open_recordings_folder)
        
        self.build_menu()
        self.tray_icon.show()
        
        self.settings_window = SettingsWindow()
        self.settings_window.settings_saved.connect(self.register_hotkeys)
        self.hotkey_manager = create_hotkey_manager(self.app)
        
        self.show_tray_notification(
            READY_MESSAGE_TITLE,
            READY_MESSAGE_BODY,
            QSystemTrayIcon.MessageIcon.Information,
            2000,
        )
        self.register_hotkeys()

    def generate_icons(self):
        missing = [
            path
            for path in (self.icon_idle_path, self.icon_rec_path)
            if not os.path.exists(path)
        ]
        if missing:
            raise FileNotFoundError(f"Missing application icon asset: {missing[0]}")

    def build_menu(self):
        self.menu = QMenu()
        self.action_record_mic = QAction("Start Recording (Mic)", self)
        self.action_record_mic.triggered.connect(lambda: self.start_recording("mic"))
        self.action_record_loop = QAction("Start Recording (Loopback)", self)
        self.action_record_loop.triggered.connect(lambda: self.start_recording("loopback"))
        self.action_record_both = QAction("Start Recording (Both)", self)
        self.action_record_both.triggered.connect(lambda: self.start_recording("both"))
        self.action_stop = QAction("Stop Recording", self)
        self.action_stop.triggered.connect(self.stop_recording)
        self.action_stop.setEnabled(False)
        self.action_settings = QAction("Settings", self)
        self.action_settings.triggered.connect(self.open_settings)
        self.action_open_folder = QAction("Open Recordings Folder", self)
        self.action_open_folder.triggered.connect(self.open_recordings_folder)
        self.action_exit = QAction("Exit", self)
        self.action_exit.triggered.connect(self.exit_app)
        
        self.menu.addAction(self.action_record_mic)
        self.menu.addAction(self.action_record_loop)
        self.menu.addAction(self.action_record_both)
        self.menu.addAction(self.action_stop)
        self.menu.addSeparator()
        self.menu.addAction(self.action_open_folder)
        self.menu.addAction(self.action_settings)
        self.menu.addAction(self.action_exit)
        self.tray_icon.setContextMenu(self.menu)
        recording_indicator = getattr(self, "recording_indicator", None)
        if recording_indicator:
            recording_indicator.setContextMenu(self.menu)

    def register_hotkeys(self):
        hotkey_manager = getattr(self, "hotkey_manager", None)
        if hotkey_manager is None:
            hotkey_manager = KeyboardHotkeyManager()
            self.hotkey_manager = hotkey_manager
        try:
            hotkey_manager.clear()
        except Exception as e:
            print(f"Failed to clear hotkeys: {e}")
        settings = self.settings_window.get_settings()
        hk_mic = settings.get("hk_mic")
        hk_loop = settings.get("hk_loop")
        hk_both = settings.get("hk_both")
        hk_stop = settings.get("hk_stop")
        try:
            if hk_mic: hotkey_manager.register(hk_mic, lambda: self.toggle_recording("mic"))
            if hk_loop: hotkey_manager.register(hk_loop, lambda: self.toggle_recording("loopback"))
            if hk_both: hotkey_manager.register(hk_both, lambda: self.toggle_recording("both"))
            if hk_stop and not settings.get("stop_with_record_hotkeys", True):
                hotkey_manager.register(hk_stop, self.stop_recording)
        except Exception as e: print(f"Failed to register hotkeys: {e}")

    def notifications_enabled(self):
        try:
            return self.settings_window.get_settings().get("show_notifications", True)
        except Exception:
            return True

    def show_tray_notification(self, title, message, icon=QSystemTrayIcon.MessageIcon.Information, duration=2000):
        notifications_enabled = getattr(self, "notifications_enabled", lambda: TrayApplication.notifications_enabled(self))
        if notifications_enabled():
            self.tray_icon.showMessage(title, message, icon, duration)

    def toggle_recording(self, mode="mic"):
        if self.recorder and self.recorder.is_alive():
            settings = self.settings_window.get_settings()
            if settings.get("stop_with_record_hotkeys", True):
                self.stop_recording()
            return

        self.start_recording(mode)

    def on_tray_activated(self, reason):
        if reason == QSystemTrayIcon.ActivationReason.Trigger:
            if self.recorder and self.recorder.is_alive():
                self.stop_recording()
            else:
                settings = self.settings_window.get_settings()
                click_mode = settings.get("tray_click_mode", "Last Used")
                target_mode = self.last_mode
                if click_mode == "Microphone": target_mode = "mic"
                elif click_mode == "Loopback": target_mode = "loopback"
                elif click_mode == "Both": target_mode = "both"
                self.start_recording(target_mode)

    def open_recordings_folder(self):
        try:
            folder = self.settings_window.get_settings().get("output_folder") or os.getcwd()
            folder = os.path.abspath(folder)
            if not os.path.exists(folder):
                os.makedirs(folder, exist_ok=True)
            if sys.platform == "win32":
                os.startfile(folder)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", folder])
            else:
                subprocess.Popen(["xdg-open", folder])
        except Exception as e:
            self.show_tray_notification(
                "Error",
                f"Failed to open recordings folder: {e}",
                QSystemTrayIcon.MessageIcon.Critical,
                4000,
            )

    def open_settings(self):
        self.settings_window.show()
        self.settings_window.raise_()
        self.settings_window.activateWindow()

    def start_recording(self, mode="mic"):
        if self.recorder and self.recorder.is_alive(): return
        self.last_mode = mode    
        settings = self.settings_window.get_settings()
        target_id = settings['device_id']
        
        def finish_callback(path, error):
            self.signals.recording_finished.emit(path if path else "", error if error else "")

        self.recorder = AudioRecorder(
            mic_id=target_id,
            source_mode=mode,
            output_folder=settings['output_folder'],
            output_format=settings['format'],
            quality=settings['quality'],
            stereo=settings['stereo'],
            normalize=settings['normalize'],
            trim_silence=settings.get("trim_silence", False),
            auto_stop_silence_seconds=settings.get("auto_stop_silence_seconds"),
            on_finish_callback=finish_callback
        )
        self.recorder.start()
        self.action_record_mic.setEnabled(False)
        self.action_record_loop.setEnabled(False)
        self.action_record_both.setEnabled(False)
        self.action_stop.setEnabled(True)
        self.tray_icon.setIcon(QIcon(self.icon_rec_path)) 
        self.tray_icon.setToolTip(f"{APP_NAME} Recording ({mode})")
        if settings.get("show_recording_indicator", True):
            recording_indicator = getattr(self, "recording_indicator", None)
            if recording_indicator:
                recording_indicator.show_recording()
        self.show_tray_notification("Started", f"Recording {mode}", QSystemTrayIcon.MessageIcon.NoIcon, 1000)

    def stop_recording(self):
        recording_indicator = getattr(self, "recording_indicator", None)
        if recording_indicator and recording_indicator.isVisible():
            recording_indicator.show_finished(RecordingIndicator.FINISHED_HIDE_DELAY_MS)
        if self.recorder: self.recorder.stop()

    def on_recording_finished(self, path, error):
        self.action_record_mic.setEnabled(True)
        self.action_record_loop.setEnabled(True)
        self.action_record_both.setEnabled(True)
        self.action_stop.setEnabled(False)
        self.tray_icon.setIcon(QIcon(self.icon_idle_path))
        self.tray_icon.setToolTip(TRAY_IDLE_TOOLTIP)
        recording_indicator = getattr(self, "recording_indicator", None)
        if recording_indicator and not getattr(recording_indicator, "is_finishing", False):
            recording_indicator.hide_recording()
        self.recorder = None
        
        if error:
            self.show_tray_notification("Error", f"Recording failed: {error}", QSystemTrayIcon.MessageIcon.Critical, 4000)
            return
            
        settings = self.settings_window.get_settings()
        final_path = path
        msg = f"Saved to {os.path.basename(path)}"
        
        if settings['clipboard'] and os.path.exists(path):
            try:
                if settings['delete_after']:
                    temp_dir = tempfile.gettempdir()
                    new_path = os.path.join(temp_dir, os.path.basename(path))
                    if os.path.exists(new_path):
                        base, ext = os.path.splitext(new_path)
                        import time
                        new_path = f"{base}_{int(time.time())}{ext}"
                    shutil.move(path, new_path)
                    final_path = new_path
                    msg = "Moved to Temp & Copied to Clipboard."
                else:
                    msg += "\nCopied to clipboard."

                # Use Robust Clipboard Utility
                success, status = copy_file_to_clipboard(final_path)
                if not success:
                    msg += f"\nClipboard Error: {status}"
                else:
                    # Optional: Log success?
                    pass
                
            except Exception as e:
                msg += f"\nClipboard/Move error: {e}"

        self.show_tray_notification("Finished", msg, QSystemTrayIcon.MessageIcon.Information, 2000)

    def exit_app(self):
        if self.recorder: self.recorder.stop()
        recording_indicator = getattr(self, "recording_indicator", None)
        if recording_indicator:
            recording_indicator.hide_recording()
        self.app.quit()
