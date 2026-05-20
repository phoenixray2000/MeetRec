import sys
import os
import json
import shutil
import tempfile
import subprocess
from PyQt6.QtWidgets import (QApplication, QSystemTrayIcon, QMenu, QMainWindow, 
                             QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, 
                             QPushButton, QFileDialog, QMessageBox, QGroupBox, 
                             QLineEdit, QFormLayout, QCheckBox)
from PyQt6.QtGui import QIcon, QAction, QColor, QPixmap, QPainter, QBrush, QKeySequence
from PyQt6.QtCore import pyqtSignal, QObject, Qt, QUrl, QMimeData, QDir, QTimer
import soundcard as sc
import keyboard
from app_metadata import (
    APP_NAME,
    ICON_IDLE_FILENAME,
    ICON_RECORDING_FILENAME,
    READY_MESSAGE_BODY,
    READY_MESSAGE_TITLE,
    SETTINGS_WINDOW_TITLE,
    TRAY_IDLE_TOOLTIP,
)
from audio_recorder import (
    AudioRecorder,
    FORMAT_CONFIG,
    QUALITY_CONFIG,
    describe_output_profile,
    get_devices,
)
from clipboard_utils import copy_file_to_clipboard

CONFIG_FILE = "settings.json"

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

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
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setPlaceholderText("Click to set hotkey...")
        self.setReadOnly(True) 
        self.current_sequence = None

    def mousePressEvent(self, event):
        self.setFocus()
        super().mousePressEvent(event)

    def keyPressEvent(self, event):
        key = event.key()
        modifiers = event.modifiers()
        
        if key == Qt.Key.Key_Backspace or key == Qt.Key.Key_Delete:
            self.clear()
            self.current_sequence = None
            return
            
        if key == Qt.Key.Key_Escape:
            self.clearFocus()
            return

        if key in (Qt.Key.Key_Control, Qt.Key.Key_Shift, Qt.Key.Key_Alt, Qt.Key.Key_Meta):
            return

        parts = []
        if modifiers & Qt.KeyboardModifier.ControlModifier: parts.append("ctrl")
        if modifiers & Qt.KeyboardModifier.ShiftModifier:   parts.append("shift")
        if modifiers & Qt.KeyboardModifier.AltModifier:     parts.append("alt")
        if modifiers & Qt.KeyboardModifier.MetaModifier:    parts.append("windows")

        key_text = ""
        if key >= 0x20 and key <= 0x7E:
            key_text = chr(key).lower()
        else:
            key_map = {
                Qt.Key.Key_F1: "f1", Qt.Key.Key_F2: "f2", Qt.Key.Key_F3: "f3", Qt.Key.Key_F4: "f4",
                Qt.Key.Key_F5: "f5", Qt.Key.Key_F6: "f6", Qt.Key.Key_F7: "f7", Qt.Key.Key_F8: "f8",
                Qt.Key.Key_F9: "f9", Qt.Key.Key_F10: "f10", Qt.Key.Key_F11: "f11", Qt.Key.Key_F12: "f12",
                Qt.Key.Key_Left: "left", Qt.Key.Key_Right: "right", Qt.Key.Key_Up: "up", Qt.Key.Key_Down: "down",
                Qt.Key.Key_Space: "space", Qt.Key.Key_Tab: "tab", Qt.Key.Key_Return: "enter", Qt.Key.Key_Enter: "enter",
                Qt.Key.Key_Backspace: "backspace", Qt.Key.Key_Delete: "delete", Qt.Key.Key_Insert: "insert",
                Qt.Key.Key_Home: "home", Qt.Key.Key_End: "end", Qt.Key.Key_PageUp: "pageup", Qt.Key.Key_PageDown: "pagedown",
                Qt.Key.Key_CapsLock: "capslock", Qt.Key.Key_NumLock: "numlock", Qt.Key.Key_ScrollLock: "scrolllock",
                Qt.Key.Key_Print: "print_screen", Qt.Key.Key_Pause: "pause"
            }
            key_text = key_map.get(key)
            if not key_text:
                try: key_text = QKeySequence(key).toString().lower()
                except: pass

        if key_text:
            parts.append(key_text)
            
        final_hotkey = "+".join(parts)
        self.setText(final_hotkey)
        self.current_sequence = final_hotkey
        self.clearFocus()

class SettingsWindow(QMainWindow):
    settings_saved = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(SETTINGS_WINDOW_TITLE)
        self.setGeometry(100, 100, 500, 600)
        
        self.init_ui()
        self.load_settings()

    def init_ui(self):
        layout = QVBoxLayout()
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

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
        layout_notifications = QVBoxLayout()
        self.chk_recording_indicator = QCheckBox("Show floating recording timer")
        self.chk_recording_indicator.setChecked(True)
        layout_notifications.addWidget(self.chk_recording_indicator)
        group_notifications.setLayout(layout_notifications)
        layout.addWidget(group_notifications)

        # Post-Processing
        group_post = QGroupBox("Post-Processing & Clipboard")
        layout_post = QVBoxLayout()
        self.chk_normalize = QCheckBox("Normalize Audio (Apply first)")
        self.chk_clipboard = QCheckBox("Copy File to Clipboard")
        self.chk_delete = QCheckBox("Delete after Copy (Move to Temp)")
        self.chk_delete.setToolTip("Moves the file to the system temp folder before copying, keeping your output folder clean.")
        self.chk_delete.setEnabled(False)
        self.chk_clipboard.toggled.connect(lambda c: self.chk_delete.setEnabled(c))
        
        layout_post.addWidget(self.chk_normalize)
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
        layout_hotkeys.addRow("Record Mic:", self.hk_mic)
        layout_hotkeys.addRow("Record Loopback:", self.hk_loop)
        layout_hotkeys.addRow("Record Both:", self.hk_both)
        layout_hotkeys.addRow("Stop Recording:", self.hk_stop)
        group_hotkeys.setLayout(layout_hotkeys)
        layout.addWidget(group_hotkeys)

        btn_save = QPushButton("Save Settings")
        btn_save.clicked.connect(self.save_settings)
        layout.addWidget(btn_save)

        self.refresh_devices()

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
                with open(CONFIG_FILE, 'r') as f:
                    data = json.load(f)
                if not isinstance(data, dict):
                    data = {}
            except Exception as e:
                print(f"Error loading settings: {e}")

        self.lbl_folder.setText(data.get("output_folder", os.getcwd()))
        self._set_combo_by_data(self.combo_fmt, data.get("format"), "flac")
        self._set_combo_by_data(self.combo_quality, data.get("quality"), "balanced")
        self.chk_stereo.setChecked(self._parse_bool_setting(data.get("stereo")))

        saved_id = data.get("device_id")
        if saved_id:
            idx = self.combo_mic.findData(saved_id)
            if idx >= 0: self.combo_mic.setCurrentIndex(idx)

        mode = data.get("tray_click_mode", "Last Used")
        mode_idx = self.combo_left_click.findText(mode)
        if mode_idx >= 0: self.combo_left_click.setCurrentIndex(mode_idx)

        self.chk_normalize.setChecked(data.get("normalize", False))
        self.chk_clipboard.setChecked(data.get("clipboard", False))
        self.chk_delete.setChecked(data.get("delete_after", False))
        self.chk_delete.setEnabled(self.chk_clipboard.isChecked())
        if "show_recording_indicator" in data:
            self.chk_recording_indicator.setChecked(
                self._parse_bool_setting(data.get("show_recording_indicator"))
            )
        else:
            self.chk_recording_indicator.setChecked(True)

        self.hk_mic.setText(data.get("hk_mic", ""))
        self.hk_loop.setText(data.get("hk_loop", ""))
        self.hk_both.setText(data.get("hk_both", ""))
        self.hk_stop.setText(data.get("hk_stop", ""))
        self.update_output_preview()

    def save_settings(self):
        data = self.get_settings()
        try:
            with open(CONFIG_FILE, 'w') as f:
                json.dump(data, f)
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
            "tray_click_mode": self.combo_left_click.currentText(),
            "show_recording_indicator": self.chk_recording_indicator.isChecked(),
            "normalize": self.chk_normalize.isChecked(),
            "clipboard": self.chk_clipboard.isChecked(),
            "delete_after": self.chk_delete.isChecked(),
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
        
        self.tray_icon.showMessage(
            READY_MESSAGE_TITLE,
            READY_MESSAGE_BODY,
            QSystemTrayIcon.MessageIcon.Information,
            2000,
        )
        self.register_hotkeys()

    def generate_icons(self):
        if not os.path.exists(self.icon_idle_path):
            pix = QPixmap(64, 64)
            pix.fill(Qt.GlobalColor.transparent)
            painter = QPainter(pix)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setBrush(QBrush(QColor(80, 80, 80)))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(4, 4, 56, 56)
            painter.end()
            pix.save(self.icon_idle_path)

        if not os.path.exists(self.icon_rec_path):
            pix = QPixmap(64, 64)
            pix.fill(Qt.GlobalColor.transparent)
            painter = QPainter(pix)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setBrush(QBrush(QColor(220, 0, 0)))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(4, 4, 56, 56)
            painter.setBrush(QBrush(QColor(255, 255, 255)))
            painter.drawEllipse(22, 22, 20, 20)
            painter.end()
            pix.save(self.icon_rec_path)

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
        try: keyboard.unhook_all_hotkeys() # Ensure no old hotkeys are active
        except: pass
        settings = self.settings_window.get_settings()
        hk_mic = settings.get("hk_mic")
        hk_loop = settings.get("hk_loop")
        hk_both = settings.get("hk_both")
        hk_stop = settings.get("hk_stop")
        try:
            if hk_mic: keyboard.add_hotkey(hk_mic, lambda: self.start_recording("mic"))
            if hk_loop: keyboard.add_hotkey(hk_loop, lambda: self.start_recording("loopback"))
            if hk_both: keyboard.add_hotkey(hk_both, lambda: self.start_recording("both"))
            if hk_stop: keyboard.add_hotkey(hk_stop, self.stop_recording)
        except Exception as e: print(f"Failed to register hotkeys: {e}")

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
            self.tray_icon.showMessage(
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
        self.tray_icon.showMessage("Started", f"Recording {mode}", QSystemTrayIcon.MessageIcon.NoIcon, 1000)

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
            self.tray_icon.showMessage("Error", f"Recording failed: {error}", QSystemTrayIcon.MessageIcon.Critical, 4000)
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

        self.tray_icon.showMessage("Finished", msg, QSystemTrayIcon.MessageIcon.Information, 2000)

    def exit_app(self):
        if self.recorder: self.recorder.stop()
        recording_indicator = getattr(self, "recording_indicator", None)
        if recording_indicator:
            recording_indicator.hide_recording()
        self.app.quit()
