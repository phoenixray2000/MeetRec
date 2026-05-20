# MeetRec

**MeetRec** is a lightweight Windows recorder for meetings, calls, microphone audio, system audio, or both simultaneously.
It sits quietly in your system tray and is always ready with a single click or global hotkey.

<img width="393" height="156" alt="MeetRec tray status" src="https://github.com/user-attachments/assets/7e3bfcaf-6f58-4404-b85a-4ba0b6fea085" />

## Features

<img width="575" height="724" alt="MeetRec settings" src="https://github.com/user-attachments/assets/b35131bc-1ff8-41e1-87b5-1e472f9da981" />

- **Modes**
  - Microphone: Record your voice.
  - System Audio: Record what you hear from the computer.
  - Both: Record both sources simultaneously and mix them into one file.
- **Output Profiles**
  - Format: Save recordings as WAV, FLAC, or MP3.
  - Quality: Choose Balanced for compact 16 kHz output or High Quality for 48 kHz output.
  - Stereo: Keep stereo channels when needed, or leave it off for mono recordings.
- **Post-Processing**
  - Auto-Normalize: Lifts the main voice/body of each source before mixing and limits sharp peaks so brief spikes do not bury the recording.
  - Clipboard Integration: Automatically copies the file or file path to your clipboard.
  - Clean Workflow: Option to move the file to a temp folder and copy it, keeping your desktop clean.
- **Control**
  - Global Hotkeys: Start or stop recording from anywhere.
  - Tray Icon: Left-click to toggle recording immediately; right-click to open the recordings folder, settings, or exit.
  - Visual Feedback: Tray icon changes color while recording, and an optional always-on-top floating timer shows recording status with the same right-click menu.

## Installation

1. Go to the [Releases](https://github.com/phoenixray2000/MeetRec/releases) page.
2. Download `MeetRec.exe`.
3. Run it. No installation is required.

## Usage

1. Right-click the tray icon to open **Settings**.
2. Select your microphone, output folder, format, quality, and stereo preference.
3. Set your hotkeys if needed.
4. Disable **Show floating recording timer** if you do not want the compact always-on-top recording indicator.
5. Left-click the tray icon or use a hotkey to start recording.
6. Click the tray icon again, use a stop hotkey, or click the floating timer to stop. The floating timer changes state immediately, then hides after 5 seconds; click it again before it hides to open the recordings folder.

## Development

### Requirements

- Python 3.12+
- `pip install PyQt6 soundcard soundfile numpy lameenc keyboard`

### Build From Source

To create the standalone executable:

```bash
pip install pyinstaller
pyinstaller --noconsole --onefile --name MeetRec main.py
```

For the maintained Windows package configuration, use:

```bash
pyinstaller --noconfirm MeetRec.spec
```

## Acknowledgements

MeetRec started from an open-source Windows tray recording project by [lukmay](https://github.com/lukmay). Thanks to the original author and contributors for the initial foundation.
