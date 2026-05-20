from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]

BANNED_LEGACY_NAMES = (
    "Quick " + "Audio " + "Recorder",
    "Quick" + "Audio" + "Recorder",
    "quick" + "audio" + "recorder",
    "Simple " + "Audio " + "Recorder",
    "lukmay/" + "Quick" + "Audio" + "Recorder",
)

PRODUCT_TEXT_FILES = (
    ".gitignore",
    "MeetRec.spec",
    "README.md",
    "app_metadata.py",
    "audio_recorder.py",
    "clipboard_utils.py",
    "gui.py",
    "main.py",
    "tests/test_app_metadata.py",
    "tests/test_audio_output_profile.py",
    "tests/test_gui_hotkeys.py",
    "tests/test_product_identity_text.py",
)

ACKNOWLEDGEMENT_TEXT = (
    "MeetRec started from an open-source Windows tray recording project "
    "by [lukmay](https://github.com/lukmay)."
)


class ProductIdentityTextTests(unittest.TestCase):
    def test_product_files_do_not_contain_legacy_names(self):
        missing_files = []
        for relative_path in PRODUCT_TEXT_FILES:
            path = ROOT / relative_path
            if not path.exists():
                missing_files.append(relative_path)
                continue
            text = path.read_text(encoding="utf-8")
            for legacy_name in BANNED_LEGACY_NAMES:
                self.assertNotIn(legacy_name, text, relative_path)
        self.assertEqual(missing_files, [])

    def test_readme_acknowledges_original_source_without_legacy_name(self):
        text = (ROOT / "README.md").read_text(encoding="utf-8")

        self.assertIn("## Acknowledgements", text)
        self.assertIn(ACKNOWLEDGEMENT_TEXT, text)
        for legacy_name in BANNED_LEGACY_NAMES:
            self.assertNotIn(legacy_name, text)


if __name__ == "__main__":
    unittest.main()
