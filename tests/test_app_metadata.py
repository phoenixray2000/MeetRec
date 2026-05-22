import unittest
from pathlib import Path

import app_metadata


class AppMetadataTests(unittest.TestCase):
    def test_public_product_identity_is_meetrec(self):
        self.assertEqual(app_metadata.APP_NAME, "MeetRec")
        self.assertEqual(app_metadata.ORGANIZATION_NAME, "MeetRec")
        self.assertEqual(app_metadata.SETTINGS_WINDOW_TITLE, "Settings - MeetRec")
        self.assertEqual(app_metadata.TRAY_IDLE_TOOLTIP, "MeetRec (Idle)")
        self.assertEqual(app_metadata.RECORDING_FILENAME_PREFIX, "MeetRec")

    def test_packaging_and_release_names_are_meetrec(self):
        self.assertEqual(app_metadata.EXECUTABLE_NAME, "MeetRec")
        self.assertEqual(app_metadata.EXECUTABLE_FILENAME, "MeetRec.exe")
        self.assertEqual(app_metadata.RELEASES_URL, "https://github.com/phoenixray2000/MeetRec/releases")

    def test_generated_icon_names_do_not_reuse_legacy_project_name(self):
        self.assertEqual(app_metadata.ICON_IDLE_FILENAME, "docs/meetrec_idle.ico")
        self.assertEqual(app_metadata.ICON_RECORDING_FILENAME, "docs/meetrec_recording.ico")
        self.assertEqual(app_metadata.APP_ICON_FILENAME, app_metadata.ICON_IDLE_FILENAME)

    def test_icon_assets_exist(self):
        self.assertTrue(Path(app_metadata.ICON_IDLE_FILENAME).exists())
        self.assertTrue(Path(app_metadata.ICON_RECORDING_FILENAME).exists())
        self.assertTrue(Path(app_metadata.APP_ICON_FILENAME).exists())


if __name__ == "__main__":
    unittest.main()
