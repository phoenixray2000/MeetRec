import unittest
from pathlib import Path


class PackagingSpecTests(unittest.TestCase):
    def test_meetrec_spec_builds_onedir_to_avoid_onefile_bootloader_process(self):
        spec_text = Path("MeetRec.spec").read_text(encoding="utf-8")

        self.assertIn("exclude_binaries=True", spec_text)
        self.assertIn("COLLECT(", spec_text)
        self.assertIn("name='MeetRec'", spec_text)

    def test_meetrec_spec_bundles_rnnoise_when_available(self):
        spec_text = Path("MeetRec.spec").read_text(encoding="utf-8")

        self.assertIn('"rnnoise.dll"', spec_text)
        self.assertIn("binaries=_binaries", spec_text)


if __name__ == "__main__":
    unittest.main()
