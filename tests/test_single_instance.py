import os
import tempfile
import unittest

import app_metadata
from single_instance import acquire_single_instance_lock, single_instance_lock_path


class SingleInstanceTests(unittest.TestCase):
    def test_lock_path_uses_meetrec_name(self):
        path = single_instance_lock_path()

        self.assertEqual(os.path.basename(path), app_metadata.SINGLE_INSTANCE_LOCK_FILENAME)
        self.assertIn(tempfile.gettempdir(), path)
        self.assertNotIn("QuickAudioRecorder", path)

    def test_second_lock_for_same_path_is_rejected_until_first_unlocks(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        lock_path = os.path.join(temp_dir.name, "MeetRec.lock")

        first_lock = acquire_single_instance_lock(lock_path)
        self.addCleanup(first_lock.unlock)

        second_lock = acquire_single_instance_lock(lock_path)

        self.assertIsNotNone(first_lock)
        self.assertIsNone(second_lock)

        first_lock.unlock()
        third_lock = acquire_single_instance_lock(lock_path)
        self.addCleanup(third_lock.unlock)

        self.assertIsNotNone(third_lock)


if __name__ == "__main__":
    unittest.main()
