import os
import tempfile

from PyQt6.QtCore import QLockFile

from app_metadata import SINGLE_INSTANCE_LOCK_FILENAME


def single_instance_lock_path():
    return os.path.join(tempfile.gettempdir(), SINGLE_INSTANCE_LOCK_FILENAME)


def acquire_single_instance_lock(path=None):
    lock = QLockFile(path or single_instance_lock_path())
    if not lock.tryLock(0):
        return None
    return lock
