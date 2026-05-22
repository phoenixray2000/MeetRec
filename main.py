import sys
from PyQt6.QtWidgets import QApplication
from app_metadata import APP_NAME, ORGANIZATION_NAME
from gui import TrayApplication
from single_instance import acquire_single_instance_lock

def main():
    instance_lock = acquire_single_instance_lock()
    if instance_lock is None:
        return 0

    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName(ORGANIZATION_NAME)
    app.setQuitOnLastWindowClosed(False)
    
    tray = TrayApplication(app)
    
    try:
        return app.exec()
    finally:
        instance_lock.unlock()

if __name__ == "__main__":
    sys.exit(main())
