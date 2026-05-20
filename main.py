import sys
from PyQt6.QtWidgets import QApplication
from app_metadata import APP_NAME, ORGANIZATION_NAME
from gui import TrayApplication

def main():
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName(ORGANIZATION_NAME)
    app.setQuitOnLastWindowClosed(False)
    
    tray = TrayApplication(app)
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
