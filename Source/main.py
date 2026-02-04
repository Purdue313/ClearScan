import sys
import traceback
from PySide6.QtWidgets import QApplication
from UserInterface.ui import MainWindow

def main():
    try:
        print("Creating QApplication...")
        app = QApplication(sys.argv)
        print("Creating MainWindow...")
        window = MainWindow()
        print("Showing window...")
        window.show()
        print("Starting event loop...")
        sys.exit(app.exec())
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
