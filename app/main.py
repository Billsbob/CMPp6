import sys
import os

# Add the project root to sys.path so we can use absolute imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

os.environ['QT_API'] = 'pyside6'
from PySide6.QtWidgets import QApplication

try:
    from app.UI import MainWindow
except ImportError:
    from UI import MainWindow

def main():
    app = QApplication(sys.argv)
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
