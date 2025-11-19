"""Run Electrode Geometry Optimizer GUI."""
import sys
from PySide6.QtWidgets import QApplication
from electrode_optimizer.gui import ElectrodeOptimizerApp

def main():
    app = QApplication(sys.argv)
    win = ElectrodeOptimizerApp()
    win.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
