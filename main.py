import sys
from PyQt6.QtWidgets import QApplication
from view.app_window import LPSolverApp
from view import InputSection, ResultSection


def main():
    app = QApplication(sys.argv)
    window = LPSolverApp(
        input_section=InputSection(),
        results_section=ResultSection()
    )
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()