import sys
from PyQt6.QtWidgets import QApplication

from core.solvers.simplex_solver import SimplexSolver
from core import solvers, bfs_finders

from view.app_window import LPSolverApp
from view import InputSection, ResultSection


def main():
    app = QApplication(sys.argv)

    window = LPSolverApp(
        input_section=InputSection(),
        results_section=ResultSection(),
        bfs_finders=bfs_finders,
        solvers=solvers
    )
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()