import sys
from PyQt6.QtWidgets import QApplication

from core.lp_solver import LPSolver
from core import SimplexAlgorithm, BFSFinder

from view.app_window import LPSolverApp
from view import InputSection, ResultSection


def main():
    app = QApplication(sys.argv)

    solver = LPSolver(
        bfs_finder=BFSFinder(),
        algorithm=SimplexAlgorithm()
    )
    window = LPSolverApp(
        input_section=InputSection(),
        results_section=ResultSection(),
        solver=solver
    )
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()