from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QPushButton, QTabWidget, QMessageBox, QComboBox
)
from PyQt6.QtGui import QFont
from typing import Dict
from . import InputSection, ResultSection
from core.solvers.simplex_solver import SimplexSolver
from utils import ( 
    StyleSheet, AppConstants,
    IBFSFinder, ISolver
)


class LPSolverApp(QMainWindow):
    """Main application window for LP solver"""
    def __init__(self, input_section: InputSection, results_section: ResultSection,
                 bfs_finders: Dict[str, IBFSFinder], solvers: Dict[str, ISolver]) -> None:
        """
        Initialize the main application window.
        Args:
            input_section: InputSection widget instance for problem setup.
            results_section: ResultSection widget instance for displaying results.
            bfs_finders: Different implementations of BFS finder.
            solvers: Different implementations of simplex algorithm.
        """
        super().__init__()
        self.input_section = input_section
        self.results_section = results_section

        self.bfs_finders = bfs_finders
        self.solvers = solvers
        
        self._setup_window()
        self.init_ui()
        self._connect_signals()

    def _setup_window(self) -> None:
        """Set ups window settings"""
        self.setWindowTitle(AppConstants.WINDOW_TITLE)
        self.setMinimumSize(AppConstants.WINDOW_SIZE[0], AppConstants.WINDOW_SIZE[1])
        self.setStyleSheet(StyleSheet.DARK_STYLE)

    def init_ui(self) -> None:
        """Initialize the user interface layout and components"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(15, 15, 15, 15)
        
        # title
        title = self._create_title()
        main_layout.addWidget(title)
        
        # tabs
        self.tabs = QTabWidget()
        
        # input tab
        input_tab = QWidget()
        input_layout = QVBoxLayout(input_tab)
        input_layout.addWidget(self.input_section)
        self.tabs.addTab(input_tab, "Input")
        
        # results tab
        results_tab = QWidget()
        results_layout = QVBoxLayout(results_tab)
        results_layout.addWidget(self.results_section)
        self.tabs.addTab(results_tab, "Results")
        
        main_layout.addWidget(self.tabs)

        # bfs combo & solver combo
        self.bfs_combo = QComboBox()
        self.bfs_combo.addItems(self.bfs_finders.keys())

        self.solver_combo = QComboBox()
        self.solver_combo.addItems(self.solvers.keys())

        main_layout.addWidget(QLabel("BFS Finder"))
        main_layout.addWidget(self.bfs_combo)

        main_layout.addWidget(QLabel("Solver method"))
        main_layout.addWidget(self.solver_combo)
                
        # controls
        buttons_layout = self._create_buttons_layout()
        main_layout.addLayout(buttons_layout)
    
    def _create_title(self) -> QLabel:
        """Create and configure the title label"""
        title = QLabel("Linear Programming Solver")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title.setFont(title_font)
        return title
    
    def _create_buttons_layout(self) -> QHBoxLayout:
        """Create and configure the control buttons layout"""
        buttons_layout = QHBoxLayout()
        
        self.solve_btn = self._create_button("Solve", 40, 12)
        buttons_layout.addWidget(self.solve_btn)
        
        self.clear_btn = self._create_button("Clear", 40, 12)
        buttons_layout.addWidget(self.clear_btn)
        
        buttons_layout.addStretch()
        return buttons_layout
    
    @staticmethod
    def _create_button(text: str, height: int, font_size: int) -> QPushButton:
        """Create a styled button"""
        button = QPushButton(text)
        button.setMinimumHeight(height)
        
        font = QFont()
        font.setPointSize(font_size)
        font.setBold(True)
        button.setFont(font)
        
        return button
    
    def _connect_signals(self) -> None:
        """Connect button signals to their slots"""
        self.clear_btn.clicked.connect(self.on_clear)
        self.solve_btn.clicked.connect(self.on_solve)
    
    def on_clear(self) -> None:
        """Handle clear button click - reset all forms"""
        self.input_section.clear()
        self.results_section.clear()

    def on_solve(self) -> None:
        """Handle solve button click"""
        try:
            problem_data, success, error_msg = self.input_section.get_data()
            if success and problem_data:
                self.tabs.setCurrentIndex(1)

                # get bfs_finde and solver
                bfs_name = self.bfs_combo.currentText()
                solver_name = self.solver_combo.currentText()

                if bfs_name not in self.bfs_finders:
                    self._show_error(f"Unknown BFS finder: {bfs_name}")
                    return
                if solver_name not in self.solvers:
                    self._show_error(f"Unknown solver: {solver_name}")
                    return

                # create solver
                solver_cls = self.solvers[solver_name]
                solver: ISolver = solver_cls(self.bfs_finders[bfs_name])

                if problem_data.integer_indices is not None and not solver.SUPPORT_INTEGER_CONSTRAINTS:
                    self._show_error(f"This method doesn't support integer constraints: {solver.__class__.__name__}")
                    return
                
                result = solver.solve(problem_data)
                self.results_section.display_results(result)
            else:
                self._show_error(error_msg)
        except Exception as e:
            self._show_error(str(e))

    def _show_error(self, message: str) -> None:
        """Show error message to user"""
        QMessageBox.warning(self, "Input Error", message)