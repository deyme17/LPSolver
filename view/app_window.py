
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
)
from PyQt6.QtGui import QFont
from . import InputSection, ResultSection
from utils import StyleSheet



class LPSolverApp(QMainWindow):
    """Main application window for LP solver"""
    def __init__(self, input_section: InputSection, results_section: ResultSection) -> None:
        """
        Initialize the main application window.
        Args:
            input_section: InputSection widget for problem setup
            results_section: ResultSection widget for displaying results
        """
        super().__init__()
        self.setWindowTitle("Linear Programming Solver")
        self.setGeometry(100, 100, 1200, 1000)
        self.setStyleSheet(StyleSheet.DARK_STYLE)
        
        # widgets
        self.input_section = input_section()
        self.results_section = results_section()
        
        self.init_ui()
    
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
        
        # input
        main_layout.addWidget(self.input_section)
        
        # controls
        buttons_layout = self._create_buttons_layout()
        main_layout.addLayout(buttons_layout)
        
        # result
        main_layout.addWidget(self.results_section)
        
        # signals
        self._connect_signals()
    
    def _create_title(self) -> QLabel:
        """Create and configure the title label"""
        title = QLabel("Linear Programming Solver - Simplex Method")
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
    
    def on_clear(self) -> None:
        """Handle clear button click - reset all forms"""
        self.input_section.clear()
        self.results_section.clear()