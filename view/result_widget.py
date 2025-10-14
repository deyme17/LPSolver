from typing import List, Optional
from PyQt6.QtWidgets import (
    QGroupBox, QVBoxLayout, QHBoxLayout, QLabel, 
)
from core import ITable
from utils import (
    ResultUIHelper, SimplexTableManager,
    LPResult, SolutionStatus,
    ResultFormatter
)


class ResultSection(QGroupBox):
    """Widget for displaying optimization results"""
    
    def __init__(self) -> None:
        super().__init__("Results")
        self._init_widgets()
        self._init_ui()
    
    def _init_widgets(self) -> None:
        """Initialize widgets"""
        self.status_label = ResultUIHelper.create_status_label()
        self.optimal_value_label = ResultUIHelper.create_value_label()
        self.solution_text = ResultUIHelper.create_solution_text()
        self.simplex_table = ResultUIHelper.create_table()
        self.table_manager = SimplexTableManager(self.simplex_table)
    
    def _init_ui(self) -> None:
        """Initialize the results section UI"""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        
        layout.addWidget(self.status_label)
        layout.addLayout(self._create_optimal_value_layout())
        layout.addWidget(QLabel("Solution:"))
        layout.addWidget(self.solution_text)
        layout.addWidget(QLabel("Final Simplex table:"))
        layout.addWidget(self.simplex_table)
    
    def _create_optimal_value_layout(self) -> QHBoxLayout:
        """Create optimal value display layout"""
        layout = QHBoxLayout()
        layout.addWidget(QLabel("Optimal Value:"))
        layout.addWidget(self.optimal_value_label)
        layout.addStretch()
        return layout
    
    def display_results(self, result: LPResult) -> None:
        """Display optimization results"""
        if result.error_message:
            self.display_error(result.error_message)
            return
        
        self._update_status(result.status)
        self._update_optimal_value(result.optimal_value)
        self._update_solution(result.solution)
        self._update_table(result.table)
    
    def _update_status(self, status: str) -> None:
        """Update status label"""
        color = ResultUIHelper.get_status_color(status)
        formatted_status = ResultFormatter.format_status(status)
        
        self.status_label.setText(f"Status: {formatted_status}")
        self.status_label.setStyleSheet(f"color: {color}; font-weight: bold;")
    
    def _update_optimal_value(self, value: Optional[float]) -> None:
        """Update optimal value label"""
        formatted_value = ResultFormatter.format_optimal_value(value)
        self.optimal_value_label.setText(formatted_value)
    
    def _update_solution(self, solution: Optional[List[float]]) -> None:
        """Update solution text"""
        if solution:
            formatted_solution = ResultFormatter.format_solution(solution)
            self.solution_text.setPlainText(formatted_solution)
        else:
            self.solution_text.clear()
    
    def _update_table(self, table: Optional[ITable]) -> None:
        """Update simplex table"""
        self.table_manager.display_table(table)
    
    def display_error(self, error_message: str) -> None:
        """Display an error message"""
        self._update_status(SolutionStatus.ERROR.value)
        self.optimal_value_label.setText("—")
        self.solution_text.setPlainText(error_message)
        self.table_manager.clear()
    
    def clear(self) -> None:
        """Clear all result displays"""
        self.status_label.setText("No solution yet")
        self.status_label.setStyleSheet("color: #aaaaaa; font-style: italic;")
        self.optimal_value_label.setText("—")
        self.solution_text.clear()
        self.table_manager.clear()