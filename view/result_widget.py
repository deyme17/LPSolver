from typing import List, Optional
from PyQt6.QtWidgets import (
    QGroupBox, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
)
from utils import ITable
from utils import (
    ResultUIHelper, SimplexTableManager,
    LPResult, SolutionStatus, ResultFormatter
)


class ResultSection(QGroupBox):
    """Widget for displaying optimization results and simplex table iterations"""
    def __init__(self) -> None:
        super().__init__("Results")
        self.current_iteration = 0
        self.table_ref: Optional[ITable] = None
        self._init_widgets()
        self._init_ui()
    
    def _init_widgets(self) -> None:
        """Initialize widgets"""
        self.status_label = ResultUIHelper.create_status_label()
        self.optimal_value_label = ResultUIHelper.create_value_label()
        self.solution_text = ResultUIHelper.create_solution_text()
        self.simplex_table = ResultUIHelper.create_table()
        self.table_manager = SimplexTableManager(self.simplex_table)

        # Navigation buttons for iteration control
        self.prev_btn = QPushButton("<<")
        self.next_btn = QPushButton(">>")
        self.prev_btn.clicked.connect(self._show_prev_iteration)
        self.next_btn.clicked.connect(self._show_next_iteration)
        self.prev_btn.setEnabled(False)
        self.next_btn.setEnabled(False)
    
    def _init_ui(self) -> None:
        """Initialize the results section UI"""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        
        layout.addWidget(self.status_label)
        layout.addLayout(self._create_optimal_value_layout())
        layout.addWidget(QLabel("Solution:"))
        layout.addWidget(self.solution_text)
        layout.addWidget(QLabel("Simplex Table:"))
        layout.addWidget(self.simplex_table)
        layout.addLayout(self._create_iteration_controls())
    
    def _create_optimal_value_layout(self) -> QHBoxLayout:
        """Create optimal value display layout"""
        layout = QHBoxLayout()
        layout.addWidget(QLabel("Optimal Value:"))
        layout.addWidget(self.optimal_value_label)
        layout.addStretch()
        return layout

    def _create_iteration_controls(self) -> QHBoxLayout:
        """Create iteration navigation buttons layout"""
        layout = QHBoxLayout()
        layout.addStretch()
        layout.addWidget(self.prev_btn)
        layout.addWidget(self.next_btn)
        layout.addStretch()
        return layout
    
    # main interface
    def display_results(self, result: LPResult) -> None:
        """Display optimization results"""
        if result.error_message:
            self.display_error(result.error_message)
            return
        
        self._update_status(result.status)
        self._update_optimal_value(result.optimal_value)
        self._update_solution(result.solution)
        self._update_table(result.table)
    
    def display_error(self, error_message: str) -> None:
        """Display an error message"""
        self._update_status(SolutionStatus.ERROR.value)
        self.optimal_value_label.setText("-")
        self.solution_text.setPlainText(error_message)
        self.table_manager.clear()
        self._toggle_iteration_controls(False)
    
    def clear(self) -> None:
        """Clear all result displays"""
        self.status_label.setText("No solution yet")
        self.status_label.setStyleSheet("color: #aaaaaa; font-style: italic;")
        self.optimal_value_label.setText("-")
        self.solution_text.clear()
        self.table_manager.clear()
        self._toggle_iteration_controls(False)
    
    # updates
    def _update_status(self, status: str) -> None:
        color = ResultUIHelper.get_status_color(status)
        formatted_status = ResultFormatter.format_status(status)
        self.status_label.setText(f"Status: {formatted_status}")
        self.status_label.setStyleSheet(f"color: {color}; font-weight: bold;")
    
    def _update_optimal_value(self, value: Optional[float]) -> None:
        formatted_value = ResultFormatter.format_optimal_value(value)
        self.optimal_value_label.setText(formatted_value)
    
    def _update_solution(self, solution: Optional[List[float]]) -> None:
        if solution:
            formatted_solution = ResultFormatter.format_solution(solution)
            self.solution_text.setPlainText(formatted_solution)
        else:
            self.solution_text.clear()
    
    def _update_table(self, table: Optional[ITable]) -> None:
        self.table_ref = table
        self.current_iteration = 0

        if not table:
            self.table_manager.clear()
            self._toggle_iteration_controls(False)
            return
        
        self.table_manager.display_table(table)
        self._toggle_iteration_controls(True)

    # iter
    def _show_next_iteration(self) -> None:
        """Display the next iteration"""
        if not self.table_ref:
            return
        history = self.table_ref.get_full_history()
        if self.current_iteration < len(history) - 1:
            self.current_iteration += 1
            self.table_manager.display_iteration(self.table_ref, self.current_iteration)
        self._update_iteration_buttons()

    def _show_prev_iteration(self) -> None:
        """Display the previous iteration"""
        if not self.table_ref:
            return
        if self.current_iteration > 0:
            self.current_iteration -= 1
            self.table_manager.display_iteration(self.table_ref, self.current_iteration)
        self._update_iteration_buttons()
    
    def _toggle_iteration_controls(self, enable: bool) -> None:
        self.prev_btn.setEnabled(enable)
        self.next_btn.setEnabled(enable)
    
    def _update_iteration_buttons(self) -> None:
        """Enable/disable buttons based on iteration index"""
        if not self.table_ref:
            self._toggle_iteration_controls(False)
            return

        history_len = len(self.table_ref.get_full_history())
        self.prev_btn.setEnabled(self.current_iteration > 0)
        self.next_btn.setEnabled(self.current_iteration < history_len - 1)