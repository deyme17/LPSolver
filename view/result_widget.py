from typing import List, Dict, Any
from PyQt6.QtWidgets import (
    QGroupBox, QVBoxLayout, QHBoxLayout, QLabel, 
    QTextEdit, QTableWidget, QTableWidgetItem
)
from PyQt6.QtCore import Qt



STATUS_COLORS = {
    'optimal': '#4CAF50',
    'infeasible': '#f44336',
    'unbounded': '#FF9800',
    'error': '#f44336'
}


class ResultSection(QGroupBox):
    """Widget for displaying optimization results"""
    def __init__(self) -> None:
        super().__init__("Results")
        self.init_ui()
    
    def init_ui(self) -> None:
        """Initialize the results section UI"""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        
        # status label
        self.status_label = QLabel("No solution yet")
        self.status_label.setStyleSheet("color: #aaaaaa; font-style: italic;")
        layout.addWidget(self.status_label)
        
        # optimal value
        optimal_layout = QHBoxLayout()
        optimal_layout.addWidget(QLabel("Optimal Value:"))
        self.optimal_value_label = QLabel("—")
        self.optimal_value_label.setStyleSheet("font-weight: bold; color: #4CAF50;")
        optimal_layout.addWidget(self.optimal_value_label)
        optimal_layout.addStretch()
        layout.addLayout(optimal_layout)
        
        # sulution
        layout.addWidget(QLabel("Solution:"))
        self.solution_text = QTextEdit()
        self.solution_text.setReadOnly(True)
        self.solution_text.setMaximumHeight(20)
        self.solution_text.setMaximumWidth(200)
        self.solution_text.setPlaceholderText("Variable values will appear here...")
        layout.addWidget(self.solution_text)
        
        # table
        layout.addWidget(QLabel("Final Simplex table:"))
        self.simplex_table = QTableWidget()
        self.simplex_table.setMaximumHeight(200)
        layout.addWidget(self.simplex_table)
        
        layout.addStretch()
    
    def display_results(self, results: Dict[str, Any]) -> None:
        """
        Display optimization results.
        Args:
            results: Dictionary {
                'status': str
                'optimal_value': float
                'solution': List[float]
                'tableau': List[List[float]]
            }
        """
        status = results.get('status', 'unknown')
        
        # update status
        color = STATUS_COLORS.get(status, '#aaaaaa')
        self.status_label.setText(f"Status: {status.upper()}")
        self.status_label.setStyleSheet(f"color: {color}; font-weight: bold;")
        
        # update optimal value
        optimal_value = results.get('optimal_value', None)
        if optimal_value is not None:
            self.optimal_value_label.setText(f"{optimal_value:.6f}")
        else:
            self.optimal_value_label.setText("—")
        
        # update solution
        solution = results.get('solution', None)
        if solution is not None:
            solution_text = ""
            
            for i, val in enumerate(solution):
                solution_text += f"x_{i+1} = {val:.6f}\n"
            
            self.solution_text.setPlainText(solution_text.strip())
        else:
            self.solution_text.clear()
        
        # update table
        table = results.get('table', None)
        if table is not None:
            self._display_table(table)
        else:
            self.simplex_table.clear()
            self.simplex_table.setRowCount(0)
            self.simplex_table.setColumnCount(0)
    
    def _display_table(self, table: List[List[float]]) -> None:
        """Display the simplex table in the table widget"""
        if not table:
            return
        
        rows = len(table)
        cols = len(table[0]) if rows > 0 else 0
        
        self.simplex_table.setRowCount(rows)
        self.simplex_table.setColumnCount(cols)
        
        # fill table
        for i in range(rows):
            for j in range(cols):
                item = QTableWidgetItem(f"{table[i][j]:.4f}")
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.simplex_table.setItem(i, j, item)
    
    def display_error(self, error_message: str) -> None:
        """Display an error message"""
        self.status_label.setText("Error")
        self.status_label.setStyleSheet("color: #f44336; font-weight: bold;")
        self.optimal_value_label.setText("—")
        self.solution_text.setPlainText(error_message)
        self.simplex_table.clear()
        self.simplex_table.setRowCount(0)
        self.simplex_table.setColumnCount(0)
    
    def clear(self) -> None:
        """Clear all result displays"""
        self.status_label.setText("No solution yet")
        self.status_label.setStyleSheet("color: #aaaaaa; font-style: italic;")
        self.optimal_value_label.setText("—")
        self.solution_text.clear()
        self.simplex_table.clear()
        self.simplex_table.setRowCount(0)
        self.simplex_table.setColumnCount(0)