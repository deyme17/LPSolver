from typing import Optional, List
from PyQt6.QtWidgets import QLabel, QLineEdit, QSpinBox, QLayout, QTextEdit, QTableWidget, QTableWidgetItem
from utils.constants import ResultConstants, SolutionStatus, StatusColor
from utils.formatters import ResultFormatter
from PyQt6.QtCore import Qt


class UIHelper:
    """Helper methods for UI operations"""
    @staticmethod
    def clear_layout(layout: QLayout) -> None:
        """Clear all widgets from layout"""
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
    
    @staticmethod
    def create_numeric_input(placeholder: str = "0", max_width: int = 70) -> QLineEdit:
        """Factory method for numeric input fields"""
        line_edit = QLineEdit()
        line_edit.setPlaceholderText(placeholder)
        line_edit.setMaximumWidth(max_width)
        return line_edit
    
    @staticmethod
    def create_label(text: str, max_width: Optional[int] = None, 
                    style: Optional[str] = None) -> QLabel:
        """Factory method for labels"""
        label = QLabel(text)
        if max_width:
            label.setMaximumWidth(max_width)
        if style:
            label.setStyleSheet(style)
        return label
    
    @staticmethod
    def create_spinbox(min_val: int, max_val: int, default: int, 
                      max_width: int = 100) -> QSpinBox:
        """Factory method for spinboxes"""
        spinbox = QSpinBox()
        spinbox.setMinimum(min_val)
        spinbox.setMaximum(max_val)
        spinbox.setValue(default)
        spinbox.setMaximumWidth(max_width)
        return spinbox
    

class ResultUIHelper:
    """Helper methods for result UI operations"""
    
    @staticmethod
    def create_status_label(text: str = "No solution yet") -> QLabel:
        """Create status label with default styling"""
        label = QLabel(text)
        label.setStyleSheet("color: #aaaaaa; font-style: italic;")
        return label
    
    @staticmethod
    def create_value_label(text: str = "—") -> QLabel:
        """Create value label with bold styling"""
        label = QLabel(text)
        label.setStyleSheet("font-weight: bold; color: #4CAF50;")
        return label
    
    @staticmethod
    def create_solution_text() -> QTextEdit:
        """Create solution text widget"""
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setMaximumHeight(ResultConstants.SOLUTION_TEXT_HEIGHT)
        text_edit.setMaximumWidth(ResultConstants.SOLUTION_TEXT_WIDTH)
        text_edit.setPlaceholderText("Variable values will appear here...")
        return text_edit
    
    @staticmethod
    def create_table() -> QTableWidget:
        """Create simplex table widget"""
        table = QTableWidget()
        table.setMaximumHeight(ResultConstants.TABLE_MAX_HEIGHT)
        return table
    
    @staticmethod
    def get_status_color(status: str) -> str:
        """Get color for status"""
        try:
            status_enum = SolutionStatus(status.lower())
            return StatusColor[status_enum.name].value
        except (ValueError, KeyError):
            return StatusColor.UNKNOWN.value
        

class SimplexTableManager:
    """Manages simplex table display"""
    
    def __init__(self, table_widget: QTableWidget):
        self.table_widget = table_widget
    
    def display_table(self, table: Optional[List[List[float]]]) -> None:
        """Display the simplex table"""
        if not table:
            self.clear()
            return
        
        rows = len(table)
        cols = len(table[0]) if rows > 0 else 0
        
        self._setup_dimensions(rows, cols)
        self._fill_table(table, rows, cols)
    
    def _setup_dimensions(self, rows: int, cols: int) -> None:
        """Setup table dimensions"""
        self.table_widget.setRowCount(rows)
        self.table_widget.setColumnCount(cols)
    
    def _fill_table(self, table: List[List[float]], rows: int, cols: int) -> None:
        """Fill table with values"""
        for i in range(rows):
            for j in range(cols):
                self._set_cell(i, j, table[i][j])
    
    def _set_cell(self, row: int, col: int, value: float) -> None:
        """Set single cell value"""
        formatted_value = ResultFormatter.format_table_value(value)
        item = QTableWidgetItem(formatted_value)
        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self.table_widget.setItem(row, col, item)
    
    def clear(self) -> None:
        """Clear table"""
        self.table_widget.clear()
        self.table_widget.setRowCount(0)
        self.table_widget.setColumnCount(0)