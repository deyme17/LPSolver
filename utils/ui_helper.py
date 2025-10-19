from typing import Optional, List
from PyQt6.QtWidgets import QLabel, QLineEdit, QSpinBox, QLayout, QTextEdit, QTableWidget, QTableWidgetItem
from utils.constants import ResultConstants, SolutionStatus, StatusColor
from utils.interfaces import ITable
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
    """Handles displaying LP tables (Simplex, Dual, etc.) in a QTableWidget."""
    def __init__(self, table_widget: QTableWidget):
        """
        Args:
            table_widget (QTableWidget): The table widget instance in the GUI.
        """
        self.table_widget = table_widget

    def display_table(self, table: Optional[ITable]) -> None:
        """
        Display the current (latest) simplex table.
        Args:
            table (ITable | None): The simplex or dual table to display.
        """
        if not table:
            self.clear()
            return

        table_data = table.get_table()
        self._render_table(table_data)

    def display_iteration(self, table: ITable, iteration_index: int) -> None:
        """
        Display a specific iteration from the simplex table history.

        Args:
            table (ITable): The simplex table instance (implements iteration history).
            iteration_index (int): Index of the iteration to display (0-based).
        """
        history = table.get_full_history()
        if not history:
            self.clear()
            return

        if iteration_index < 0 or iteration_index >= len(history):
            raise IndexError(
                f"Iteration index {iteration_index} out of range "
                f"(0..{len(history)-1})"
            )

        self._render_table(history[iteration_index])

    def _render_table(self, table_data: dict) -> None:
        """Internal helper to render a table structure."""
        headers: List[str] = table_data.get("headers", [])
        data: List[List[str]] = table_data.get("data", [])

        self._setup_dimensions(len(data), len(headers))
        self.table_widget.setHorizontalHeaderLabels(headers)
        self._fill_table(data)

    def _setup_dimensions(self, rows: int, cols: int) -> None:
        """Configure row and column counts before filling the table."""
        self.table_widget.clear()
        self.table_widget.setRowCount(rows)
        self.table_widget.setColumnCount(cols)

    def _fill_table(self, data: List[List]) -> None:
        """Fill the table widget with simplex data."""
        for i, row in enumerate(data):
            for j, value in enumerate(row):
                item = QTableWidgetItem(str(value) if value is not None else "")
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table_widget.setItem(i, j, item)

    def clear(self) -> None:
        """Completely clear the table widget."""
        self.table_widget.clear()
        self.table_widget.setRowCount(0)
        self.table_widget.setColumnCount(0)