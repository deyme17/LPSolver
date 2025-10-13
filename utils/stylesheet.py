class StyleSheet:   
    DARK_STYLE = """
        QMainWindow {
            background-color: #1e1e1e;
            color: #ffffff;
        }
        
        QWidget {
            background-color: #1e1e1e;
            color: #ffffff;
        }
        
        QGroupBox {
            color: #ffffff;
            border: 2px solid #3d3d3d;
            border-radius: 5px;
            margin-top: 10px;
            padding-top: 10px;
        }
        
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 3px 0 3px;
        }
        
        QLabel {
            color: #ffffff;
        }
        
        QLineEdit {
            background-color: #2d2d2d;
            color: #ffffff;
            border: 1px solid #3d3d3d;
            border-radius: 4px;
            padding: 5px;
            selection-background-color: #0d47a1;
        }
        
        QComboBox {
            background-color: #2d2d2d;
            color: #ffffff;
            border: 1px solid #3d3d3d;
            border-radius: 4px;
            padding: 5px;
        }
        
        QComboBox::drop-down {
            border: none;
            background-color: #3d3d3d;
        }
        
        QComboBox QAbstractItemView {
            background-color: #2d2d2d;
            color: #ffffff;
            selection-background-color: #0d47a1;
            border: 1px solid #3d3d3d;
        }
        
        QPushButton {
            background-color: #0d47a1;
            color: #ffffff;
            border: none;
            border-radius: 4px;
            padding: 8px 16px;
            font-weight: bold;
            font-size: 12px;
        }
        
        QPushButton:hover {
            background-color: #1565c0;
        }
        
        QPushButton:pressed {
            background-color: #0a3d91;
        }
        
        QPushButton:disabled {
            background-color: #555555;
            color: #888888;
        }
        
        QTableWidget {
            background-color: #2d2d2d;
            gridline-color: #3d3d3d;
            border: 1px solid #3d3d3d;
            border-radius: 4px;
        }
        
        QTableWidget::item {
            background-color: #2d2d2d;
            color: #ffffff;
            padding: 5px;
        }
        
        QTableWidget::item:selected {
            background-color: #0d47a1;
            color: #ffffff;
        }
        
        QHeaderView::section {
            background-color: #3d3d3d;
            color: #ffffff;
            padding: 5px;
            border: 1px solid #4d4d4d;
        }
        
        QTextEdit {
            background-color: #2d2d2d;
            color: #ffffff;
            border: 1px solid #3d3d3d;
            border-radius: 4px;
            padding: 5px;
        }
        
        QScrollBar:vertical {
            background-color: #2d2d2d;
            width: 12px;
            border: none;
        }
        
        QScrollBar::handle:vertical {
            background-color: #555555;
            border-radius: 6px;
            min-height: 20px;
        }
        
        QScrollBar::handle:vertical:hover {
            background-color: #666666;
        }
        
        QScrollBar:horizontal {
            background-color: #2d2d2d;
            height: 12px;
            border: none;
        }
        
        QScrollBar::handle:horizontal {
            background-color: #555555;
            border-radius: 6px;
            min-width: 20px;
        }
        
        QScrollBar::handle:horizontal:hover {
            background-color: #666666;
        }
    """