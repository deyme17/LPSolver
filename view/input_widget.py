from typing import List, Optional, Tuple
from PyQt6.QtWidgets import (
    QGroupBox, QHBoxLayout, QVBoxLayout, QLabel, QLineEdit, 
    QComboBox, QPushButton, QScrollArea, QWidget
)
from utils import (
    UIHelper, InputWidgetConstants, InputValidator,
    ConstraintData, LPProblem,
    ConstraintOperator, OptimizationType
)


class InputSection(QGroupBox):
    """Widget for problem input and configuration"""
    def __init__(self) -> None:
        super().__init__("Problem Configuration")
        self.constraint_rows: List[ConstraintRow] = []
        self.objective_inputs: List[QLineEdit] = []
        self._init_widgets()
        self._init_ui()
    
    def _init_widgets(self) -> None:
        """Initialize widgets"""
        self.optimization_type = self._create_optimization_combo()
        self.var_count = UIHelper.create_spinbox(
            1, InputWidgetConstants.MAX_VARIABLES, 
            InputWidgetConstants.DEFAULT_VARIABLES, 
            InputWidgetConstants.SPINBOX_WIDTH
        )
        self.constraint_count = UIHelper.create_spinbox(
            1, InputWidgetConstants.MAX_CONSTRAINTS, 
            InputWidgetConstants.DEFAULT_CONSTRAINTS, 
            InputWidgetConstants.SPINBOX_WIDTH
        )
        self.generate_btn = QPushButton("Create Form")
    
    def _init_ui(self) -> None:
        """Initialize the input section UI"""
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        
        # config
        config_layout = self._create_config_layout()
        layout.addLayout(config_layout)
        
        self.generate_btn.clicked.connect(self.update)
        layout.addWidget(self.generate_btn)
        
        layout.addSpacing(15)
        
        # obj func
        layout.addWidget(self._create_objective_label())
        
        objective_container = QWidget()
        objective_container.setMaximumWidth(InputWidgetConstants.MAX_OBJECTIVE_WIDTH)
        self.objective_layout = QHBoxLayout(objective_container)
        self.objective_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(objective_container)
        
        layout.addSpacing(15)
        
        # constraints
        layout.addWidget(self._create_constraints_label())
        self.constraints_scroll = self._create_constraints_scroll()
        layout.addWidget(self.constraints_scroll)
    
    def _create_optimization_combo(self) -> QComboBox:
        """Create optimization type combo box"""
        combo = QComboBox()
        combo.addItems([opt.value for opt in OptimizationType])
        combo.setMaximumWidth(InputWidgetConstants.COMBO_WIDTH)
        return combo
    
    def _create_config_layout(self) -> QHBoxLayout:
        """Create configuration panel layout"""
        config_layout = QHBoxLayout()
        config_layout.setSpacing(30)
        
        # opt type
        config_layout.addWidget(QLabel("Optimization Type:"))
        config_layout.addWidget(self.optimization_type)
        
        # vars
        config_layout.addWidget(QLabel("Variables (n):"))
        config_layout.addWidget(self.var_count)
        
        # constrs
        config_layout.addWidget(QLabel("Constraints (m):"))
        config_layout.addWidget(self.constraint_count)
        
        return config_layout
    
    @staticmethod
    def _create_objective_label() -> QLabel:
        """Create objective function label"""
        return UIHelper.create_label(
            "Objective Function: f = a1*x1 + a2*x2 + ...",
            style="color: #aaaaaa; font-style: italic;"
        )
    
    @staticmethod
    def _create_constraints_label() -> QLabel:
        """Create constraints label"""
        return UIHelper.create_label(
            "Constraints: aij*xij {<=,>=,=} bi",
            style="color: #aaaaaa; font-style: italic;"
        )
    
    def _create_constraints_scroll(self) -> QScrollArea:
        """Create scrollable constraints area"""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        
        self.constraints_container = QWidget()
        self.constraints_layout = QVBoxLayout(self.constraints_container)
        self.constraints_layout.setSpacing(10)
        self.constraints_layout.setContentsMargins(5, 5, 5, 5)
        
        scroll.setWidget(self.constraints_container)
        return scroll
    
    def update(self) -> None:
        """Update input form based on selected parameters"""
        var_count = self.var_count.value()
        constraint_count = self.constraint_count.value()
        
        self._update_objective_inputs(var_count)
        self._update_constraint_rows(var_count, constraint_count)
    
    def _update_objective_inputs(self, var_count: int) -> None:
        """Update objective function coefficient inputs"""
        UIHelper.clear_layout(self.objective_layout)
        self.objective_inputs.clear()
        
        for i in range(var_count):
            label, line_edit = self._create_coefficient_input(i + 1)
            self.objective_inputs.append(line_edit)
            self.objective_layout.addWidget(label)
            self.objective_layout.addWidget(line_edit)
        
        self.objective_layout.addStretch()
    
    def _create_coefficient_input(self, index: int) -> Tuple[QLabel, QLineEdit]:
        """Factory method for creating coefficient inputs"""
        label = UIHelper.create_label(f"a{index}:", InputWidgetConstants.LABEL_WIDTH)
        line_edit = UIHelper.create_numeric_input("0", InputWidgetConstants.OBJECTIVE_INPUT_WIDTH)
        return label, line_edit
    
    def _update_constraint_rows(self, var_count: int, constraint_count: int) -> None:
        """Update constraint rows"""
        self._cleanup_constraint_rows()
        self._create_new_constraint_rows(var_count, constraint_count)
    
    def _cleanup_constraint_rows(self) -> None:
        """Clean up existing constraint rows"""
        for row in self.constraint_rows:
            row.cleanup()
        self.constraint_rows.clear()
        UIHelper.clear_layout(self.constraints_layout)
    
    def _create_new_constraint_rows(self, var_count: int, constraint_count: int) -> None:
        """Create new constraint rows"""
        for i in range(constraint_count):
            row = ConstraintRow(i + 1, var_count)
            self.constraint_rows.append(row)
            
            row_widget = QWidget()
            row_widget.setLayout(row)
            row_widget.setMaximumWidth(InputWidgetConstants.MAX_CONSTRAINT_ROW_WIDTH)
            self.constraints_layout.addWidget(row_widget)
        
        self.constraints_layout.addStretch()
    
    def get_data(self) -> Tuple[LPProblem, bool, str]:
        """
        Returns:
            LPProblem: Dataclass object containing:
                - optimization_type (str): "max" or "min"
                - objective_coefficients (list[float]): Coefficients of the objective function
                - constraints (list[Constraint]): Parsed constraints
                - variables_count (int): Number of decision variables
            bool: True if extraction and validation succeeded, otherwise False.
            str: Error message if validation failed, empty string otherwise.
        """
        try:
            objective_coeffs = InputValidator.validate_coefficients(
                self.objective_inputs
            )
            constraints_data = self._parse_constraints()
            
            return LPProblem(
                optimization_type=self.optimization_type.currentText(),
                objective_coefficients=objective_coeffs,
                constraints=constraints_data,
                variables_count=len(objective_coeffs)
            ), True, ""
        except ValueError as e:
            return None, False, str(e)
    
    def _parse_constraints(self) -> List[ConstraintData]:
        """Parse constraint rows"""
        constraints = []
        for i, row in enumerate(self.constraint_rows):
            try:
                constraint = row.get_data()
                constraints.append(constraint)
            except ValueError as e:
                raise ValueError(f"Constraint {i+1}: {str(e)}")
        return constraints
    
    def clear(self) -> None:
        """Clear all input fields"""
        for line_edit in self.objective_inputs:
            line_edit.clear()
        
        for row in self.constraint_rows:
            row.clear()


class ConstraintRow(QHBoxLayout):
    """Single constraint row widget"""
    def __init__(self, constraint_num: int, var_count: int) -> None:
        super().__init__()
        self.constraint_num = constraint_num
        self.var_count = var_count
        
        self.coefficient_inputs: List[QLineEdit] = []
        self.operator_combo: Optional[QComboBox] = None
        self.free_vars_input: Optional[QLineEdit] = None
        
        self.setContentsMargins(0, 0, 0, 0)
        self.setSpacing(5)
        
        self._build_row()
    
    def _build_row(self) -> None:
        """Build the constraint row layout"""
        self._add_constraint_label()
        self._add_coefficient_inputs()
        self._add_operator_combo()
        self._add_free_value_input()
        self.addStretch()
    
    def _add_constraint_label(self) -> None:
        """Add constraint number label"""
        label = UIHelper.create_label(
            f"{self.constraint_num}:", 
            InputWidgetConstants.CONSTRAINT_NUM_WIDTH
        )
        self.addWidget(label)
    
    def _add_coefficient_inputs(self) -> None:
        """Add coefficient input fields"""
        for i in range(self.var_count):
            coef_input = UIHelper.create_numeric_input(
                "0", 
                InputWidgetConstants.COEFFICIENT_INPUT_WIDTH
            )
            self.coefficient_inputs.append(coef_input)
            self.addWidget(coef_input)
            
            var_label = UIHelper.create_label(
                f"x{i+1}", 
                InputWidgetConstants.VAR_LABEL_WIDTH
            )
            self.addWidget(var_label)
    
    def _add_operator_combo(self) -> None:
        """Add operator combo box"""
        self.operator_combo = QComboBox()
        self.operator_combo.addItems([op.value for op in ConstraintOperator])
        self.operator_combo.setMaximumWidth(InputWidgetConstants.OPERATOR_WIDTH)
        self.addWidget(self.operator_combo)
    
    def _add_free_value_input(self) -> None:
        """Add free value input field"""
        self.free_vars_input = UIHelper.create_numeric_input(
            "0", 
            InputWidgetConstants.FREE_VAL_WIDTH
        )
        self.addWidget(self.free_vars_input)
    
    def get_data(self) -> ConstraintData:
        """Extract constraint data"""
        coefficients = InputValidator.validate_coefficients(
            self.coefficient_inputs,
            prefix="x_"
        )
        free_val = InputValidator.validate_coefficient(
            self.free_vars_input.text(),
            "free value"
        )
        return ConstraintData(
            coefficients=coefficients,
            operator=self.operator_combo.currentText(),
            free_val=free_val
        )
    
    def clear(self) -> None:
        """Clear all input fields in this row"""
        for line_edit in self.coefficient_inputs:
            line_edit.clear()
        self.free_vars_input.clear()
    
    def cleanup(self) -> None:
        """Clean up resources"""
        UIHelper.clear_layout(self)