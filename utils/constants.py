from enum import Enum

# app
class AppConstants:
    WINDOW_TITLE = "Linear Programming Solver"
    WINDOW_SIZE = (900, 700)
    TITLE_FONT_SIZE = 16
    BUTTON_HEIGHT = 40
    BUTTON_FONT_SIZE = 12
    LAYOUT_SPACING = 15
    LAYOUT_MARGINS = 15

# input
class InputWidgetConstants:
    MAX_VARIABLES = 20
    MAX_CONSTRAINTS = 20
    DEFAULT_VARIABLES = 2
    DEFAULT_CONSTRAINTS = 2
    OBJECTIVE_INPUT_WIDTH = 70
    COEFFICIENT_INPUT_WIDTH = 60
    LABEL_WIDTH = 30
    OPERATOR_WIDTH = 60
    FREE_VAL_WIDTH = 70
    CONSTRAINT_NUM_WIDTH = 40
    VAR_LABEL_WIDTH = 25
    COMBO_WIDTH = 150
    SPINBOX_WIDTH = 100
    MAX_CONSTRAINT_ROW_WIDTH = 1000
    MAX_OBJECTIVE_WIDTH = 800

class OptimizationType(Enum):
    MAXIMIZE = "Maximize"
    MINIMIZE = "Minimize"

class ConstraintOperator(Enum):
    LEQ = "<="
    GEQ = ">="
    EQ = "="

# result
class ResultConstants:
    SOLUTION_TEXT_HEIGHT = 100
    SOLUTION_TEXT_WIDTH = 400
    TABLE_MAX_HEIGHT = 600
    DECIMAL_PLACES = 6
    TABLE_DECIMAL_PLACES = 4

class SolutionStatus(Enum):
    OPTIMAL = 'optimal'
    INFEASIBLE = 'infeasible'
    UNBOUNDED = 'unbounded'
    ERROR = 'error'
    UNKNOWN = 'unknown'
    PENDING = 'pending'

class StatusColor(Enum):
    OPTIMAL = '#4CAF50'
    INFEASIBLE = '#f44336'
    UNBOUNDED = '#FF9800'
    ERROR = '#f44336'
    UNKNOWN = '#aaaaaa'
    PENDING = '#aaaaaa'