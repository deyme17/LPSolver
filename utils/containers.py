from typing import List, Optional
from dataclasses import dataclass, field

@dataclass
class ConstraintData:
    """Container for single constraint data"""
    coefficients: List[float]
    operator: str
    free_val: float


@dataclass
class LPProblem:
    """Container for Linear Programming problem data"""
    success: bool
    optimization_type: str = ""
    objective_coefficients: List[float] = field(default_factory=list)
    constraints: List[ConstraintData] = field(default_factory=list)
    variables_count: int = 0
    error: str = ""


@dataclass
class OptimizationResult:
    """Container for optimization results"""
    status: str
    optimal_value: Optional[float] = None
    solution: Optional[List[float]] = None
    table: Optional[List[List[float]]] = None
    error_message: Optional[str] = None