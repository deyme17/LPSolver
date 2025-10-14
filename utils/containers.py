from typing import List, Optional
from core.simplex_table import ITable
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
    optimization_type: str = ""
    objective_coefficients: List[float] = field(default_factory=list)
    constraints: List[ConstraintData] = field(default_factory=list)
    variables_count: int = 0
    
    def get_A_matrix(self) -> List[List[float]]:
        return [c.coefficients for c in self.constraints]

    def get_b_vector(self) -> List[float]:
        return [c.free_val for c in self.constraints]


@dataclass
class LPResult:
    """Container for Linear Programming problem results"""
    status: str
    optimal_value: Optional[float] = None
    solution: Optional[List[float]] = None
    table: Optional[ITable] = None
    error_message: Optional[str] = None


@dataclass
class BFSolution:
    """Basic Feasible Solution (BFS) representation"""
    basis_indices: List[int]
    basic_values: List[float]
    full_solution: Optional[List[float]] = None

    def is_feasible(self) -> bool:
        return all(val >= 0 for val in self.basic_values)