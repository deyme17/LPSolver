from typing import Any, List, Optional
from utils import ITable
import numpy as np

EPSILON = 1e-10


class SimplexTable(ITable):
    """Concrete table structure for the Simplex algorithm."""
    def _build_headers(self) -> List[str]:
        n = self.problem.variables_count
        return ["X_basis", "ci", "B"] + [f"A{i+1}" for i in range(n)] + ["Q"]

    def _build_table(self) -> List[List[Any]]:
        """Build the current simplex table."""
        rows: List[List[Any]] = []
        
        # basis rows
        for i, bi in enumerate(self.basis):
            cb = float(self.c[bi])
            Ai = self.A[i]
            Bi = float(self.b[i])
            
            row = [f"A{bi + 1}", cb, Bi] + [float(x) for x in Ai] + ["-"]
            rows.append(row)
        
        # delta row
        delta = self._compute_delta()
        z0 = self.get_objective_value()
        
        footer = ["Δj = cj - zj", "-", z0] + [float(d) for d in delta] + ["-"]
        rows.append(footer)
        
        return rows
    
    def is_optimal(self) -> bool:
        """    
        Check if current solution is optimal.
        For maximization: all delta_j <= 0
        """
        delta = self._compute_delta()
        return np.all(delta <= EPSILON)
    
    def is_unbounded(self, entering_col: int) -> bool:
        """
        Check if problem is unbounded for given entering variable.
        """
        return self.get_leaving_variable(entering_col) is None

    def _compute_delta(self) -> np.ndarray:
        """
        Compute reduced costs: delta_j = c_j - z_j
        where z_j = cB^T * A_j
        """
        cB = self.c[self.basis]
        z = cB @ self.A
        return self.c - z
    
    def get_entering_variable(self) -> Optional[int]:
        """
        Select entering variable (most positive delta for max).
        Returns:
            Column index of entering variable, or None if optimal
        """
        delta = self._compute_delta()
        positive_delta = delta > EPSILON
        if not np.any(positive_delta):
            return None
        return int(np.argmax(positive_delta))
    
    def get_leaving_variable(self, entering_col: int) -> Optional[int]:
        """
        Perform min-ratio test to find leaving variable.
        Args:
            entering_col: Index of entering variable column
        Returns:
            Row index of leaving variable, or None if unbounded
        """
        column = self.A[:, entering_col]

        positive = column > EPSILON
        if not any(positive):
            return None
        
        ratios = np.where(positive, self.b / column, np.inf)
        leaving_row = int(np.argmin(ratios))

        return leaving_row if ratios[leaving_row] != np.inf else None
    
    def pivot(self, leaving_row: int, entering_col: int) -> None:
        """
        Perform pivot operation using 2x2 determinant formula.
        For each element A[i,j]:
            new_A[i,j] = (A[i,j] * pivot - A[i,entering_col] * A[leaving_row,j]) / pivot
        Args:
            leaving_row: Row index of leaving variable
            entering_col: Column index of entering variable
        """
        pivot_element = self.A[leaving_row, entering_col]
        if abs(pivot_element) < EPSILON:
            raise ValueError(f"Pivot element too close to zero: {pivot_element}")
        
        pivot_row = self.A[leaving_row].copy()
        pivot_b = self.b[leaving_row]
        entering_col_vec = self.A[:, entering_col].copy()
        
        # apply method with determinants 2x2
        # new = (current * pivot - entering_col * pivot_row) / pivot
        self.A = (self.A * pivot_element - np.outer(entering_col_vec, pivot_row))   / pivot_element
        self.b = (self.b * pivot_element - entering_col_vec * pivot_b)              / pivot_element

        # update basis
        self.basis[leaving_row] = entering_col
        
        # update table
        self.table = self._build_table()
        self.save_iteration()

    def save_iteration(self) -> None:
        """Save current table state to iteration history."""
        self.iterations.append({
            "headers": self.headers,
            "data": self.table
        })

    def get_solution_vector(self) -> List[float]:
        """
        Extract full solution vector (all variables).
        Non-basic variables = 0, basic variables from b.
        """
        n = self.problem.variables_count
        solution = [0.0] * n
        
        for i, var_index in enumerate(self.basis):
            if var_index < n:
                solution[var_index] = float(self.b[i])
        
        return solution

    def get_objective_value(self) -> float:
        """Get current objective function value: cB^T * b"""
        cB = self.c[self.basis]
        return float(cB @ self.b)