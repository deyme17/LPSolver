from abc import ABC, abstractmethod
from typing import Any, Dict, List
from utils import BFSolution, LPProblem
import numpy as np


class ITable(ABC):
    """Interface for LP algorithm tables (Simplex, Dual, etc.)"""
    def __init__(self, problem: LPProblem, bfs: BFSolution):
        """
        Initialize simplex table based on the given problem and BFS.
        Args:
            problem (LPProblem): Problem in standard form (Ax = b, x >= 0)
            bfs (BFSolution): Basic feasible solution (basis indices, values)
        """
        self.problem = problem
        self.bfs = bfs

        self.A = np.array(problem.get_A_matrix(), dtype=float)
        self.b = np.array(problem.get_b_vector(), dtype=float)
        self.c = np.array(problem.objective_coefficients, dtype=float)
        self.basis = np.array(bfs.basis_indices, dtype=int)

        self.headers = self._build_headers()
        self.table = self._build_table()
        self.iterations: List[Dict[str, Any]] = [
            {"headers": self.headers, "data": self.table}
        ]

    @abstractmethod
    def _build_headers(self) -> List[str]:
        """Return table headers."""
        pass

    @abstractmethod
    def _build_table(self) -> List[List[Any]]:
        """Return initial table data."""
        pass

    def get_table(self) -> Dict[str, Any]:
        """
        Return the current simplex table representation.
        Returns:
            Dict[str, Any]: {
                "headers": ["X_basis", "ci/cj", "B", "A1", ..., "An", "Q"],
                "data": [[...], [...], ...]
            }
        """
        return {
            "headers": self.headers,
            "data": self.table
        }

    def get_full_history(self) -> List[Dict[str, Any]]:
        """
        Return the full iteration history of simplex tables.
        Returns:
            List[Dict[str, Any]]: [
                {"headers": [...], "data": [...]},
                {"headers": [...], "data": [...]}
            ]
        """
        return self.iterations



class SimplexTable(ITable):
    """Concrete table structure for the Simplex algorithm."""
    def _build_headers(self) -> List[str]:
        n = self.problem.variables_count
        return ["X_basis", "ci", "B"] + [f"A{i+1}" for i in range(n)] + ["Q"]

    def _build_table(self) -> List[List[Any]]:
        """
        Build the initial simplex table.
        Each row corresponds to a basic variable, and the last row contains Δj = cj - zj.
        """
        A, b, c, basis = self.A, self.b, self.c, self.basis
        rows: List[List[Any]] = []

        # basis rows
        for i, bi in enumerate(basis):
            cb = c[bi]
            Ai = A[i]
            Bi = b[i]

            # Q
            positive_A = Ai > 0
            Q_val = min(Bi / Ai[positive_A]) if np.any(positive_A) else None

            row = [f"x{bi + 1}", cb, Bi] + list(Ai) + [Q_val]
            rows.append(row)

        # delta
        cB = c[basis]
        z = np.dot(cB, A)
        delta = c - z
        z0 = np.dot(cB, b)

        footer = ["Δj = cj - zj", "-", z0] + list(delta) + ["-"]
        rows.append(footer)

        return rows