from abc import ABC, abstractmethod
from typing import Any, Dict, List
from utils.containers import LPProblem, LPProblem, BFSolution, LPResult
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
    


class IBFSFinder(ABC):
    """Class-interface for finding initial basic feasible solutions (BFS) for linear programming problems."""
    @abstractmethod
    def find_initial_bfs(self, standard_form: LPProblem) -> BFSolution:
        """
        Computes an initial basic feasible solution for linear programming problems.
        Args:
            standard_form (LPProblem): The linear programming problem in standard form
        Returns:
            BFSolution: Basic feasible solution
        """
        pass



class ISimplexAlgorithm(ABC):
    """
    Class-interface for simplex algorithms that's 
    used for solving Linear Progrmmin problem
    """
    @abstractmethod
    def solve_from_bfs(self, standard_form: LPProblem, initial_solution: BFSolution) -> LPResult:
        """
        Solve linear programming problem starting from initial basic feasible solution.
        Args:
            problem (LPProblem): The linear programming problem in standard form to solve
            initial_solution (BFSolution): Initial basic feasible solution containing
        Returns:
            LPResult: The solution containing optimal value and variables
        """
        pass