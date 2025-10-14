from utils import LPProblem, BFSolution
from abc import ABC, abstractmethod


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



class BasicBFSFinder(IBFSFinder):
    """Finds initial BFS assuming all constraints are of the form Ax = b, x >= 0."""
    def find_initial_bfs(self, standard_form: LPProblem) -> BFSolution:
        m = len(standard_form.constraints)
        n = standard_form.variables_count

        basis_indices = list(range(n, n + m))
        basic_values = [c.free_val for c in standard_form.constraints]

        full_solution = [0.0] * (n + m)
        for i, bi in enumerate(basis_indices):
            full_solution[bi] = basic_values[i]

        return BFSolution(
            basis_indices=basis_indices,
            basic_values=basic_values,
            full_solution=full_solution
        )