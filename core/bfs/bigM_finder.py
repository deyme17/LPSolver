import numpy as np
from typing import List
from utils import LPProblem, BFSolution, IBFSFinder, ConstraintData

EPSILON = 1e-10
BIG_M = 1e6


class BigM_BFSFinder(IBFSFinder):
    """
    Finds initial BFS using the Big-M method.

    For constraints that lack an obvious basic variable (i.e. '=' or '>=' rows),
    an artificial variable a_i is introduced with a very large penalty coefficient
    -M in the objective.  After augmentation the artificial variables form an
    identity basis, so a BFSolution can be returned immediately.
    """
    def __init__(self, big_m: float = BIG_M) -> None:
        self.big_m = big_m

    def find_initial_bfs(self, standard_form: LPProblem) -> BFSolution:
        """
        Augment the problem with artificial variables and return the
        modified LPProblem together with the initial BFSolution.
        """
        m = len(standard_form.constraints)
        n = standard_form.variables_count   # variables before augmentation

        needs_artificial = self._rows_needing_artificial(standard_form, m, n)
        num_artificials = sum(needs_artificial)

        if num_artificials == 0:
            basis_indices = list(range(n - m, n))
            basic_values  = [c.free_val for c in standard_form.constraints]
            full_solution  = [0.0] * n
            for i, bi in enumerate(basis_indices):
                full_solution[bi] = basic_values[i]
            return BFSolution(
                basis_indices=basis_indices,
                basic_values=basic_values,
                full_solution=full_solution,
            )

        # objective with -M for coeffs
        standard_form.objective_coefficients += [-self.big_m] * num_artificials

        art_col = 0
        basis_indices: List[int] = []

        for i, constraint in enumerate(standard_form.constraints):
            # add zeros for all artificial columns first
            constraint.coefficients += [0] * num_artificials

            if needs_artificial[i]:
                # place a 1 in the column that belongs to this artificial
                col_index = n + art_col
                constraint.coefficients[col_index] = 1
                basis_indices.append(col_index)
                art_col += 1
            else:
                # find the existing slack column that is basic in this row
                basis_indices.append(self._find_slack_basis_col(constraint, n))

        total_vars = n + num_artificials
        standard_form.variables_count = total_vars

        basic_values  = [c.free_val for c in standard_form.constraints]
        full_solution  = [0.0] * total_vars
        for i, bi in enumerate(basis_indices):
            full_solution[bi] = basic_values[i]

        return BFSolution(
            basis_indices=basis_indices,
            basic_values=basic_values,
            full_solution=full_solution,
        )

    # --- helpers ---

    def _rows_needing_artificial(self, problem: LPProblem, m: int, n: int) -> List[bool]:
        """
        Determine which rows do NOT already have a unit-vector basis column
        among the existing variables (i.e. slack columns from '<=' constraints).

        A row i has a ready basis column j if:
          - A[i, j] == 1
          - A[k, j] == 0 for all k != i
        """
        A = np.array([c.coefficients for c in problem.constraints], dtype=float)
        has_basis = [False] * m

        for j in range(n):
            col = A[:, j]
            ones  = np.where(np.abs(col - 1.0) < EPSILON)[0]
            zeros = np.where(np.abs(col)       < EPSILON)[0]
            if len(ones) == 1 and len(zeros) == m - 1:
                has_basis[ones[0]] = True

        return [not h for h in has_basis]

    def _find_slack_basis_col(self, constraint: ConstraintData, n: int) -> int:
        """Return the column index of the unit-vector basis for this row."""
        for j in range(n - 1, -1, -1):
            if abs(constraint.coefficients[j] - 1.0) < EPSILON:
                return j
        # should not happen if standard_form is correct
        return n - 1