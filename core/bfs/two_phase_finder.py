import numpy as np
from typing import List, Optional
from utils import LPProblem, BFSolution, IBFSFinder, ConstraintData
from utils.constants import ConstraintOperator as CO
from utils.constants import OptimizationType as OPT
from ..simplex_table import SimplexTable

EPSILON = 1e-10


class TwoPhase_BFSFinder(IBFSFinder):
    """
    Finds initial BFS using the Two-Phase Simplex method.

    Phase 1: build a temporary LPProblem with artificial variables and
             solve it using SimplexTable: max -sum(artificials).
    Phase 2: Is held in SimplexAlgorithm.
    """
    def __init__(self, max_iterations: int = 10_000) -> None:
        self.max_iterations = max_iterations
        self.phase1_table: Optional[SimplexTable] = None

    def find_initial_bfs(self, standard_form: LPProblem) -> BFSolution:
        self.phase1_table = None

        m = len(standard_form.constraints)
        n = standard_form.variables_count

        needs_artificial = self._rows_needing_artificial(standard_form, m, n)
        num_art = sum(needs_artificial)
        if num_art == 0:
            return self._plain_slack_bfs(standard_form, m, n)

        phase1_problem = self._build_phase1_problem(standard_form, m, n, num_art, needs_artificial)
        phase1_bfs = self._build_phase1_bfs(phase1_problem, m, n, num_art, needs_artificial)
        self.phase1_table = SimplexTable(phase1_problem, phase1_bfs)
        self._run_phase1(self.phase1_table)

        self._remove_artificials_from_basis(self.phase1_table, n)

        if any(bi >= n and abs(self.phase1_table.b[i]) > EPSILON
               for i, bi in enumerate(self.phase1_table.basis)):
            return BFSolution(basis_indices=[0], basic_values=[-1.0], full_solution=None)

        # phase1_table back into standard_form
        for i, constraint in enumerate(standard_form.constraints):
            constraint.coefficients = list(self.phase1_table.A[i, :n])
            constraint.free_val = float(self.phase1_table.b[i])

        return self._extract_phase2_bfs(self.phase1_table, n)

    # --- helpers ---

    def _build_phase1_problem(self, standard_form: LPProblem, m: int, n: int,
                              num_art: int, needs_artificial: List[bool]) -> LPProblem:
        """
        Create a temporary LPProblem for Phase 1:
          variables = original vars + artificial vars
          objective = maximise -sum(a_i) -> c[original]=0, c[art]=-1
          constraints = original A | identity block
        """
        art_col = 0
        constraints = []

        for i, constraint in enumerate(standard_form.constraints):
            art_cols = [0.0] * num_art
            if needs_artificial[i]:
                art_cols[art_col] = 1.0
                art_col += 1
            new_coefs = list(constraint.coefficients) + art_cols
            constraints.append(ConstraintData(new_coefs, CO.EQ.value, constraint.free_val))

        return LPProblem(
            optimization_type=OPT.MAXIMIZE.value,
            objective_coefficients=[0.0] * n + [-1.0] * num_art,
            constraints=constraints,
            variables_count=n + num_art,
        )

    def _build_phase1_bfs(self, phase1_problem: LPProblem, m: int, n: int,
                          num_art: int, needs_artificial: List[bool]) -> BFSolution:
        """
        Initial BFS for Phase-1 problem: artificials in basis for rows
        that needed them, existing slack column for the rest.
        """
        total = n + num_art
        basis_indices = []
        art_col = 0

        A = np.array([c.coefficients for c in phase1_problem.constraints], dtype=float)

        for i in range(m):
            if needs_artificial[i]:
                basis_indices.append(n + art_col)
                art_col += 1
            else:
                basis_indices.append(self._find_unit_col(A, i, n))

        basic_values = [c.free_val for c in phase1_problem.constraints]
        full_solution = [0.0] * total
        for i, bi in enumerate(basis_indices):
            full_solution[bi] = basic_values[i]

        return BFSolution(
            basis_indices=basis_indices,
            basic_values=basic_values,
            full_solution=full_solution,
        )

    def _run_phase1(self, table: SimplexTable) -> None:
        """Drive the Phase-1 simplex iterations on the given table."""
        for _ in range(self.max_iterations):
            if table.is_optimal():
                break
            entering_col = table.get_entering_variable()
            if entering_col is None:
                break
            leaving_row = table.get_leaving_variable(entering_col)
            if leaving_row is None:
                break
            table.pivot(leaving_row, entering_col)

    def _remove_artificials_from_basis(self, table: SimplexTable, n: int) -> None:
        """
        If an artificial variable (index >= n) remains in the basis with
        value ~0 (degenerate), pivot it out with any available original column.
        Mutates table in-place.
        """
        for i, bi in enumerate(table.basis):
            if bi < n or abs(table.b[i]) > EPSILON:
                continue
            for j in range(n):
                if j not in table.basis and abs(table.A[i, j]) > EPSILON:
                    table.pivot(i, j)
                    break

    def _extract_phase2_bfs(self, table: SimplexTable, n: int) -> BFSolution:
        """Extract a BFSolution for Phase 2, keeping only original variables."""
        phase2_basis = []
        phase2_values = []

        for i, bi in enumerate(table.basis):
            if bi < n:
                phase2_basis.append(int(bi))
                phase2_values.append(float(table.b[i]))

        full_solution = [0.0] * n
        for bi, val in zip(phase2_basis, phase2_values):
            full_solution[bi] = val

        return BFSolution(
            basis_indices=phase2_basis,
            basic_values=phase2_values,
            full_solution=full_solution,
        )

    def _plain_slack_bfs(self, standard_form: LPProblem, m: int, n: int) -> BFSolution:
        """Fallback: all constraints already have slack basis columns."""
        basis_indices = list(range(n - m, n))
        basic_values = [c.free_val for c in standard_form.constraints]
        full_solution = [0.0] * n
        for i, bi in enumerate(basis_indices):
            full_solution[bi] = basic_values[i]
        return BFSolution(
            basis_indices=basis_indices,
            basic_values=basic_values,
            full_solution=full_solution,
        )

    def _rows_needing_artificial(self, problem: LPProblem, m: int, n: int) -> List[bool]:
        """
        Detect rows that have no unit-vector basis column among
        the existing variables (i.e. rows from '=' or '>=' constraints).
        """
        A = np.array([c.coefficients for c in problem.constraints], dtype=float)
        has_basis: List[bool] = [False] * m
        for j in range(n):
            col = A[:, j]
            ones = np.where(np.abs(col - 1.0) < EPSILON)[0]
            zeros = np.where(np.abs(col) < EPSILON)[0]
            if len(ones) == 1 and len(zeros) == m - 1:
                has_basis[ones[0]] = True
        return [not h for h in has_basis]

    def _find_unit_col(self, A: np.ndarray, row: int, n: int) -> int:
        """
        Return the index of a valid basis column for the given row
        within the first n columns.
        """
        m = A.shape[0]
        for j in range(n):
            if abs(A[row][j] - 1.0) >= EPSILON:
                continue
            col = A[:, j]
            ones = np.where(np.abs(col - 1.0) < EPSILON)[0]
            zeros = np.where(np.abs(col) < EPSILON)[0]
            if len(ones) == 1 and len(zeros) == m - 1:
                return j