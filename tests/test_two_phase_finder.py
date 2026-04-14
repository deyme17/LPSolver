"""
Unit tests for TwoPhase_BFSFinder. Thanks to Claude by Anthropic.

TwoPhase_BFSFinder expects a problem already in standard form:
  - all constraints are '='
  - slack columns are already present in the constraint matrix
  - variables_count includes original + slack vars

This mirrors what LPSolver._build_standard_form produces before
calling find_initial_bfs.
"""

import pytest
import numpy as np
from copy import deepcopy

from core.bfs.two_phase_finder import TwoPhase_BFSFinder
from utils import LPProblem, ConstraintData, BFSolution, OptimizationType


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def make_leq_standard(obj, rows, rhs_list):
    """
    Build a standard-form LP where every constraint was originally '<='.
    Adds one slack per row → constraint matrix = [A | I].
    variables_count = len(obj) + m
    """
    m = len(rows)
    n = len(obj)
    constraints = []
    for i, (row, rhs) in enumerate(zip(rows, rhs_list)):
        slack = [0.0] * m
        slack[i] = 1.0
        constraints.append(ConstraintData(list(row) + slack, "=", rhs))
    return LPProblem(
        optimization_type=OptimizationType.MAXIMIZE.value,
        objective_coefficients=list(obj) + [0.0] * m,
        constraints=constraints,
        variables_count=n + m,
    )


def make_eq_standard(obj, rows, rhs_list):
    """
    Build a standard-form LP where all constraints are '='.
    No slack columns — every row needs an artificial.
    variables_count = len(obj)
    """
    constraints = [
        ConstraintData(list(row), "=", rhs)
        for row, rhs in zip(rows, rhs_list)
    ]
    return LPProblem(
        optimization_type=OptimizationType.MAXIMIZE.value,
        objective_coefficients=list(obj),
        constraints=constraints,
        variables_count=len(obj),
    )


# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────

@pytest.fixture
def finder() -> TwoPhase_BFSFinder:
    return TwoPhase_BFSFinder(max_iterations=10_000)


# ══════════════════════════════════════════════════════════════════════
# 1. Plain-slack path (_plain_slack_bfs)
# ══════════════════════════════════════════════════════════════════════

class TestTwoPhase_PlainSlack:
    """All constraints '<=': no Phase-1 needed, returns slack basis directly."""

    def test_two_leq_constraints(self, finder: TwoPhase_BFSFinder):
        """2 original vars + 2 slacks → basis = [2, 3]."""
        problem = make_leq_standard([3.0, 2.0], [[1, 1], [2, 1]], [4, 5])
        result = finder.find_initial_bfs(problem)

        assert result.basis_indices == [2, 3]
        assert result.basic_values == [4, 5]
        assert result.full_solution == [0.0, 0.0, 4.0, 5.0]
        assert result.is_feasible()
        # phase1_table should not be built for the slack path
        assert finder.phase1_table is None

    def test_three_leq_constraints(self, finder: TwoPhase_BFSFinder):
        """3 vars + 3 slacks → basis = [3, 4, 5]."""
        problem = make_leq_standard(
            [1.0, 2.0, 3.0],
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [5, 7, 9],
        )
        result = finder.find_initial_bfs(problem)

        assert result.basis_indices == [3, 4, 5]
        assert result.basic_values == [5, 7, 9]
        assert result.is_feasible()

    def test_single_leq_constraint(self, finder: TwoPhase_BFSFinder):
        """1 var + 1 slack → basis = [1]."""
        problem = make_leq_standard([5.0], [[3]], [12])
        result = finder.find_initial_bfs(problem)

        assert result.basis_indices == [1]
        assert result.basic_values == [12]
        assert result.is_feasible()

    def test_zero_rhs(self, finder: TwoPhase_BFSFinder):
        """Zero RHS is a degenerate but feasible BFS."""
        problem = make_leq_standard([1.0, 1.0], [[1, 0], [0, 1]], [0, 0])
        result = finder.find_initial_bfs(problem)

        assert result.basis_indices == [2, 3]
        assert result.basic_values == [0, 0]
        assert result.is_feasible()   # 0 >= 0 → feasible

    def test_standard_form_not_mutated_in_plain_path(self, finder: TwoPhase_BFSFinder):
        """Plain-slack path must not mutate standard_form."""
        problem = make_leq_standard([3.0, 2.0], [[1, 1], [2, 1]], [4, 5])
        original_coefs = [c.coefficients[:] for c in problem.constraints]
        original_obj   = problem.objective_coefficients[:]
        original_n     = problem.variables_count

        finder.find_initial_bfs(problem)

        assert problem.variables_count == original_n
        assert problem.objective_coefficients == original_obj
        for i, c in enumerate(problem.constraints):
            assert c.coefficients == original_coefs[i]


# ══════════════════════════════════════════════════════════════════════
# 2. Phase-1 path — feasible problems
# ══════════════════════════════════════════════════════════════════════

class TestTwoPhase_Phase1_Feasible:
    """Problems that need Phase-1 and are feasible."""

    def test_single_equality_constraint(self, finder: TwoPhase_BFSFinder):
        """x1 = 5  →  feasible, x1 should be in basis."""
        problem = make_eq_standard([2.0], [[1.0]], [5])
        result = finder.find_initial_bfs(problem)

        assert result.is_feasible()
        assert 0 in result.basis_indices        # x1 driven into basis
        idx = result.basis_indices.index(0)
        assert abs(result.basic_values[idx] - 5.0) < 1e-8

    def test_two_equality_constraints(self, finder: TwoPhase_BFSFinder):
        """x1 = 3, x2 = 4  →  trivial feasible BFS."""
        problem = make_eq_standard([1.0, 2.0], [[1, 0], [0, 1]], [3, 4])
        result = finder.find_initial_bfs(problem)

        assert result.is_feasible()
        sol = result.full_solution
        assert abs(sol[0] - 3.0) < 1e-8
        assert abs(sol[1] - 4.0) < 1e-8

    def test_mixed_leq_and_eq(self, finder: TwoPhase_BFSFinder):
        """
        Standard form with one '<=' slack and one '=' row.
        A = [[1, 1, 1, 0],   ← slack col 2 is basis for row 0
             [2, 1, 0, 0]]   ← no unit col → needs artificial
        """
        constraints = [
            ConstraintData([1.0, 1.0, 1.0, 0.0], "=", 6.0),   # slack row
            ConstraintData([2.0, 1.0, 0.0, 0.0], "=", 8.0),   # no basis col
        ]
        problem = LPProblem(
            optimization_type=OptimizationType.MAXIMIZE.value,
            objective_coefficients=[5.0, 4.0, 0.0, 0.0],
            constraints=constraints,
            variables_count=4,
        )
        result = finder.find_initial_bfs(problem)

        assert result.is_feasible()
        # BFS must satisfy Ax = b within the original 4-column problem
        A = np.array([[1, 1, 1, 0], [2, 1, 0, 0]], dtype=float)
        b = np.array([6.0, 8.0])
        x = np.array(result.full_solution[:4])
        assert np.allclose(A @ x, b, atol=1e-8)

    def test_phase1_table_is_built(self, finder: TwoPhase_BFSFinder):
        """After a Phase-1 run phase1_table must be populated."""
        problem = make_eq_standard([1.0, 1.0], [[1, 1], [1, -1]], [3, 1]) # art_vars required
        finder.find_initial_bfs(problem)

        assert finder.phase1_table is not None
        history = finder.phase1_table.get_full_history()
        assert isinstance(history, list)
        assert len(history) >= 1

    def test_standard_form_not_mutated_in_phase1(self, finder: TwoPhase_BFSFinder):
        """Phase-1 path must NOT mutate standard_form."""
        problem = make_eq_standard([3.0, 5.0], [[1, 0], [0, 1]], [4, 6])
        original_n   = problem.variables_count
        original_obj = problem.objective_coefficients[:]
        original_c0  = problem.constraints[0].coefficients[:]
        original_c1  = problem.constraints[1].coefficients[:]

        finder.find_initial_bfs(problem)

        assert problem.variables_count == original_n
        assert problem.objective_coefficients == original_obj
        assert problem.constraints[0].coefficients == original_c0
        assert problem.constraints[1].coefficients == original_c1

    def test_bfs_satisfies_constraints(self, finder: TwoPhase_BFSFinder):
        """
        General feasible LP: returned x must satisfy Ax = b.
        max  5x1 + 4x2 + 3x3
        s.t. 6x1 + 4x2 + 2x3 = 240   (= constraint, needs artificial)
             3x1 + 2x2 + 5x3 + s1 = 270   (slack row)
        """
        constraints = [
            ConstraintData([6.0, 4.0, 2.0, 0.0], "=", 240.0),
            ConstraintData([3.0, 2.0, 5.0, 1.0], "=", 270.0),
        ]
        problem = LPProblem(
            optimization_type=OptimizationType.MAXIMIZE.value,
            objective_coefficients=[5.0, 4.0, 3.0, 0.0],
            constraints=constraints,
            variables_count=4,
        )
        result = finder.find_initial_bfs(problem)

        assert result.is_feasible()
        A = np.array([[6, 4, 2, 0], [3, 2, 5, 1]], dtype=float)
        b = np.array([240.0, 270.0])
        x = np.array(result.full_solution)
        assert np.allclose(A @ x, b, atol=1e-8)

    def test_degenerate_zero_rhs_equality(self, finder: TwoPhase_BFSFinder):
        """x1 = 0, x2 = 0 — degenerate but feasible."""
        problem = make_eq_standard([1.0, 1.0], [[1, 0], [0, 1]], [0, 0])
        result = finder.find_initial_bfs(problem)

        assert result.is_feasible()
        assert all(abs(v) < 1e-8 for v in result.full_solution)

    def test_large_rhs_values(self, finder: TwoPhase_BFSFinder):
        """Numerics should hold for large RHS."""
        problem = make_eq_standard(
            [1.0, 1.0],
            [[1.0, 0.0], [0.0, 1.0]],
            [1_000_000.0, 2_500_000.0],
        )
        result = finder.find_initial_bfs(problem)

        assert result.is_feasible()
        assert abs(result.full_solution[0] - 1_000_000.0) < 1e-4
        assert abs(result.full_solution[1] - 2_500_000.0) < 1e-4

    def test_fractional_rhs(self, finder: TwoPhase_BFSFinder):
        """Fractional RHS values."""
        problem = make_eq_standard(
            [2.0, 3.0],
            [[1.0, 0.0], [0.0, 1.0]],
            [3.14159, 2.71828],
        )
        result = finder.find_initial_bfs(problem)

        assert result.is_feasible()
        assert abs(result.full_solution[0] - 3.14159) < 1e-6
        assert abs(result.full_solution[1] - 2.71828) < 1e-6


# ══════════════════════════════════════════════════════════════════════
# 3. Phase-1 path — infeasible problems
# ══════════════════════════════════════════════════════════════════════

class TestTwoPhase_Phase1_Infeasible:
    """Problems where Phase-1 reveals infeasibility."""

    def test_contradictory_equality_constraints(self, finder: TwoPhase_BFSFinder):
        """
        x1 + x2 = 5
        x1 + x2 = 8
        Contradictory → infeasible.
        """
        problem = make_eq_standard(
            [1.0, 1.0],
            [[1.0, 1.0], [1.0, 1.0]],
            [5.0, 8.0],
        )
        result = finder.find_initial_bfs(problem)
        assert not result.is_feasible()

    def test_negative_rhs_equality_infeasible(self, finder: TwoPhase_BFSFinder):
        """
        x1 = -3 with x1 >= 0 → infeasible.
        Standard form: -x1 = 3 (after negating to make RHS positive).
        Represented as x1 + a1 = -3 which is impossible for x,a >= 0.
        We simulate this by passing RHS < 0 directly — Phase-1 must detect it.
        """
        # RHS = -3: no non-negative solution exists
        constraints = [ConstraintData([1.0], "=", -3.0)]
        problem = LPProblem(
            optimization_type=OptimizationType.MAXIMIZE.value,
            objective_coefficients=[1.0],
            constraints=constraints,
            variables_count=1,
        )
        result = finder.find_initial_bfs(problem)
        assert not result.is_feasible()

    def test_three_constraints_one_contradiction(self, finder: TwoPhase_BFSFinder):
        """
        x1 + x2 + s1 = 10   (slack row)
        x1           = 3    (feasible alone)
              x2     = 8    (contradicts with row above: 3+8=11 > 10)
        """
        constraints = [
            ConstraintData([1.0, 1.0, 1.0], "=", 10.0),  # slack in col 2
            ConstraintData([1.0, 0.0, 0.0], "=",  3.0),  # needs artificial
            ConstraintData([0.0, 1.0, 0.0], "=",  8.0),  # needs artificial
        ]
        problem = LPProblem(
            optimization_type=OptimizationType.MAXIMIZE.value,
            objective_coefficients=[5.0, 4.0, 0.0],
            constraints=constraints,
            variables_count=3,
        )
        result = finder.find_initial_bfs(problem)
        assert not result.is_feasible()


# ══════════════════════════════════════════════════════════════════════
# 4. BFSolution contract
# ══════════════════════════════════════════════════════════════════════

class TestTwoPhase_BFSSolutionContract:
    """Structural guarantees on the returned BFSolution."""

    def test_basis_length_equals_m(self, finder: TwoPhase_BFSFinder):
        """len(basis_indices) == number of constraints."""
        problem = make_leq_standard([1.0, 2.0], [[1, 0], [0, 1]], [3, 4])
        result = finder.find_initial_bfs(problem)
        assert len(result.basis_indices) == len(problem.constraints)

    def test_basic_values_length_equals_m(self, finder: TwoPhase_BFSFinder):
        """len(basic_values) == number of constraints."""
        problem = make_leq_standard([1.0, 2.0], [[1, 0], [0, 1]], [3, 4])
        result = finder.find_initial_bfs(problem)
        assert len(result.basic_values) == len(problem.constraints)

    def test_full_solution_length_equals_n(self, finder: TwoPhase_BFSFinder):
        """len(full_solution) == variables_count of original problem."""
        problem = make_leq_standard([1.0, 2.0], [[1, 0], [0, 1]], [3, 4])
        n = problem.variables_count
        result = finder.find_initial_bfs(problem)
        assert len(result.full_solution) == n

    def test_basis_indices_unique(self, finder: TwoPhase_BFSFinder):
        """All basis indices must be distinct."""
        problem = make_leq_standard(
            [1.0, 2.0, 3.0],
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [5, 6, 7],
        )
        result = finder.find_initial_bfs(problem)
        assert len(set(result.basis_indices)) == len(result.basis_indices)

    def test_basis_indices_in_range(self, finder: TwoPhase_BFSFinder):
        """All basis indices must be within [0, variables_count)."""
        problem = make_leq_standard([1.0, 2.0], [[3, 1], [1, 2]], [9, 8])
        n = problem.variables_count
        result = finder.find_initial_bfs(problem)
        assert all(0 <= bi < n for bi in result.basis_indices)

    def test_phase1_basis_indices_within_original_n(self, finder: TwoPhase_BFSFinder):
        """
        After Phase-1, returned basis indices must be < original n
        so SimplexTable(standard_form, bfs) does not get IndexError.
        """
        problem = make_eq_standard([1.0, 2.0], [[1, 0], [0, 1]], [4, 5])
        n_original = problem.variables_count
        result = finder.find_initial_bfs(problem)

        assert result.is_feasible()
        assert all(bi < n_original for bi in result.basis_indices)

    def test_max_iterations_respected(self):
        """Finder with max_iterations=1 still returns a BFSolution."""
        finder_1iter = TwoPhase_BFSFinder(max_iterations=1)
        problem = make_eq_standard([1.0, 1.0], [[1, 0], [0, 1]], [3, 4])
        result = finder_1iter.find_initial_bfs(problem)
        # May or may not be fully feasible, but must not raise
        assert isinstance(result, BFSolution)