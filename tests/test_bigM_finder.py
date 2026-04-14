"""
Unit tests for BigM_BFSFinder. Thanks to Claude by Anthropic.

BigM_BFSFinder accepts a raw LPProblem whose constraints still carry
their original operators ('<=', '>=', '=').  It mutates standard_form
in-place by appending one extra column per constraint:

  Column layout: [ original vars (n) | one column per constraint (m) ]
  '<='  →  slack     (+1), objective coef = 0
  '='   →  artificial (+1), objective coef = -M
  '>='  →  artificial (+1), objective coef = -M

Tests are organised into:
  1. Column layout & basis indices
  2. Objective augmentation
  3. Constraint matrix augmentation
  4. BFSolution values
  5. In-place mutation verification
  6. Edge cases
"""

import pytest
import numpy as np

from core.bfs.bigM_finder import BigM_BFSFinder
from utils import LPProblem, ConstraintData, BFSolution, OptimizationType


BIG_M = 1e6


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def make_problem(obj, constraints):
    """
    Shorthand: constraints is a list of (coefficients, operator, rhs).
    variables_count = len(obj)  (original vars only, no slacks yet).
    """
    return LPProblem(
        optimization_type=OptimizationType.MAXIMIZE.value,
        objective_coefficients=list(obj),
        constraints=[ConstraintData(list(c), op, rhs) for c, op, rhs in constraints],
        variables_count=len(obj),
    )


@pytest.fixture
def finder() -> BigM_BFSFinder:
    return BigM_BFSFinder(big_m=BIG_M)


# ══════════════════════════════════════════════════════════════════════
# 1. Column layout & basis indices
# ══════════════════════════════════════════════════════════════════════

class TestBigM_BasisIndices:

    def test_all_leq_basis(self, finder: BigM_BFSFinder):
        """'<=' rows → slack columns n, n+1, … in order."""
        p = make_problem([3, 2], [([1, 1], "<=", 4), ([2, 1], "<=", 5)])
        r = finder.find_initial_bfs(p)
        assert r.basis_indices == [2, 3]

    def test_all_eq_basis(self, finder: BigM_BFSFinder):
        """'=' rows → artificial columns n, n+1, … in order."""
        p = make_problem([1, 2], [([1, 0], "=", 3), ([0, 1], "=", 4)])
        r = finder.find_initial_bfs(p)
        assert r.basis_indices == [2, 3]

    def test_all_geq_basis(self, finder: BigM_BFSFinder):
        """'>=' rows → artificial columns n, n+1, … in order."""
        p = make_problem([2, 3], [([1, 0], ">=", 2), ([0, 1], ">=", 3)])
        r = finder.find_initial_bfs(p)
        assert r.basis_indices == [2, 3]

    def test_mixed_leq_eq_geq_basis(self, finder: BigM_BFSFinder):
        """Mixed: one column per constraint in constraint order."""
        p = make_problem(
            [5, 4, 3],
            [
                ([2, 3, 1], "<=", 10),
                ([4, 1, 2], "=",  15),
                ([3, 4, 2], ">=",  8),
            ],
        )
        r = finder.find_initial_bfs(p)
        assert r.basis_indices == [3, 4, 5]

    def test_basis_order_mixed(self, finder: BigM_BFSFinder):
        """Order: =, <=, >= → cols n, n+1, n+2 regardless of type."""
        p = make_problem(
            [1, 2],
            [([1, 0], "=", 2), ([0, 1], "<=", 4), ([1, 1], ">=", 6)],
        )
        r = finder.find_initial_bfs(p)
        assert r.basis_indices == [2, 3, 4]

    def test_single_eq_constraint(self, finder: BigM_BFSFinder):
        """One '=' row with 1 original var → basis = [1]."""
        p = make_problem([3], [([1], "=", 10)])
        r = finder.find_initial_bfs(p)
        assert r.basis_indices == [1]

    def test_single_leq_constraint(self, finder: BigM_BFSFinder):
        """One '<=' row with 1 original var → basis = [1]."""
        p = make_problem([3], [([1], "<=", 10)])
        r = finder.find_initial_bfs(p)
        assert r.basis_indices == [1]

    def test_basis_indices_unique(self, finder: BigM_BFSFinder):
        """All returned basis indices must be distinct."""
        p = make_problem(
            [1, 2, 3],
            [([1, 0, 0], "<=", 5), ([0, 1, 0], "=", 3), ([0, 0, 1], ">=", 1)],
        )
        r = finder.find_initial_bfs(p)
        assert len(set(r.basis_indices)) == 3

    def test_basis_indices_within_range(self, finder: BigM_BFSFinder):
        """All basis indices must be < total_vars = n + m."""
        p = make_problem([1, 2], [([1, 1], "<=", 4), ([1, 0], "=", 2)])
        n, m = 2, 2
        r = finder.find_initial_bfs(p)
        assert all(0 <= bi < n + m for bi in r.basis_indices)


# ══════════════════════════════════════════════════════════════════════
# 2. Objective augmentation
# ══════════════════════════════════════════════════════════════════════

class TestBigM_ObjectiveAugmentation:

    def test_leq_appends_zero_to_objective(self, finder: BigM_BFSFinder):
        """Slack variable gets objective coefficient 0."""
        p = make_problem([3, 2], [([1, 1], "<=", 5)])
        finder.find_initial_bfs(p)
        # original [3,2] + [0] for slack
        assert p.objective_coefficients == [3, 2, 0.0]

    def test_eq_appends_neg_M_to_objective(self, finder: BigM_BFSFinder):
        """Artificial variable for '=' gets coefficient -M."""
        p = make_problem([1, 2], [([1, 0], "=", 3)])
        finder.find_initial_bfs(p)
        assert p.objective_coefficients == [1, 2, -BIG_M]

    def test_geq_appends_neg_M_to_objective(self, finder: BigM_BFSFinder):
        """Artificial variable for '>=' gets coefficient -M."""
        p = make_problem([4, 5], [([2, 1], ">=", 6)])
        finder.find_initial_bfs(p)
        assert p.objective_coefficients == [4, 5, -BIG_M]

    def test_mixed_objective_augmentation(self, finder: BigM_BFSFinder):
        """<=, =, >= → objective extended with [0, -M, -M]."""
        p = make_problem(
            [5, 4, 3],
            [([2, 3, 1], "<=", 10), ([4, 1, 2], "=", 15), ([3, 4, 2], ">=", 8)],
        )
        finder.find_initial_bfs(p)
        assert p.objective_coefficients == [5, 4, 3, 0.0, -BIG_M, -BIG_M]

    def test_custom_big_m_value(self):
        """Custom big_m value is used in objective coefficients."""
        finder_custom = BigM_BFSFinder(big_m=999.0)
        p = make_problem([1], [([1], "=", 5)])
        finder_custom.find_initial_bfs(p)
        assert p.objective_coefficients[-1] == -999.0

    def test_objective_length_after_augmentation(self, finder: BigM_BFSFinder):
        """len(objective) == n + m after augmentation."""
        p = make_problem(
            [1, 2, 3],
            [([1, 0, 0], "<=", 5), ([0, 1, 0], "=", 3), ([0, 0, 1], ">=", 1)],
        )
        n, m = 3, 3
        finder.find_initial_bfs(p)
        assert len(p.objective_coefficients) == n + m


# ══════════════════════════════════════════════════════════════════════
# 3. Constraint matrix augmentation
# ══════════════════════════════════════════════════════════════════════

class TestBigM_ConstraintAugmentation:

    def test_leq_adds_identity_column(self, finder: BigM_BFSFinder):
        """Each '<=' row gets a +1 in its own position, 0 elsewhere."""
        p = make_problem([1, 1], [([1, 2], "<=", 8), ([3, 1], "<=", 6)])
        finder.find_initial_bfs(p)

        A = np.array([c.coefficients for c in p.constraints])
        # col 2 = [1, 0], col 3 = [0, 1]
        np.testing.assert_array_equal(A[:, 2], [1, 0])
        np.testing.assert_array_equal(A[:, 3], [0, 1])

    def test_eq_adds_artificial_column(self, finder: BigM_BFSFinder):
        """Each '=' row gets a +1 artificial in its own column."""
        p = make_problem([1, 2], [([1, 0], "=", 3), ([0, 1], "=", 4)])
        finder.find_initial_bfs(p)

        A = np.array([c.coefficients for c in p.constraints])
        np.testing.assert_array_equal(A[:, 2], [1, 0])
        np.testing.assert_array_equal(A[:, 3], [0, 1])

    def test_geq_adds_artificial_column(self, finder: BigM_BFSFinder):
        """Each '>=' row gets a +1 artificial in its own column."""
        p = make_problem([2, 3], [([1, 0], ">=", 2), ([0, 1], ">=", 3)])
        finder.find_initial_bfs(p)

        A = np.array([c.coefficients for c in p.constraints])
        np.testing.assert_array_equal(A[:, 2], [1, 0])
        np.testing.assert_array_equal(A[:, 3], [0, 1])

    def test_original_coefficients_preserved(self, finder: BigM_BFSFinder):
        """Original columns of the constraint matrix must not change."""
        p = make_problem(
            [5, 4],
            [([2.0, 3.0], "<=", 12), ([1.0, 2.0], "=", 6)],
        )
        finder.find_initial_bfs(p)

        A = np.array([c.coefficients for c in p.constraints])
        np.testing.assert_array_equal(A[:, :2], [[2, 3], [1, 2]])

    def test_constraints_normalised_to_equality(self, finder: BigM_BFSFinder):
        """All constraint operators are '=' after augmentation."""
        p = make_problem(
            [1, 2],
            [([1, 0], "<=", 4), ([0, 1], ">=", 2), ([1, 1], "=", 5)],
        )
        finder.find_initial_bfs(p)
        assert all(c.operator == "=" for c in p.constraints)

    def test_constraint_coefficient_length(self, finder: BigM_BFSFinder):
        """Each constraint row has exactly n + m coefficients."""
        p = make_problem(
            [1, 2, 3],
            [([1, 0, 0], "<=", 5), ([0, 1, 0], "=", 3), ([0, 0, 1], ">=", 1)],
        )
        n, m = 3, 3
        finder.find_initial_bfs(p)
        for c in p.constraints:
            assert len(c.coefficients) == n + m

    def test_variables_count_updated(self, finder: BigM_BFSFinder):
        """standard_form.variables_count is updated to n + m."""
        p = make_problem([1, 2], [([1, 1], "<=", 4), ([2, 1], "=", 5)])
        finder.find_initial_bfs(p)
        assert p.variables_count == 4   # 2 orig + 2 extra


# ══════════════════════════════════════════════════════════════════════
# 4. BFSolution values
# ══════════════════════════════════════════════════════════════════════

class TestBigM_BFSValues:

    def test_basic_values_match_rhs(self, finder: BigM_BFSFinder):
        """basic_values must equal the RHS (free_val) of each constraint."""
        p = make_problem([3, 2], [([1, 1], "<=", 4), ([2, 1], "<=", 5)])
        r = finder.find_initial_bfs(p)
        assert r.basic_values == [4, 5]

    def test_full_solution_basis_entries(self, finder: BigM_BFSFinder):
        """full_solution[bi] == basic_values[i] for each basis variable."""
        p = make_problem(
            [5, 4, 3],
            [([2, 3, 1], "<=", 10), ([4, 1, 2], "=", 15), ([3, 4, 2], ">=", 8)],
        )
        r = finder.find_initial_bfs(p)
        for i, bi in enumerate(r.basis_indices):
            assert abs(r.full_solution[bi] - r.basic_values[i]) < 1e-10

    def test_full_solution_non_basic_are_zero(self, finder: BigM_BFSFinder):
        """Non-basic variables in full_solution must be 0."""
        p = make_problem([1, 2], [([1, 0], "=", 3), ([0, 1], "=", 4)])
        r = finder.find_initial_bfs(p)
        non_basic = [j for j in range(len(r.full_solution))
                     if j not in r.basis_indices]
        for j in non_basic:
            assert r.full_solution[j] == 0.0

    def test_is_feasible_all_positive_rhs(self, finder: BigM_BFSFinder):
        """Positive RHS → is_feasible() == True."""
        p = make_problem([1, 1], [([1, 1], "=", 10)])
        r = finder.find_initial_bfs(p)
        assert r.is_feasible()

    def test_is_feasible_zero_rhs(self, finder: BigM_BFSFinder):
        """Zero RHS is degenerate but feasible (0 >= 0)."""
        p = make_problem(
            [1, 1],
            [([1, 0], "=", 0), ([0, 1], "=", 0)],
        )
        r = finder.find_initial_bfs(p)
        assert r.is_feasible()

    def test_full_solution_length(self, finder: BigM_BFSFinder):
        """full_solution length == total_vars == n + m."""
        p = make_problem([3, 2], [([1, 1], "<=", 4), ([2, 1], "<=", 5)])
        r = finder.find_initial_bfs(p)
        assert len(r.full_solution) == 4   # 2 orig + 2 slack

    def test_fractional_rhs_preserved(self, finder: BigM_BFSFinder):
        """Fractional RHS values are returned exactly."""
        p = make_problem(
            [1, 2],
            [([1, 1], "=", 1000.5), ([2, 1], ">=", 2500.75)],
        )
        r = finder.find_initial_bfs(p)
        assert np.allclose(r.basic_values, [1000.5, 2500.75])
        assert np.allclose(r.full_solution[2], 1000.5)
        assert np.allclose(r.full_solution[3], 2500.75)


# ══════════════════════════════════════════════════════════════════════
# 5. In-place mutation
# ══════════════════════════════════════════════════════════════════════

class TestBigM_Mutation:

    def test_variables_count_equals_n_plus_m(self, finder: BigM_BFSFinder):
        """variables_count is updated to n + m."""
        p = make_problem(
            [1, 2, 3],
            [([1, 0, 0], "<=", 5), ([0, 1, 0], "=", 3), ([0, 0, 1], ">=", 1)],
        )
        finder.find_initial_bfs(p)
        assert p.variables_count == 6

    def test_mutation_allows_simplex_table_construction(self, finder: BigM_BFSFinder):
        """
        After mutation, SimplexTable(problem, bfs) must not raise.
        We cannot import SimplexTable here, so we verify the shapes are
        consistent: A has shape (m, n+m), c has length n+m, basis has m entries.
        """
        p = make_problem(
            [5, 4],
            [([6, 4], "<=", 24), ([1, 2], "=", 6)],
        )
        r = finder.find_initial_bfs(p)
        m = len(p.constraints)
        n_total = p.variables_count

        A = np.array([c.coefficients for c in p.constraints])
        assert A.shape == (m, n_total)
        assert len(p.objective_coefficients) == n_total
        assert len(r.basis_indices) == m

    def test_repeated_calls_are_idempotent_on_copies(self, finder: BigM_BFSFinder):
        """
        Calling find_initial_bfs twice on the SAME object accumulates columns
        (expected: mutation is intentional). But two independent copies should
        produce identical results.
        """
        from copy import deepcopy
        p1 = make_problem([1, 2], [([1, 0], "=", 3), ([0, 1], "=", 4)])
        p2 = deepcopy(p1)

        r1 = BigM_BFSFinder(BIG_M).find_initial_bfs(p1)
        r2 = BigM_BFSFinder(BIG_M).find_initial_bfs(p2)

        assert r1.basis_indices == r2.basis_indices
        assert r1.basic_values  == r2.basic_values
        assert r1.full_solution == r2.full_solution


# ══════════════════════════════════════════════════════════════════════
# 6. Edge cases
# ══════════════════════════════════════════════════════════════════════

class TestBigM_EdgeCases:

    def test_single_variable_single_leq(self, finder: BigM_BFSFinder):
        p = make_problem([5], [([1], "<=", 7)])
        r = finder.find_initial_bfs(p)
        assert r.basis_indices == [1]
        assert r.basic_values  == [7]
        assert r.full_solution == [0.0, 7.0]

    def test_single_variable_single_eq(self, finder: BigM_BFSFinder):
        p = make_problem([3], [([1], "=", 10)])
        r = finder.find_initial_bfs(p)
        assert r.basis_indices == [1]
        assert r.basic_values  == [10]
        assert r.is_feasible()

    def test_single_variable_single_geq(self, finder: BigM_BFSFinder):
        p = make_problem([2], [([1], ">=", 4)])
        r = finder.find_initial_bfs(p)
        assert r.basis_indices == [1]
        assert r.basic_values  == [4]
        assert r.is_feasible()

    def test_many_constraints_all_leq(self, finder: BigM_BFSFinder):
        """5 constraints, all '<=': basis = [5,6,7,8,9]."""
        rows = [[float(i == j) for j in range(5)] for i in range(5)]
        p = make_problem([1]*5, [(r, "<=", float(i+1)) for i, r in enumerate(rows)])
        r = finder.find_initial_bfs(p)
        assert r.basis_indices == [5, 6, 7, 8, 9]

    def test_operator_with_whitespace(self, finder: BigM_BFSFinder):
        """Operators with surrounding whitespace are handled correctly."""
        p = LPProblem(
            optimization_type=OptimizationType.MAXIMIZE.value,
            objective_coefficients=[1.0, 2.0],
            constraints=[
                ConstraintData([1.0, 0.0], " <= ", 5.0),
                ConstraintData([0.0, 1.0], " = ",  3.0),
            ],
            variables_count=2,
        )
        r = finder.find_initial_bfs(p)
        assert r.basis_indices == [2, 3]
        # row 0 → slack (<=), objective coef = 0
        assert p.objective_coefficients[2] == 0.0
        # row 1 → artificial (=), objective coef = -M
        assert p.objective_coefficients[3] == -BIG_M

    def test_zero_rhs_all_eq(self, finder: BigM_BFSFinder):
        """All RHS = 0: degenerate but structurally valid BFS."""
        p = make_problem(
            [1, 1],
            [([1, 0], "=", 0), ([0, 1], "=", 0)],
        )
        r = finder.find_initial_bfs(p)
        assert r.basis_indices == [2, 3]
        assert r.basic_values  == [0, 0]
        assert r.is_feasible()