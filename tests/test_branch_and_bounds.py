"""
Unit tests for BranchAndBoundSolver (branch_bounds_solver.py).
Thanks to Claude by Anthropic.
Structure:
  - Fixtures / helpers
  - TestBranchAndBoundSolverBasic   – happy-path integer solutions
  - TestBranchAndBoundSolverEdge    – edge / boundary conditions
  - TestBranchAndBoundSolverPrivate – internal helper methods
  - TestBranchAndBoundSolverIntegration – full solve() round-trips

Run with:
    pytest test_branch_bounds_solver.py -v
"""

import math
import pytest
from unittest.mock import MagicMock, patch

from utils import (
    LPProblem, LPResult, ConstraintData,
    SolutionStatus, OptimizationType,
)
from core.solvers.branch_bounds_solver import BranchAndBoundSolver, BBNode
from core.bfs import TwoPhase_BFSFinder, BigM_BFSFinder

bfs_finder = TwoPhase_BFSFinder
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_problem(
    obj_coefs,
    constraints,
    opt_type=OptimizationType.MAXIMIZE.value,
    integer_indices=None,
):
    """Convenience factory for LPProblem."""
    return LPProblem(
        optimization_type=opt_type,
        objective_coefficients=obj_coefs,
        constraints=[
            ConstraintData(coefficients=c[0], operator=c[1], free_val=c[2])
            for c in constraints
        ],
        variables_count=len(obj_coefs),
        integer_indices=integer_indices,
    )


def make_lp_result(status, optimal_value=None, solution=None):
    r = LPResult(status=status)
    r.optimal_value = optimal_value
    r.solution = solution
    return r


def default_solver(max_nodes=5000):
    return BranchAndBoundSolver(bfs_finder(), max_nodes=max_nodes)


# ---------------------------------------------------------------------------
# TestBranchAndBoundSolverBasic
# ---------------------------------------------------------------------------

class TestBranchAndBoundSolverBasic:
    """Happy-path tests: problems with known integer optima."""

    def test_simple_maximize_integer(self):
        """
        max  x1 + x2
        s.t. x1 + x2 <= 3.5
             x1, x2 >= 0, integer
        Optimum: x1=1, x2=2 (or x1=2, x2=1) → value 3
        """
        prob = make_problem(
            obj_coefs=[1.0, 1.0],
            constraints=[
                ([1.0, 1.0], "<=", 3.5),
                ([1.0, 0.0], "<=", 3.5),
                ([0.0, 1.0], "<=", 3.5),
            ],
            integer_indices=[0, 1],
        )
        solver = default_solver()
        result = solver.solve(prob)

        assert result.status == SolutionStatus.OPTIMAL.value
        assert abs(result.optimal_value - 3.0) < 1e-5
        assert all(
            abs(result.solution[i] - round(result.solution[i])) < 1e-5
            for i in [0, 1]
        )

    def test_simple_minimize_integer(self):
        """
        min  x1 + x2
        s.t. x1 + x2 >= 2.5
             x1, x2 >= 0, integer
        Optimum: x1+x2 = 3 → value 3
        """
        prob = make_problem(
            obj_coefs=[1.0, 1.0],
            constraints=[
                ([1.0, 1.0], ">=", 2.5),
                ([1.0, 0.0], "<=", 10.0),
                ([0.0, 1.0], "<=", 10.0),
            ],
            opt_type=OptimizationType.MINIMIZE.value,
            integer_indices=[0, 1],
        )
        solver = default_solver()
        result = solver.solve(prob)

        assert result.status == SolutionStatus.OPTIMAL.value
        assert abs(result.optimal_value - 3.0) < 1e-5

    def test_already_integer_relaxation(self):
        """
        When LP relaxation is already integer-valued, B&B should return it
        without any branching.
        max x1
        s.t. x1 <= 4   (relaxation gives x1=4 — already integer)
        """
        prob = make_problem(
            obj_coefs=[1.0, 0.0],
            constraints=[
                ([1.0, 0.0], "<=", 4.0),
                ([0.0, 1.0], "<=", 4.0),
            ],
            integer_indices=[0],
        )
        solver = default_solver()
        result = solver.solve(prob)

        assert result.status == SolutionStatus.OPTIMAL.value
        assert abs(result.optimal_value - 4.0) < 1e-5
        assert abs(result.solution[0] - 4.0) < 1e-5

    def test_no_integer_indices_delegates_to_simplex(self):
        """
        If integer_indices is None, B&B must delegate directly to
        SimplexSolver without any branching.
        """
        prob = make_problem(
            obj_coefs=[3.0, 2.0],
            constraints=[
                ([1.0, 1.0], "<=", 4.0),
                ([2.0, 1.0], "<=", 6.0),
            ],
            integer_indices=None,
        )
        solver = default_solver()

        with patch.object(solver.simplex_solver, "solve", wraps=solver.simplex_solver.solve) as mock_solve:
            result = solver.solve(prob)
            mock_solve.assert_called_once_with(prob)

        assert result.status == SolutionStatus.OPTIMAL.value

    def test_single_variable_integer(self):
        """
        max  3x
        s.t. x <= 2.7
             x integer
        Optimum: x = 2 → value 6
        """
        prob = make_problem(
            obj_coefs=[3.0],
            constraints=[([1.0], "<=", 2.7)],
            integer_indices=[0],
        )
        solver = default_solver()
        result = solver.solve(prob)

        assert result.status == SolutionStatus.OPTIMAL.value
        assert abs(result.optimal_value - 6.0) < 1e-5
        assert abs(result.solution[0] - 2.0) < 1e-5


# ---------------------------------------------------------------------------
# TestBranchAndBoundSolverEdge
# ---------------------------------------------------------------------------

class TestBranchAndBoundSolverEdge:
    """Edge and boundary conditions."""

    def test_infeasible_problem(self):
        """
        Contradictory constraints — B&B should return INFEASIBLE.
        x >= 5 and x <= 3 (both with x integer)
        """
        prob = make_problem(
            obj_coefs=[1.0],
            constraints=[
                ([1.0], ">=", 5.0),
                ([1.0], "<=", 3.0),
            ],
            integer_indices=[0],
        )
        solver = default_solver()
        result = solver.solve(prob)

        assert result.status in (
            SolutionStatus.INFEASIBLE.value,
            SolutionStatus.ERROR.value,
        )

    def test_node_limit_returns_error(self):
        """
        With max_nodes=1 the solver should hit the limit and return ERROR.
        """
        prob = make_problem(
            obj_coefs=[1.0, 1.0],
            constraints=[
                ([1.0, 1.0], "<=", 5.5),
                ([1.0, 0.0], "<=", 5.5),
                ([0.0, 1.0], "<=", 5.5),
            ],
            integer_indices=[0, 1],
        )
        solver = BranchAndBoundSolver(bfs_finder(), max_nodes=1)
        result = solver.solve(prob)

        assert result.status == SolutionStatus.ERROR.value

    def test_all_constraints_already_tight_integer(self):
        """
        max  2x1 + x2
        s.t. x1 = 2 (implemented as <= and >=)
             x2 = 1
        Both integer → value 5
        """
        prob = make_problem(
            obj_coefs=[2.0, 1.0],
            constraints=[
                ([1.0, 0.0], "<=", 2.0),
                ([1.0, 0.0], ">=", 2.0),
                ([0.0, 1.0], "<=", 1.0),
                ([0.0, 1.0], ">=", 1.0),
            ],
            integer_indices=[0, 1],
        )
        solver = default_solver()
        result = solver.solve(prob)

        assert result.status == SolutionStatus.OPTIMAL.value
        assert abs(result.optimal_value - 5.0) < 1e-5

    def test_partial_integer_constraints(self):
        """
        Only x1 needs to be integer.
        max  x1 + 0.5*x2
        s.t. x1 + x2 <= 3.5
        x1 integer, x2 continuous
        Optimal: x1=0, x2=3.5 → 1.75   or   x1=1, x2=2.5 → 2.25
                 or  x1=3, x2=0.5 → 3.25
        """
        prob = make_problem(
            obj_coefs=[1.0, 0.5],
            constraints=[
                ([1.0, 1.0], "<=", 3.5),
                ([1.0, 0.0], "<=", 3.5),
                ([0.0, 1.0], "<=", 3.5),
            ],
            integer_indices=[0],
        )
        solver = default_solver()
        result = solver.solve(prob)

        assert result.status == SolutionStatus.OPTIMAL.value
        # x1 must be integer
        assert abs(result.solution[0] - round(result.solution[0])) < 1e-5

    def test_large_bound_doesnt_crash(self):
        """Solver should handle large right-hand-side values gracefully."""
        prob = make_problem(
            obj_coefs=[1.0],
            constraints=[([1.0], "<=", 1_000_000.0)],
            integer_indices=[0],
        )
        solver = default_solver()
        result = solver.solve(prob)

        assert result.status == SolutionStatus.OPTIMAL.value
        assert abs(result.solution[0] - 1_000_000.0) < 1.0


# ---------------------------------------------------------------------------
# TestBranchAndBoundSolverPrivate
# ---------------------------------------------------------------------------

class TestBranchAndBoundSolverPrivate:
    """Tests for internal helper methods."""

    def setup_method(self):
        self.solver = default_solver()

    # --- _should_prune ---

    def test_should_prune_no_best_result(self):
        result = make_lp_result(SolutionStatus.OPTIMAL.value, optimal_value=10.0)
        assert self.solver._should_prune(result, is_max=True) is False

    def test_should_prune_max_worse(self):
        self.solver.best_result = make_lp_result(
            SolutionStatus.OPTIMAL.value, optimal_value=10.0
        )
        worse = make_lp_result(SolutionStatus.OPTIMAL.value, optimal_value=5.0)
        assert self.solver._should_prune(worse, is_max=True) is True

    def test_should_prune_max_better(self):
        self.solver.best_result = make_lp_result(
            SolutionStatus.OPTIMAL.value, optimal_value=5.0
        )
        better = make_lp_result(SolutionStatus.OPTIMAL.value, optimal_value=10.0)
        assert self.solver._should_prune(better, is_max=True) is False

    def test_should_prune_min_worse(self):
        self.solver.best_result = make_lp_result(
            SolutionStatus.OPTIMAL.value, optimal_value=3.0
        )
        worse = make_lp_result(SolutionStatus.OPTIMAL.value, optimal_value=8.0)
        assert self.solver._should_prune(worse, is_max=False) is True

    def test_should_prune_min_better(self):
        self.solver.best_result = make_lp_result(
            SolutionStatus.OPTIMAL.value, optimal_value=8.0
        )
        better = make_lp_result(SolutionStatus.OPTIMAL.value, optimal_value=3.0)
        assert self.solver._should_prune(better, is_max=False) is False

    # --- _is_better ---

    def test_is_better_max_new_higher(self):
        new = make_lp_result(SolutionStatus.OPTIMAL.value, optimal_value=10.0)
        old = make_lp_result(SolutionStatus.OPTIMAL.value, optimal_value=5.0)
        assert self.solver._is_better(new, old, is_max=True) is True

    def test_is_better_max_new_lower(self):
        new = make_lp_result(SolutionStatus.OPTIMAL.value, optimal_value=3.0)
        old = make_lp_result(SolutionStatus.OPTIMAL.value, optimal_value=8.0)
        assert self.solver._is_better(new, old, is_max=True) is False

    def test_is_better_min_new_lower(self):
        new = make_lp_result(SolutionStatus.OPTIMAL.value, optimal_value=2.0)
        old = make_lp_result(SolutionStatus.OPTIMAL.value, optimal_value=7.0)
        assert self.solver._is_better(new, old, is_max=False) is True

    def test_is_better_min_new_higher(self):
        new = make_lp_result(SolutionStatus.OPTIMAL.value, optimal_value=9.0)
        old = make_lp_result(SolutionStatus.OPTIMAL.value, optimal_value=4.0)
        assert self.solver._is_better(new, old, is_max=False) is False

    def test_is_better_equal_within_tolerance(self):
        """Equal values should NOT count as 'better'."""
        val = 5.0
        new = make_lp_result(SolutionStatus.OPTIMAL.value, optimal_value=val)
        old = make_lp_result(SolutionStatus.OPTIMAL.value, optimal_value=val)
        assert self.solver._is_better(new, old, is_max=True) is False
        assert self.solver._is_better(new, old, is_max=False) is False

    # --- _is_integer_feasible ---

    def test_integer_feasible_exact(self):
        result = make_lp_result(SolutionStatus.OPTIMAL.value, solution=[2.0, 3.0, 1.0])
        assert self.solver._is_integer_feasible(result, [0, 1, 2]) is True

    def test_integer_feasible_fractional(self):
        result = make_lp_result(SolutionStatus.OPTIMAL.value, solution=[2.5, 3.0])
        assert self.solver._is_integer_feasible(result, [0, 1]) is False

    def test_integer_feasible_no_indices(self):
        result = make_lp_result(SolutionStatus.OPTIMAL.value, solution=[1.7, 2.3])
        assert self.solver._is_integer_feasible(result, None) is True
        assert self.solver._is_integer_feasible(result, []) is True

    def test_integer_feasible_within_tolerance(self):
        """Values within tolerance should be treated as integer."""
        eps = self.solver.tol / 2
        result = make_lp_result(
            SolutionStatus.OPTIMAL.value, solution=[3.0 + eps, 1.0 - eps]
        )
        assert self.solver._is_integer_feasible(result, [0, 1]) is True

    # --- _get_fractional_var ---

    def test_get_fractional_var_first_found(self):
        result = make_lp_result(SolutionStatus.OPTIMAL.value, solution=[1.0, 2.7, 3.0])
        idx = self.solver._get_fractional_var(result, [0, 1, 2])
        assert idx == 1

    def test_get_fractional_var_raises_when_none(self):
        result = make_lp_result(SolutionStatus.OPTIMAL.value, solution=[2.0, 3.0])
        with pytest.raises(ValueError, match="No fractional variable"):
            self.solver._get_fractional_var(result, [0, 1])

    # --- _add_constraint ---

    def test_add_constraint_leq(self):
        prob = make_problem(
            obj_coefs=[1.0, 1.0],
            constraints=[([1.0, 1.0], "<=", 4.0)],
        )
        new_prob = self.solver._add_constraint(prob, var_idx=0, op="<=", bound=2.0)
        assert len(new_prob.constraints) == 2
        last = new_prob.constraints[-1]
        assert last.coefficients == [1.0, 0.0]
        assert last.operator == "<="
        assert last.free_val == 2.0

    def test_add_constraint_geq(self):
        prob = make_problem(
            obj_coefs=[1.0, 1.0],
            constraints=[([1.0, 1.0], "<=", 4.0)],
        )
        new_prob = self.solver._add_constraint(prob, var_idx=1, op=">=", bound=1.0)
        last = new_prob.constraints[-1]
        assert last.coefficients == [0.0, 1.0]
        assert last.operator == ">="
        assert last.free_val == 1.0

    def test_add_constraint_does_not_mutate_original(self):
        """_add_constraint must return a new problem, not alter the original."""
        prob = make_problem(
            obj_coefs=[1.0, 1.0],
            constraints=[([1.0, 1.0], "<=", 4.0)],
        )
        original_len = len(prob.constraints)
        self.solver._add_constraint(prob, 0, "<=", 1.0)
        assert len(prob.constraints) == original_len

    def test_add_constraint_preserves_integer_indices(self):
        prob = make_problem(
            obj_coefs=[1.0, 1.0],
            constraints=[([1.0, 1.0], "<=", 4.0)],
            integer_indices=[0, 1],
        )
        new_prob = self.solver._add_constraint(prob, 0, "<=", 1.0)
        assert new_prob.integer_indices == [0, 1]

    def test_add_constraint_integer_indices_none(self):
        prob = make_problem(
            obj_coefs=[1.0],
            constraints=[([1.0], "<=", 5.0)],
            integer_indices=None,
        )
        new_prob = self.solver._add_constraint(prob, 0, "<=", 3.0)
        assert new_prob.integer_indices is None


# ---------------------------------------------------------------------------
# TestBranchAndBoundSolverIntegration
# ---------------------------------------------------------------------------

class TestBranchAndBoundSolverIntegration:
    """End-to-end integration tests using real SimplexSolver under the hood."""

    def test_classic_ip_example(self):
        """
        Classic integer programming textbook example:
        max  x1 + x2
        s.t. -x1 + x2 <= 1
              3x1 + 2x2 <= 12
              2x1 + 3x2 <= 12
              x1, x2 >= 0, integer
        Known optimum: x1=2, x2=2 → value 5  (or x1=2, x2=2 → 4)
        """
        prob = make_problem(
            obj_coefs=[1.0, 1.0],
            constraints=[
                ([-1.0,  1.0], "<=", 1.0),
                ([ 3.0,  2.0], "<=", 12.0),
                ([ 2.0,  3.0], "<=", 12.0),
            ],
            integer_indices=[0, 1],
        )
        solver = default_solver()
        result = solver.solve(prob)

        assert result.status == SolutionStatus.OPTIMAL.value
        assert abs(result.optimal_value - 4.0) < 1e-5
        assert all(
            abs(result.solution[i] - round(result.solution[i])) < 1e-5
            for i in [0, 1]
        )

    def test_nodes_explored_counter_increments(self):
        """nodes_explored must grow during the search."""
        prob = make_problem(
            obj_coefs=[1.0, 1.0],
            constraints=[
                ([1.0, 1.0], "<=", 3.5),
                ([1.0, 0.0], "<=", 3.5),
                ([0.0, 1.0], "<=", 3.5),
            ],
            integer_indices=[0, 1],
        )
        solver = default_solver()
        assert solver.nodes_explored == 0
        solver.solve(prob)
        assert solver.nodes_explored > 0

    def test_best_result_updated_correctly(self):
        """After solve(), best_result should equal the returned solution."""
        prob = make_problem(
            obj_coefs=[2.0, 1.0],
            constraints=[
                ([1.0, 1.0], "<=", 4.5),
                ([1.0, 0.0], "<=", 4.5),
                ([0.0, 1.0], "<=", 4.5),
            ],
            integer_indices=[0, 1],
        )
        solver = default_solver()
        result = solver.solve(prob)

        assert solver.best_result is not None
        assert abs(solver.best_result.optimal_value - result.optimal_value) < 1e-9

    def test_bb_node_dataclass_defaults(self):
        """BBNode should initialise with sane defaults."""
        prob = make_problem(obj_coefs=[1.0], constraints=[([1.0], "<=", 5.0)])
        node = BBNode(problem=prob)
        assert node.depth == 0
        assert node.branch_var is None
        assert node.branch_dir is None

    def test_infeasible_after_branching(self):
        """
        When floor/ceil branching makes both children infeasible,
        the solver should return INFEASIBLE.
        x1 must be integer and lie strictly between 1 and 2 (impossible).
        """
        prob = make_problem(
            obj_coefs=[1.0],
            constraints=[
                ([1.0], ">=", 1.1),
                ([1.0], "<=", 1.9),
            ],
            integer_indices=[0],
        )
        solver = default_solver()
        result = solver.solve(prob)

        assert result.status == SolutionStatus.INFEASIBLE.value