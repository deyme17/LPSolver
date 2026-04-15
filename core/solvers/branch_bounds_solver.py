from dataclasses import dataclass
from typing import List, Optional
import math
import heapq

from utils import ConstraintOperator as CO
from utils import (
    LPProblem, LPResult, IBFSFinder, ISolver,
    ConstraintData, SolutionStatus, OptimizationType
    )
from .simplex_solver import SimplexSolver


@dataclass
class BBNode:
    """Represents a node in the Branch and Bound search tree."""
    problem: LPProblem
    depth: int = 0
    relaxation_value: float = 0.0
    branch_var: Optional[int] = None
    branch_dir: Optional[str] = None
    is_max: bool = True

    def __lt__(self, other):
        if self.is_max:
            return self.relaxation_value > other.relaxation_value
        else:
            return self.relaxation_value < other.relaxation_value



class BranchAndBoundSolver(ISolver):
    """
    Implements the Branch and Bound algorithm for solving Integer Programming problems.

    This solver manages a tree of LP-relaxations, using a SimplexSolver to find 
    optimal continuous solutions at each node and branching on fractional variables.
    """
    SUPPORT_INTEGER_CONSTRAINTS = True

    def __init__(self, bfs_finder: IBFSFinder, max_nodes: int = 1000, tolerance: float = 1e-6):
        """
        Args:
            bfs_finder: Component used to find the initial Basic Feasible Solution for relaxations.
            max_nodes: Maximum number of nodes to explore before terminating with an error.
            tolerance: Numerical tolerance for integrality checks and value comparisons.
        """
        self.simplex_solver = SimplexSolver(bfs_finder)
        self.max_nodes = max_nodes
        self.tol = tolerance
        self.best_result: Optional[LPResult] = None
        self.nodes_explored = 0

    def solve(self, problem: LPProblem) -> LPResult:
        """
        Solves the Linear Programming problem with respect to integer constraints.
        Args:
            problem: The LP problem definition containing objective, constraints, and integer indices.
        Returns:
            LPResult: The best integer-feasible solution found, or an error status if none exist.
        """
        if not problem.integer_indices:
            return self.simplex_solver.solve(problem)

        is_max = (problem.optimization_type == OptimizationType.MAXIMIZE.value)
        root = BBNode(problem=problem, depth=0, is_max=is_max)
        node_queue = [root]
        heapq.heapify(node_queue)

        while node_queue:
            if self.nodes_explored >= self.max_nodes:
                return LPResult(status=SolutionStatus.ERROR.value,
                                error_message="B&B node limit reached")

            node = heapq.heappop(node_queue)
            self.nodes_explored += 1

            # relaxation
            relaxation = self.simplex_solver.solve(node.problem)

            # prune infeasible/error nodes
            if relaxation.status != SolutionStatus.OPTIMAL.value:
                continue
            # bounding
            if self._should_prune(relaxation, is_max):
                continue

            # int check
            if self._is_integer_feasible(relaxation, problem.integer_indices):
                if self.best_result is None or self._is_better(relaxation, self.best_result, is_max):
                    self.best_result = relaxation
                continue

            # branching
            frac_idx = self._get_fractional_var(relaxation, problem.integer_indices)
            val = relaxation.solution[frac_idx]
            floor_val, ceil_val = math.floor(val), math.ceil(val)

            lb, ub = self._get_var_bounds(node.problem, frac_idx)
            # left branch: xi <= floor(val) (skip if floor_val < lb)
            if floor_val >= lb:
                left_prob = self._add_constraint(node.problem, frac_idx, CO.LEQ.value, floor_val)
                heapq.heappush(node_queue, BBNode(
                    problem=left_prob, depth=node.depth + 1,
                    relaxation_value=relaxation.optimal_value,
                    branch_var=frac_idx, branch_dir=f"{CO.LEQ.value}{floor_val}", is_max=is_max)
                )
            # right branch xi >= ceil(var) (skip if exeedes upper bound)
            if ceil_val <= ub:
                right_prob = self._add_constraint(node.problem, frac_idx, CO.GEQ.value, ceil_val)
                heapq.heappush(node_queue, BBNode(
                    problem=right_prob, depth=node.depth + 1,
                    relaxation_value=relaxation.optimal_value,
                    branch_var=frac_idx, branch_dir=f"{CO.GEQ.value}{ceil_val}", is_max=is_max)
                )

        if self.best_result:
            self.best_result.status = SolutionStatus.OPTIMAL.value
            return self.best_result

        return LPResult(status=SolutionStatus.INFEASIBLE.value,
                        error_message="No feasible integer solution found")

    def _should_prune(self, result: LPResult, is_max: bool) -> bool:
        """Determines if the current branch should be pruned based on the bound."""
        if self.best_result is None:
            return False
        if is_max:
            return result.optimal_value < self.best_result.optimal_value - self.tol
        else:
            return result.optimal_value > self.best_result.optimal_value + self.tol

    def _is_better(self, new: LPResult, old: LPResult, is_max: bool) -> bool:
        """Compares two solutions to determine which is more optimal."""
        if is_max:
            return new.optimal_value > old.optimal_value + self.tol
        return new.optimal_value < old.optimal_value - self.tol

    def _is_integer_feasible(self, result: LPResult, int_indices: Optional[List[int]]) -> bool:
        """Checks if all variables at specified indices are integers within the tolerance."""
        if not int_indices:
            return True
        for idx in int_indices:
            if abs(result.solution[idx] - round(result.solution[idx])) > self.tol:
                return False
        return True

    def _get_fractional_var(self, result: LPResult, int_indices: Optional[List[int]]) -> int:
        """Returns the index of the biggest fractional variable that should be an integer."""
        best_var = None
        max_frac = -1.0
        for idx in int_indices:
            val = result.solution[idx]
            frac = abs(val - round(val))
            if frac > self.tol and frac > max_frac:
                max_frac = frac
                best_var = idx
        if best_var is not None:
            return best_var
        raise ValueError("No fractional variable found")
    
    def _get_var_bounds(self, prob: LPProblem, var_idx: int) -> tuple[float, float]:
        """Return the tightest lb/ub for var_idx implied by single-variable constraints in prob."""
        lb, ub = 0.0, float("inf")
        for c in prob.constraints:
            non_zero = [i for i, x in enumerate(c.coefficients) if x != 0]
            if len(non_zero) == 1 and non_zero[0] == var_idx:
                coef = c.coefficients[var_idx]
                rhs = c.free_val / coef  # normalize
                if c.operator == CO.LEQ.value:
                    ub = min(ub, rhs)
                elif c.operator == CO.GEQ.value:
                    lb = max(lb, rhs)
        return lb, ub

    def _add_constraint(self, prob: LPProblem, var_idx: int, op: str, bound: float) -> LPProblem:
        """Generates a new LPProblem instance by adding a branching constraint."""
        coeffs = [0.0] * prob.variables_count
        coeffs[var_idx] = 1.0
        new_const = ConstraintData(coefficients=coeffs, operator=op, free_val=bound)
        
        return LPProblem(
            optimization_type=prob.optimization_type,
            objective_coefficients=prob.objective_coefficients.copy(),
            constraints=prob.constraints + [new_const],
            integer_indices=prob.integer_indices.copy() if prob.integer_indices else None,
            variables_count=prob.variables_count
        )
    
    def clear_state(self) -> None:
        """Clear solver state."""
        self.best_result = None
        self.nodes_explored = 0