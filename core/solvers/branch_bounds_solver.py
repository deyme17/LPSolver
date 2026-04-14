from dataclasses import dataclass
from typing import List, Optional
import math

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
    branch_var: Optional[int] = None
    branch_dir: Optional[str] = None



class BranchAndBoundSolver(ISolver):
    """
    Implements the Branch and Bound algorithm for solving Integer Programming problems.

    This solver manages a tree of LP-relaxations, using a SimplexSolver to find 
    optimal continuous solutions at each node and branching on fractional variables.
    """
    SUPPORT_INTEGER_CONSTRAINTS = True

    def __init__(self, bfs_finder: IBFSFinder, max_nodes: int = 5000, tolerance: float = 1e-6):
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
        if not problem.integer_indicies:
            return self.simplex_solver.solve(problem)

        is_max = (problem.optimization_type == OptimizationType.MAXIMIZE.value)
        root = BBNode(problem=problem)
        active_nodes = [root]

        while active_nodes:
            if self.nodes_explored >= self.max_nodes:
                return LPResult(status=SolutionStatus.ERROR.value,
                                error_message="B&B node limit reached")

            node = active_nodes.pop()
            self.nodes_explored += 1

            # relaxation
            relaxation = self.simplex_solver.solve(node.problem)

            # prune
            if relaxation.status != SolutionStatus.OPTIMAL.value:
                continue 
            # bounding
            if self._should_prune(relaxation, is_max):
                continue

            # int check
            if self._is_integer_feasible(relaxation, problem.integer_indicies):
                if self.best_result is None or self._is_better(relaxation, self.best_result, is_max):
                    self.best_result = relaxation
                continue

            # branching
            frac_idx = self._get_fractional_var(relaxation, problem.integer_indicies)
            val = relaxation.solution[frac_idx]
            floor_val, ceil_val = math.floor(val), math.ceil(val)
            
            left_prob = self._add_constraint(node.problem, frac_idx, "<=", floor_val)
            right_prob = self._add_constraint(node.problem, frac_idx, ">=", ceil_val)

            active_nodes.append(BBNode(left_prob, node.depth + 1, frac_idx, f"<={floor_val}"))
            active_nodes.append(BBNode(right_prob, node.depth + 1, frac_idx, f">={ceil_val}"))

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
            return result.optimal_value <= self.best_result.optimal_value - self.tol
        else:
            return result.optimal_value >= self.best_result.optimal_value + self.tol

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
        """Returns the index of the first fractional variable that should be an integer."""
        for idx in int_indices:
            if abs(result.solution[idx] - round(result.solution[idx])) > self.tol:
                return idx
        raise ValueError("No fractional variable found in integer indices")

    def _add_constraint(self, prob: LPProblem, var_idx: int, op: str, bound: float) -> LPProblem:
        """Generates a new LPProblem instance by adding a branching constraint."""
        coeffs = [0.0] * prob.variables_count
        coeffs[var_idx] = 1.0
        new_const = ConstraintData(coefficients=coeffs, operator=op, free_val=bound)
        
        return LPProblem(
            optimization_type=prob.optimization_type,
            objective_coefficients=prob.objective_coefficients.copy(),
            constraints=prob.constraints + [new_const],
            integer_indicies=prob.integer_indicies.copy() if prob.integer_indicies else None,
            variables_count=prob.variables_count
        )