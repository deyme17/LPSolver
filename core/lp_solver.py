from utils import (
    LPProblem, LPResult, ConstraintData, 
    SolutionStatus, OptimizationType
    )
from . import IBFSFinder, ISimplexAlgorithm


class LPSolver:
    """A tamplate for solving linear programming problems."""
    def __init__(self, bfs_finder: IBFSFinder, algorithm: ISimplexAlgorithm) -> None:
        """
        Args:
            bfs_finder: Component for finding basic feasible solutions
            algorithm: Simplex algorithm implementation for optimization
        """
        self.bfs_finder = bfs_finder
        self.algorithm = algorithm

    def solve(self, statement: LPProblem) -> LPResult:
        """
        Solve a linear programming problem.
        Args:
            statement (LPProblem): The linear programming problem to solve
        Returns:
            LPResult: The solution containing optimal value and variables
        """
        try:
            if not statement.objective_coefficients or not statement.constraints:
                return LPResult(
                    status=SolutionStatus.ERROR,
                    error_message="Empty objective function or constraints"
                )
            # standard form
            standard_form = self._build_standard_form(statement)

            # initial basic feasible solution
            initial_solution = self.bfs_finder.find_initial_bfs(standard_form)
            if not initial_solution.is_feasible():
                return LPResult(
                    status=SolutionStatus.INFEASIBLE,
                    error_message="No initial BFS found"
                )
            # apply algorithm
            final_result = self.algorithm.solve_from_bfs(standard_form, initial_solution)
            return final_result
        
        except Exception as e:
            return LPResult(
                status=SolutionStatus.ERROR,
                error_message=f"Solver error: {str(e)}"
            )
        
    def _build_standard_form(self, statement: LPProblem) -> LPProblem:
        """
        Builds standard form of Linear Programming statement
        Args:
            statement (LPProblem): The linear programming problem
        Returns:
            LPProblem: The linear programming problem in standard form
        """
        obj_coefs = statement.objective_coefficients.copy()

        # if minimize -> maximize negative function
        if statement.optimization_type == OptimizationType.MINIMIZE:
            obj_coefs = [-a for a in obj_coefs]

        slack_count = 0
        slack_needed = sum(1 for c in statement.constraints if c.operator in ("<=", ">="))
        constraints = []

        # make all constraints '='
        for constraint in statement.constraints:
            coefs = constraint.coefficients.copy()
            free_val = constraint.free_val
            operator = constraint.operator.strip()

            # mult on -1 if free val < 0
            if free_val < 0:
                free_val = -free_val
                coefs = [-a for a in coefs]
                if operator == ">=": operator = "<="
                elif operator == "<=": operator = ">="

            # add slack variables
            slack_vars = [0] * slack_needed
            if operator in ("<=", ">="):
                slack_vars[slack_count] = 1 if operator == "<=" else -1
                slack_count += 1

            new_coeffs = coefs + slack_vars
            constraints.append(ConstraintData(new_coeffs, "=", free_val))

        return LPProblem(
            optimization_type      = OptimizationType.MAXIMIZE,
            objective_coefficients = obj_coefs + [0] * slack_needed,
            constraints            = constraints,
            variables_count        = len(obj_coefs) + slack_needed   
        )