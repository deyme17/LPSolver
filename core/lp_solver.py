from utils import LPProblem, LPResult
from . import BFSFinder, ISimplexAlgorithm
from utils import SolutionStatus


class LPSolver:
    """A tamplate for solving linear programming problems."""
    def __init__(self, bfs_finder: BFSFinder, algorithm: type[ISimplexAlgorithm]) -> None:
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
            if not initial_solution:
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
        pass