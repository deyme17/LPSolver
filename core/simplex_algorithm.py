from utils import LPProblem, LPResult, BFSolution, ISimplexAlgorithm, SolutionStatus
from .simplex_table import SimplexTable


class SimplexAlgorithm(ISimplexAlgorithm):
    """Implementation of the Simplex Method for Linear Programming."""
    def __init__(self, max_iterations: int = 1000):
        """
        Initialize simplex algorithm.
        Args:
            max_iterations: Maximum number of iterations allowed
        """
        self.max_iterations = max_iterations
        self.iteration_count = 0
        self.table = None

    def solve_from_bfs(self, standard_form: LPProblem, initial_solution: BFSolution) -> LPResult:
        """
        Solve LP problem using simplex method from initial BFS.
        Args:
            standard_form: LP in standard form (max, Ax=b, x≥0)
            initial_solution: Initial basic feasible solution
        Returns:
            LPResult with status, optimal value, solution, and table
        """
        try:
            self.table = SimplexTable(standard_form, initial_solution)
            self.iteration_count = 0
            
            while not self.table.is_optimal():
                if self.iteration_count >= self.max_iterations:
                    return self._create_error_result("Max iterations exceeded")
                
                entering_col = self.table.get_entering_variable()
                if entering_col is None: break # idk how we can get here
                
                # is unbounded
                leaving_row = self.table.get_leaving_variable(entering_col)
                if leaving_row is None:
                    return self._create_unbounded_result()
                
                # pivot
                self.table.pivot(leaving_row, entering_col)
                self.iteration_count += 1
            
            return self._create_optimal_result()
            
        except Exception as e:
            return self._create_error_result(f"Algorithm error: {str(e)}")

    def _create_optimal_result(self) -> LPResult:
        """Create result for optimal solution."""
        return LPResult(
            status=SolutionStatus.OPTIMAL,
            optimal_value=self.table.get_objective_value(),
            solution=self.table.get_solution_vector(),
            table=self.table
        )

    def _create_unbounded_result(self) -> LPResult:
        """Create result for unbounded problem."""
        return LPResult(
            status=SolutionStatus.UNBOUNDED,
            error_message="Problem is unbounded",
            table=self.table
        )

    def _create_error_result(self, message: str) -> LPResult:
        """Create result for error case."""
        return LPResult(
            status=SolutionStatus.ERROR,
            error_message=message,
            table=self.table
        )