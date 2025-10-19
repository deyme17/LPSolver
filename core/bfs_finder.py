from utils import LPProblem, BFSolution, IBFSFinder


class BasicBFSFinder(IBFSFinder):
    """Finds initial BFS assuming slack variables form the initial basis."""
    def find_initial_bfs(self, standard_form: LPProblem) -> BFSolution:
        m = len(standard_form.constraints)
        n = standard_form.variables_count

        basis_indices = list(range(n - m, n))
        basic_values = [c.free_val for c in standard_form.constraints]

        full_solution = [0.0] * n
        for i, bi in enumerate(basis_indices):
            full_solution[bi] = basic_values[i]

        return BFSolution(
            basis_indices=basis_indices,
            basic_values=basic_values,
            full_solution=full_solution
        )