from utils import LPProblem, BFSolution, IBFSFinder
from utils.constants import ConstraintOperator as CO

BIG_M = 1e6


class BigM_BFSFinder(IBFSFinder):
    """
    Finds initial BFS using the Big-M method.
    """
    def __init__(self, big_m: float = BIG_M) -> None:
        self.big_m: float = big_m

    def find_initial_bfs(self, standard_form: LPProblem) -> BFSolution:
        m = len(standard_form.constraints)
        n = standard_form.variables_count
        total_vars = n + m

        # build per-constraint obj coeffs and basis ids
        extra_obj = []
        basis_indices = []

        for i, constraint in enumerate(standard_form.constraints):
            col_idx = n + i
            op = constraint.operator.strip()

            extra = [0.0] * m
            extra[i] = 1.0
            constraint.coefficients += extra
            constraint.operator = CO.EQ.value

            if op == CO.LEQ.value:
                extra_obj.append(0.0)
            else:
                extra_obj.append(-self.big_m)

            basis_indices.append(col_idx)

        standard_form.objective_coefficients += extra_obj
        standard_form.variables_count = total_vars

        basic_values  = [c.free_val for c in standard_form.constraints]
        full_solution = [0.0] * total_vars
        for i, bi in enumerate(basis_indices):
            full_solution[bi] = basic_values[i]

        return BFSolution(
            basis_indices=basis_indices,
            basic_values=basic_values,
            full_solution=full_solution,
        )