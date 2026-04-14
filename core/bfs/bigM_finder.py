from utils import LPProblem, BFSolution, IBFSFinder, ConstraintData
from utils.constants import ConstraintOperator as CO

BIG_M = 1e6


class BigM_BFSFinder(IBFSFinder):
    """
    Finds initial BFS using the Big-M method.
    """
    def __init__(self, big_m: float = BIG_M) -> None:
        self.big_m = big_m

    def find_initial_bfs(self, standard_form: LPProblem) -> BFSolution:
        m = len(standard_form.constraints)
        n = standard_form.variables_count
        total_vars = n + m

        extra_obj = []
        basis_indices = []
        new_constraints = []

        for i, constraint in enumerate(standard_form.constraints):
            col_idx = n + i
            op = constraint.operator.strip()

            extra = [0.0] * m
            extra[i] = 1.0

            new_coefs = list(constraint.coefficients) + extra
            new_constraints.append(
                ConstraintData(new_coefs, CO.EQ.value, constraint.free_val)
            )

            if op == CO.LEQ.value:
                extra_obj.append(0.0)
            elif op in (CO.GEQ.value, CO.EQ.value):
                extra_obj.append(-self.big_m)
            else:
                raise ValueError(f"Unexpected constraint operator: '{op}'")

            basis_indices.append(col_idx)

        augmented = LPProblem(
            optimization_type=standard_form.optimization_type,
            objective_coefficients=list(standard_form.objective_coefficients) + extra_obj,
            constraints=new_constraints,
            variables_count=total_vars,
        )

        basic_values = [c.free_val for c in augmented.constraints]
        full_solution = [0.0] * total_vars
        for i, bi in enumerate(basis_indices):
            full_solution[bi] = basic_values[i]

        standard_form.objective_coefficients = augmented.objective_coefficients
        standard_form.constraints = augmented.constraints
        standard_form.variables_count = augmented.variables_count

        return BFSolution(
            basis_indices=basis_indices,
            basic_values=basic_values,
            full_solution=full_solution,
        )