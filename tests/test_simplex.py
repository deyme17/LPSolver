import pytest
import numpy as np
from core.lp_solver import LPSolver
from core.bfs_finder import BasicBFSFinder
from core.simplex_algorithm import SimplexAlgorithm
from utils import LPProblem, ConstraintData, OptimizationType, SolutionStatus


class TestFullSimplexIntegration:
    """Integration test for complete simplex method workflow"""
    
    @pytest.fixture
    def solver(self):
        """Create complete solver with all components"""
        bfs_finder = BasicBFSFinder()
        algorithm = SimplexAlgorithm(max_iterations=100)
        return LPSolver(bfs_finder, algorithm)
    
    def test_complete_minimization_problem(self, solver: LPSolver):
        """
        Complete test: Minimize z = -3x1 - 2x2
        Subject to:
            x1 + x2 <= 4
            2x1 + x2 <= 5
            x1, x2 >= 0
        
        Expected optimal solution: x1=1, x2=3, z=-9
        """
        problem = LPProblem(
            optimization_type=OptimizationType.MINIMIZE.value,
            objective_coefficients=[-3, -2],
            constraints=[
                ConstraintData([1, 1], "<=", 4),
                ConstraintData([2, 1], "<=", 5)
            ],
            variables_count=2
        )
        
        result = solver.solve(problem)
        
        # check solution status
        assert result.status == SolutionStatus.OPTIMAL.value
        
        # check optimal value (z = -9 for minimization)
        assert result.optimal_value is not None
        assert abs(result.optimal_value - (-9.0)) < 1e-6
        
        # check solution vector (x1=1, x2=3)
        assert result.solution is not None
        assert len(result.solution) == 2
        assert abs(result.solution[0] - 1.0) < 1e-6  # x1
        assert abs(result.solution[1] - 3.0) < 1e-6  # x2
    
    def test_unbounded_problem(self, solver: LPSolver):
        """
        Test unbounded problem: Maximize z = x1 + x2
        Subject to:
            -x1 + x2 <= 1
            x1, x2 >= 0
        
        Expected: UNBOUNDED
        """
        problem = LPProblem(
            optimization_type=OptimizationType.MAXIMIZE.value,
            objective_coefficients=[1, 1],
            constraints=[
                ConstraintData([-1, 1], "<=", 1)
            ],
            variables_count=2
        )
        
        result = solver.solve(problem)
        
        assert result.status == SolutionStatus.UNBOUNDED.value
        assert result.error_message is not None
        assert "unbounded" in result.error_message.lower()
    
    def test_three_variables_problem(self, solver: LPSolver):
        """
        Test with 3 variables: Maximize z = 5x1 + 4x2 + 3x3
        Subject to:
            2x1 + 3x2 + x3 <= 5
            4x1 + x2 + 2x3 <= 11
            3x1 + 4x2 + 2x3 <= 8
            x1, x2, x3 >= 0
        """
        problem = LPProblem(
            optimization_type=OptimizationType.MAXIMIZE.value,
            objective_coefficients=[5, 4, 3],
            constraints=[
                ConstraintData([2, 3, 1], "<=", 5),
                ConstraintData([4, 1, 2], "<=", 11),
                ConstraintData([3, 4, 2], "<=", 8)
            ],
            variables_count=3
        )
        
        result = solver.solve(problem)
        
        assert result.status == SolutionStatus.OPTIMAL.value
        assert result.optimal_value is not None
        assert result.solution is not None
        assert len(result.solution) == 3  # no slack
        
        # solution is feasible
        for i, constraint in enumerate(problem.constraints):
            lhs = sum(constraint.coefficients[j] * result.solution[j] 
                     for j in range(3))
            if constraint.operator == "<=":
                assert lhs <= constraint.free_val + 1e-6
    
    def test_mixed_constraints(self, solver: LPSolver):
        """
        Test with mixed constraints: Maximize z = x1 + 2x2
        Subject to:
            x1 + x2 <= 5   (<=)
            x1 + x2 >= 2   (>=)
            2x1 + x2 = 6   (=)
            x1, x2 >= 0
        """
        problem = LPProblem(
            optimization_type=OptimizationType.MAXIMIZE.value,
            objective_coefficients=[1, 2],
            constraints=[
                ConstraintData([1, 1], "<=", 5),
                ConstraintData([1, 1], ">=", 2),
                ConstraintData([2, 1], "=", 6)
            ],
            variables_count=2
        )
        
        result = solver.solve(problem)
        
        # should find optimal solution
        assert result.status == SolutionStatus.OPTIMAL.value
        assert result.optimal_value is not None
        assert result.solution is not None
        assert len(result.solution) == 2
    
    def test_standard_form_conversion(self, solver: LPSolver):
        """Test that standard form is created correctly"""
        problem = LPProblem(
            optimization_type=OptimizationType.MINIMIZE.value,
            objective_coefficients=[1, 2],
            constraints=[
                ConstraintData([1, 1], "<=", 5),
                ConstraintData([2, 1], ">=", 3)
            ],
            variables_count=2
        )
        
        standard_form = solver._build_standard_form(problem)
        
        # check conversion to maximization
        assert standard_form.optimization_type == OptimizationType.MAXIMIZE.value
        # coefficients should be negated
        assert standard_form.objective_coefficients[0] == -1
        assert standard_form.objective_coefficients[1] == -2
        # should have slack variables
        assert len(standard_form.objective_coefficients) == 4  # 2 original + 2 slack
        # all constraints should be equalities
        assert all(c.operator == "=" for c in standard_form.constraints)
    
    def test_solution_feasibility_check(self, solver: LPSolver):
        """verify that returned solution satisfies all constraints"""
        problem = LPProblem(
            optimization_type=OptimizationType.MAXIMIZE.value,
            objective_coefficients=[3, 5],
            constraints=[
                ConstraintData([1, 0], "<=", 4),
                ConstraintData([0, 2], "<=", 12),
                ConstraintData([3, 2], "<=", 18)
            ],
            variables_count=2
        )
        
        result = solver.solve(problem)
        
        assert result.status == SolutionStatus.OPTIMAL.value
        assert result.solution is not None
        
        x1, x2 = result.solution
        
        # verify each constraint
        assert x1 <= 4 + 1e-6
        assert 2 * x2 <= 12 + 1e-6
        assert 3 * x1 + 2 * x2 <= 18 + 1e-6
        
        # verify non-negativity
        assert x1 >= -1e-6
        assert x2 >= -1e-6
        
        # verify objective value
        calculated_z = 3 * x1 + 5 * x2
        assert abs(result.optimal_value - calculated_z) < 1e-6
    
    def test_iteration_history_saved(self, solver: LPSolver):
        """Test that iteration history is properly saved"""
        problem = LPProblem(
            optimization_type=OptimizationType.MAXIMIZE.value,
            objective_coefficients=[3, 2],
            constraints=[
                ConstraintData([1, 1], "<=", 4),
                ConstraintData([2, 1], "<=", 5)
            ],
            variables_count=2
        )
        
        result = solver.solve(problem)
        
        assert result.table is not None
        assert len(result.table.iterations) > 0
        
        # check that each iteration has required structure
        for iteration in result.table.iterations:
            assert "headers" in iteration
            assert "data" in iteration
            assert len(iteration["headers"]) > 0
            assert len(iteration["data"]) > 0
    
    def test_empty_problem(self, solver: LPSolver):
        """Test handling of empty problem"""
        problem = LPProblem(
            optimization_type=OptimizationType.MAXIMIZE.value,
            objective_coefficients=[],
            constraints=[],
            variables_count=0
        )
        
        result = solver.solve(problem)
        
        assert result.status == SolutionStatus.ERROR.value
        assert result.error_message is not None
        assert "empty" in result.error_message.lower()
    
    def test_degenerate_solution(self, solver: LPSolver):
        """Test problem with degenerate solution (basic variable = 0)"""
        problem = LPProblem(
            optimization_type=OptimizationType.MAXIMIZE.value,
            objective_coefficients=[1, 1],
            constraints=[
                ConstraintData([1, 0], "<=", 2),
                ConstraintData([0, 1], "<=", 2),
                ConstraintData([1, 1], "<=", 2)
            ],
            variables_count=2
        )
        
        result = solver.solve(problem)
        
        # should still find optimal solution even with degeneracy
        assert result.status == SolutionStatus.OPTIMAL.value
        assert result.optimal_value is not None
        assert result.solution is not None