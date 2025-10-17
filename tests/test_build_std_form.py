import pytest
from unittest.mock import Mock
from core.lp_solver import LPSolver, LPProblem, OptimizationType, ConstraintData


class TestBuildStandardForm:
    """Tests for _build_standard_form method"""
    
    @pytest.fixture
    def solver(self):
        """Create solver instance with mocked dependencies"""
        bfs_finder = Mock()
        algorithm = Mock()
        return LPSolver(bfs_finder, algorithm)
    
    def test_maximize_with_less_equal_constraints(self, solver: LPSolver):
        """Test maximization problem with <= constraints"""
        problem = LPProblem(
            optimization_type=OptimizationType.MAXIMIZE,
            objective_coefficients=[3, 2],
            constraints=[
                ConstraintData([1, 1], "<=", 4),
                ConstraintData([2, 1], "<=", 5)
            ],
            variables_count=2
        )
        
        result = solver._build_standard_form(problem)
        
        assert result.optimization_type == OptimizationType.MAXIMIZE
        # 2 original vars + 2 slack vars
        assert result.objective_coefficients == [3, 2, 0, 0]
        assert len(result.constraints) == 2
        assert result.constraints[0].coefficients == [1, 1, 1, 0]
        assert result.constraints[0].operator == "="
        assert result.constraints[0].free_val == 4
        assert result.constraints[1].coefficients == [2, 1, 0, 1]
        assert result.constraints[1].operator == "="
        assert result.constraints[1].free_val == 5
    
    def test_minimize_converts_to_maximize(self, solver: LPSolver):
        """Test minimization converted to maximization"""
        problem = LPProblem(
            optimization_type=OptimizationType.MINIMIZE,
            objective_coefficients=[2, 3],
            constraints=[
                ConstraintData([1, 1], "<=", 5)
            ],
            variables_count=2
        )
        
        result = solver._build_standard_form(problem)
        
        assert result.optimization_type == OptimizationType.MAXIMIZE
        # 2 original vars + 1 slack
        assert result.objective_coefficients == [-2, -3, 0]
    
    def test_greater_equal_constraints(self, solver: LPSolver):
        """Test >= constraints converted to equality"""
        problem = LPProblem(
            optimization_type=OptimizationType.MAXIMIZE,
            objective_coefficients=[1, 2],
            constraints=[
                ConstraintData([1, 2], ">=", 3)
            ],
            variables_count=2
        )
        
        result = solver._build_standard_form(problem)
        
        # 2 original vars + 1 slack var
        assert result.constraints[0].coefficients == [1, 2, -1]
        assert result.constraints[0].operator == "="
        assert result.constraints[0].free_val == 3
    
    def test_negative_free_value_with_less_equal(self, solver: LPSolver):
        """Test negative free val with <= flips to >="""
        problem = LPProblem(
            optimization_type=OptimizationType.MAXIMIZE,
            objective_coefficients=[1, 1],
            constraints=[
                ConstraintData([2, 3], "<=", -6)
            ],
            variables_count=2
        )
        
        result = solver._build_standard_form(problem)
        
        # coeffs negative, free val positive, <= becomes >=, so slack is -1
        assert result.constraints[0].coefficients == [-2, -3, -1]
        assert result.constraints[0].free_val == 6
    
    def test_negative_free_value_with_greater_equal(self, solver: LPSolver):
        """Test negative free val with >= flips to <="""
        problem = LPProblem(
            optimization_type=OptimizationType.MAXIMIZE,
            objective_coefficients=[1, 1],
            constraints=[
                ConstraintData([1, 2], ">=", -4)
            ],
            variables_count=2
        )
        
        result = solver._build_standard_form(problem)
        
        #coeffs negative, free val positive, >= becomes <=, so slack is +1
        assert result.constraints[0].coefficients == [-1, -2, 1]
        assert result.constraints[0].free_val == 4
    
    def test_equality_constraints(self, solver: LPSolver):
        """Test equality constraints remain unchanged"""
        problem = LPProblem(
            optimization_type=OptimizationType.MAXIMIZE,
            objective_coefficients=[1, 2],
            constraints=[
                ConstraintData([3, 4], "=", 10)
            ],
            variables_count=2
        )
        
        result = solver._build_standard_form(problem)
        
        # no slack variables needed
        assert result.constraints[0].coefficients == [3, 4]
        assert result.constraints[0].operator == "="
        assert result.constraints[0].free_val == 10
        assert result.objective_coefficients == [1, 2]  # No slack vars added
    
    def test_multiple_mixed_constraints(self, solver: LPSolver):
        """Test problem with multiple constraint types"""
        problem = LPProblem(
            optimization_type=OptimizationType.MINIMIZE,
            objective_coefficients=[5, 4, 3],
            constraints=[
                ConstraintData([2, 3, 1], "<=", 5),
                ConstraintData([4, 1, 2], ">=", 11),
                ConstraintData([3, 4, 2], "=", 8)
            ],
            variables_count=3
        )
        
        result = solver._build_standard_form(problem)
        
        # check optimization type
        assert result.optimization_type == OptimizationType.MAXIMIZE
        # 3 original + 2 slack
        assert result.objective_coefficients == [-5, -4, -3, 0, 0]
        
        # first constraint (<=): 3 vars + 2 slack vars
        assert result.constraints[0].coefficients == [2, 3, 1, 1, 0]
        assert result.constraints[0].operator == "="
        
        # second constraint (>=): 3 vars + 2 slack vars
        assert result.constraints[1].coefficients == [4, 1, 2, 0, -1]
        assert result.constraints[1].operator == "="
        
        # third constraint (=): 3 vars + 2 slack vars
        assert result.constraints[2].coefficients == [3, 4, 2, 0, 0]
        assert result.constraints[2].operator == "="
    
    def test_variables_count_updated(self, solver: LPSolver):
        """Test variables_count includes slack variables"""
        problem = LPProblem(
            optimization_type=OptimizationType.MAXIMIZE,
            objective_coefficients=[1, 2],
            constraints=[
                ConstraintData([1, 1], "<=", 3),
                ConstraintData([2, 1], ">=", 2)
            ],
            variables_count=2
        )
        
        result = solver._build_standard_form(problem)
        
        # 2 original + 2 slack
        assert result.variables_count == 4
    
    def test_constraint_with_whitespace_operator(self, solver: LPSolver):
        """Test operators with extra whitespace are handled"""
        problem = LPProblem(
            optimization_type=OptimizationType.MAXIMIZE,
            objective_coefficients=[1],
            constraints=[
                ConstraintData([2], "<=", 5)
            ],
            variables_count=1
        )
        
        result = solver._build_standard_form(problem)
        
        assert result.constraints[0].operator == "="
        assert result.constraints[0].coefficients == [2, 1]
    
    def test_single_variable_problem(self, solver: LPSolver):
        """Test problem with single variable"""
        problem = LPProblem(
            optimization_type=OptimizationType.MAXIMIZE,
            objective_coefficients=[3],
            constraints=[
                ConstraintData([1], "<=", 10)
            ],
            variables_count=1
        )
        
        result = solver._build_standard_form(problem)
        
        assert result.objective_coefficients == [3, 0]
        assert result.constraints[0].coefficients == [1, 1]
        assert result.variables_count == 2
    
    def test_zero_coefficients_preserved(self, solver: LPSolver):
        """Test that zero coefficients are preserved"""
        problem = LPProblem(
            optimization_type=OptimizationType.MAXIMIZE,
            objective_coefficients=[0, 5, 0],
            constraints=[
                ConstraintData([0, 1, 2], "<=", 4)
            ],
            variables_count=3
        )
        
        result = solver._build_standard_form(problem)
        
        assert result.objective_coefficients == [0, 5, 0, 0]
        # 3 original vars + 1 slack var
        assert result.constraints[0].coefficients == [0, 1, 2, 1]
    
    def test_two_constraints_different_types(self, solver: LPSolver):
        """Test two constraints with different operators"""
        problem = LPProblem(
            optimization_type=OptimizationType.MAXIMIZE,
            objective_coefficients=[1, 2],
            constraints=[
                ConstraintData([1, 1], "<=", 10),
                ConstraintData([1, 2], ">=", 5)
            ],
            variables_count=2
        )
        
        result = solver._build_standard_form(problem)
        
        # 2 original + 2 slack
        assert result.constraints[0].coefficients == [1, 1, 1, 0]
        assert result.constraints[1].coefficients == [1, 2, 0, -1]
    
    def test_all_equality_constraints(self, solver: LPSolver):
        """Test when all constraints are equalities"""
        problem = LPProblem(
            optimization_type=OptimizationType.MAXIMIZE,
            objective_coefficients=[1, 2],
            constraints=[
                ConstraintData([1, 0], "=", 3),
                ConstraintData([0, 1], "=", 4)
            ],
            variables_count=2
        )
        
        result = solver._build_standard_form(problem)
        
        # no slack vars
        assert result.constraints[0].coefficients == [1, 0]
        assert result.constraints[1].coefficients == [0, 1]
        assert result.objective_coefficients == [1, 2]
        assert result.variables_count == 2
    
    def test_mixed_equality_and_inequality(self, solver: LPSolver):
        """Test mix of = and <= constraints"""
        problem = LPProblem(
            optimization_type=OptimizationType.MAXIMIZE,
            objective_coefficients=[2, 3],
            constraints=[
                ConstraintData([1, 1], "=", 5),
                ConstraintData([2, 1], "<=", 8),
                ConstraintData([1, 3], "=", 6)
            ],
            variables_count=2
        )
        
        result = solver._build_standard_form(problem)
        
        # one slack var
        assert result.objective_coefficients == [2, 3, 0]
        assert result.constraints[0].coefficients == [1, 1, 0]
        assert result.constraints[1].coefficients == [2, 1, 1]
        assert result.constraints[2].coefficients == [1, 3, 0]
    
    def test_operator_case_sensitivity(self, solver: LPSolver):
        """Test that operator comparison is case-sensitive"""
        problem = LPProblem(
            optimization_type=OptimizationType.MAXIMIZE,
            objective_coefficients=[1],
            constraints=[
                ConstraintData([1], "<=", 5),
                ConstraintData([1], ">=", 2)
            ],
            variables_count=1
        )
        
        result = solver._build_standard_form(problem)
        
        assert len(result.objective_coefficients) == 3  # 1 original + 2 slack
        assert result.constraints[0].coefficients[1] == 1   #  +1
        assert result.constraints[1].coefficients[2] == -1  #  -1