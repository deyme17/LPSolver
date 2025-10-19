import pytest
from core.bfs_finder import BasicBFSFinder
from utils import LPProblem, ConstraintData, OptimizationType


class TestBasicBFSFinder:
    """Tests for BasicBFSFinder class"""
    
    @pytest.fixture
    def finder(self):
        """Create BasicBFSFinder instance"""
        return BasicBFSFinder()
    
    def test_simple_two_constraints(self, finder: BasicBFSFinder):
        """Test BFS with 2 variables and 2 slack variables"""
        problem = LPProblem(
            optimization_type=OptimizationType.MAXIMIZE.value,
            objective_coefficients=[3, 2, 0, 0],
            constraints=[
                ConstraintData([1, 1, 1, 0], "=", 4),
                ConstraintData([2, 1, 0, 1], "=", 5)
            ],
            variables_count=4  # 2 original + 2 slack
        )
        
        result = finder.find_initial_bfs(problem)
        
        # last 2 slack variables should be in basis
        assert result.basis_indices == [2, 3]
        assert result.basic_values == [4, 5]
        assert result.full_solution == [0.0, 0.0, 4.0, 5.0]
        assert result.is_feasible() is True
    
    def test_three_constraints(self, finder: BasicBFSFinder):
        """Test BFS with 3 constraints"""
        problem = LPProblem(
            optimization_type=OptimizationType.MAXIMIZE.value,
            objective_coefficients=[5, 4, 3, 0, 0, 0],
            constraints=[
                ConstraintData([2, 3, 1, 1, 0, 0], "=", 10),
                ConstraintData([4, 1, 2, 0, 1, 0], "=", 15),
                ConstraintData([3, 4, 2, 0, 0, 1], "=", 20)
            ],
            variables_count=6  # 3 original + 3 slack
        )
        
        result = finder.find_initial_bfs(problem)
        
        assert result.basis_indices == [3, 4, 5]
        assert result.basic_values == [10, 15, 20]
        assert result.full_solution == [0.0, 0.0, 0.0, 10.0, 15.0, 20.0]
        assert result.is_feasible() is True
    
    def test_single_constraint(self, finder: BasicBFSFinder):
        """Test BFS with single constraint"""
        problem = LPProblem(
            optimization_type=OptimizationType.MAXIMIZE.value,
            objective_coefficients=[1, 2, 0],
            constraints=[
                ConstraintData([1, 1, 1], "=", 8)
            ],
            variables_count=3  # 2 original + 1 slack
        )
        
        result = finder.find_initial_bfs(problem)
        
        assert result.basis_indices == [2]
        assert result.basic_values == [8]
        assert result.full_solution == [0.0, 0.0, 8.0]
        assert result.is_feasible() is True
    
    def test_four_variables_two_constraints(self, finder: BasicBFSFinder):
        """Test with more variables than constraints"""
        problem = LPProblem(
            optimization_type=OptimizationType.MAXIMIZE.value,
            objective_coefficients=[1, 2, 3, 4, 0, 0],
            constraints=[
                ConstraintData([1, 0, 1, 0, 1, 0], "=", 6),
                ConstraintData([0, 1, 0, 1, 0, 1], "=", 9)
            ],
            variables_count=6  # 4 original + 2 slack
        )
        
        result = finder.find_initial_bfs(problem)
        
        assert result.basis_indices == [4, 5]
        assert result.basic_values == [6, 9]
        assert result.full_solution == [0.0, 0.0, 0.0, 0.0, 6.0, 9.0]
        assert result.is_feasible() is True
    
    def test_zero_free_values(self, finder: BasicBFSFinder):
        """Test when constraints have zero on RHS"""
        problem = LPProblem(
            optimization_type=OptimizationType.MAXIMIZE.value,
            objective_coefficients=[1, 1, 0, 0],
            constraints=[
                ConstraintData([1, 0, 1, 0], "=", 0),
                ConstraintData([0, 1, 0, 1], "=", 0)
            ],
            variables_count=4
        )
        
        result = finder.find_initial_bfs(problem)
        
        assert result.basis_indices == [2, 3]
        assert result.basic_values == [0, 0]
        assert result.full_solution == [0.0, 0.0, 0.0, 0.0]
        assert result.is_feasible() is True
    
    def test_large_free_values(self, finder: BasicBFSFinder):
        """Test with large RHS values"""
        problem = LPProblem(
            optimization_type=OptimizationType.MAXIMIZE.value,
            objective_coefficients=[1, 2, 0, 0],
            constraints=[
                ConstraintData([1, 1, 1, 0], "=", 1000),
                ConstraintData([2, 1, 0, 1], "=", 5000)
            ],
            variables_count=4
        )
        
        result = finder.find_initial_bfs(problem)
        
        assert result.basis_indices == [2, 3]
        assert result.basic_values == [1000, 5000]
        assert result.full_solution == [0.0, 0.0, 1000.0, 5000.0]
        assert result.is_feasible() is True
    
    def test_fractional_free_values(self, finder: BasicBFSFinder):
        """Test with fractional RHS values"""
        problem = LPProblem(
            optimization_type=OptimizationType.MAXIMIZE.value,
            objective_coefficients=[1, 1, 0, 0],
            constraints=[
                ConstraintData([1, 1, 1, 0], "=", 3.5),
                ConstraintData([1, 2, 0, 1], "=", 7.25)
            ],
            variables_count=4
        )
        
        result = finder.find_initial_bfs(problem)
        
        assert result.basis_indices == [2, 3]
        assert result.basic_values == [3.5, 7.25]
        assert result.full_solution == [0.0, 0.0, 3.5, 7.25]
        assert result.is_feasible() is True
    
    def test_mixed_positive_zero_values(self, finder: BasicBFSFinder):
        """Test mix of positive and zero RHS values"""
        problem = LPProblem(
            optimization_type=OptimizationType.MAXIMIZE.value,
            objective_coefficients=[2, 3, 0, 0, 0],
            constraints=[
                ConstraintData([1, 0, 1, 0, 0], "=", 5),
                ConstraintData([0, 1, 0, 1, 0], "=", 0),
                ConstraintData([1, 1, 0, 0, 1], "=", 10)
            ],
            variables_count=5
        )
        
        result = finder.find_initial_bfs(problem)
        
        assert result.basis_indices == [2, 3, 4]
        assert result.basic_values == [5, 0, 10]
        assert result.full_solution == [0.0, 0.0, 5.0, 0.0, 10.0]
        assert result.is_feasible() is True
    
    def test_equality_constraints_only(self, finder: BasicBFSFinder):
        """Test problem with only equality constraints (no slack vars)"""
        # edge case - no slack variables
        problem = LPProblem(
            optimization_type=OptimizationType.MAXIMIZE.value,
            objective_coefficients=[1, 2],
            constraints=[
                ConstraintData([1, 0], "=", 3),
                ConstraintData([0, 1], "=", 4)
            ],
            variables_count=2
        )
        
        result = finder.find_initial_bfs(problem)
        
        # n=2, m=2, so basis starts at index 0
        assert result.basis_indices == [0, 1]
        assert result.basic_values == [3, 4]
        assert result.full_solution == [3.0, 4.0]
        assert result.is_feasible() is True
    
    def test_one_variable_one_constraint(self, finder: BasicBFSFinder):
        """Test minimal case: 1 variable, 1 constraint"""
        problem = LPProblem(
            optimization_type=OptimizationType.MAXIMIZE.value,
            objective_coefficients=[3, 0],
            constraints=[
                ConstraintData([1, 1], "=", 10)
            ],
            variables_count=2  # 1 original + 1 slack
        )
        
        result = finder.find_initial_bfs(problem)
        
        assert result.basis_indices == [1]
        assert result.basic_values == [10]
        assert result.full_solution == [0.0, 10.0]
        assert result.is_feasible() is True
    
    def test_basis_indices_order(self, finder: BasicBFSFinder):
        """Test that basis indices are in correct order"""
        problem = LPProblem(
            optimization_type=OptimizationType.MAXIMIZE.value,
            objective_coefficients=[1, 2, 3, 0, 0, 0, 0],
            constraints=[
                ConstraintData([1, 0, 0, 1, 0, 0, 0], "=", 2),
                ConstraintData([0, 1, 0, 0, 1, 0, 0], "=", 4),
                ConstraintData([0, 0, 1, 0, 0, 1, 0], "=", 6),
                ConstraintData([1, 1, 1, 0, 0, 0, 1], "=", 8)
            ],
            variables_count=7  # 3 original + 4 slack
        )
        
        result = finder.find_initial_bfs(problem)
        
        # Last 4 variables
        assert result.basis_indices == [3, 4, 5, 6]
        assert result.basic_values == [2, 4, 6, 8]
        assert result.full_solution == [0.0, 0.0, 0.0, 2.0, 4.0, 6.0, 8.0]
    
    def test_is_feasible_all_positive(self, finder: BasicBFSFinder):
        """Test is_feasible returns True for all positive values"""
        problem = LPProblem(
            optimization_type=OptimizationType.MAXIMIZE.value,
            objective_coefficients=[1, 2, 0, 0],
            constraints=[
                ConstraintData([1, 1, 1, 0], "=", 5),
                ConstraintData([2, 1, 0, 1], "=", 8)
            ],
            variables_count=4
        )
        
        result = finder.find_initial_bfs(problem)
        assert result.is_feasible() is True
    
    def test_is_feasible_with_zeros(self, finder: BasicBFSFinder):
        """Test is_feasible returns True when some values are zero"""
        problem = LPProblem(
            optimization_type=OptimizationType.MAXIMIZE.value,
            objective_coefficients=[1, 1, 0, 0],
            constraints=[
                ConstraintData([1, 0, 1, 0], "=", 0),
                ConstraintData([0, 1, 0, 1], "=", 5)
            ],
            variables_count=4
        )
        
        result = finder.find_initial_bfs(problem)
        assert result.is_feasible() is True