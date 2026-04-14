from utils import ISimplexAlgorithm, IBFSFinder
from typing import Dict

from .lp_solver import LPSolver
from .simplex_table import SimplexTable

# bfs finders
from .bfs import (
    Basic_BFSFinder, TwoPhase_BFSFinder, BigM_BFSFinder
)

bfs_finders: Dict[str, IBFSFinder] = {
    "Basic BFS Finder": Basic_BFSFinder(),
    "Two-Phase BFS Finder": TwoPhase_BFSFinder(),
    "Big-M BFS Finder": BigM_BFSFinder()
}

# solvers
from .algorithms import SimplexAlgorithm, BranchAndBounds

algorithms: Dict[str, ISimplexAlgorithm] = {
    "Basic Simplex method": SimplexAlgorithm(),
    "Branch and Bounds method (int)": BranchAndBounds(),
}