from utils import ISolver, IBFSFinder
from typing import Dict

from .algorithms import SimplexAlgorithm
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
from .solvers.simplex_solver import SimplexSolver

solvers: Dict[str, type[ISolver]] = {
    "Simplex method": SimplexSolver,
}