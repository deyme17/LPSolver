from abc import ABC, abstractmethod
from typing import Any, Dict, List


class ITable(ABC):
    """Interface for LP algorithm tables (Simplex, Dual, etc.)"""
    @abstractmethod
    def get_table(self) -> Dict[str, Any]:
        """
        Returns a unified table representation.
        Example:
        {
            "headers": ["Basis", "x1", "x2", "b"],
            "data": [
                ["x3", 1, 2, 5],
                ["x4", 0, 1, 3],
                ["Z", -3, -5, 0]
            ]
        }
        """
        pass


class SimplexTable(ITable):
    """Concrete table structure for the Simplex algorithm"""
    def __init__(self, headers: List[str], data: List[List[float]]):
        self.headers = headers
        self.data = data

    def get_table(self) -> Dict[str, Any]:
        """Return structured table data"""
        return {
            "headers": self.headers,
            "data": self.data
        }