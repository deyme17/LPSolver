from typing import List, Optional
from utils.constants import ResultConstants

class ResultFormatter:
    """Formats optimization results for display"""
    
    @staticmethod
    def format_optimal_value(value: Optional[float], 
                           decimals: int = ResultConstants.DECIMAL_PLACES) -> str:
        """Format optimal value for display"""
        if value is None:
            return "—"
        return f"{value:.{decimals}f}"
    
    @staticmethod
    def format_solution(solution: Optional[List[float]], 
                       decimals: int = ResultConstants.DECIMAL_PLACES) -> str:
        """Format solution vector for display"""
        if not solution:
            return ""
        
        lines = [f"x_{i+1} = {val:.{decimals}f}" 
                for i, val in enumerate(solution)]
        return "\n".join(lines)
    
    @staticmethod
    def format_table_value(value: float, 
                          decimals: int = ResultConstants.TABLE_DECIMAL_PLACES) -> str:
        """Format table cell value"""
        return f"{value:.{decimals}f}"
    
    @staticmethod
    def format_status(status: str) -> str:
        """Format status text"""
        return status.upper()