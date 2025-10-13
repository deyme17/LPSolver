
from typing import List

class InputValidator:
    """Validates user input data"""
    @staticmethod
    def validate_coefficient(value: str, field_name: str) -> float:
        """Validate and parse coefficient value"""
        try:
            return float(value) if value.strip() else 0.0
        except ValueError:
            raise ValueError(f"Invalid coefficient {field_name}")
    
    @staticmethod
    def validate_coefficients(inputs: List, prefix: str = "a") -> List[float]:
        """Validate list of coefficient inputs"""
        coefficients = []
        for i, line_edit in enumerate(inputs):
            value = InputValidator.validate_coefficient(
                line_edit.text(), 
                f"{prefix}{i+1}"
            )
            coefficients.append(value)
        return coefficients