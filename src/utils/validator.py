"""
Data validation utilities
"""
from typing import Any, List, Dict, Optional
from datetime import datetime
import re

class DataValidator:
    """Validate data quality and integrity"""
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email format"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    @staticmethod
    def validate_not_null(value: Any, field_name: str) -> None:
        """Check if value is not null"""
        if value is None or (isinstance(value, str) and not value.strip()):
            raise ValueError(f"{field_name} cannot be null or empty")
    
    @staticmethod
    def validate_range(value: float, min_val: float, max_val: float, field_name: str) -> None:
        """Validate numeric range"""
        if not (min_val <= value <= max_val):
            raise ValueError(
                f"{field_name} must be between {min_val} and {max_val}, got {value}"
            )
    
    @staticmethod
    def validate_date_format(date_str: str, format_str: str = "%Y-%m-%d") -> bool:
        """Validate date string format"""
        try:
            datetime.strptime(date_str, format_str)
            return True
        except ValueError:
            return False
    
    @staticmethod
    def validate_schema(data: Dict, required_fields: List[str]) -> List[str]:
        """Validate data contains required fields"""
        missing_fields = [field for field in required_fields if field not in data]
        return missing_fields
