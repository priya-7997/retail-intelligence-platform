import pandas as pd
import re
from typing import List, Dict, Any

def validate_csv_structure(df: pd.DataFrame) -> Dict[str, Any]:
    """Validate CSV structure for retail data"""
    validation = {
        'valid': True,
        'errors': [],
        'warnings': []
    }
    
    if df.empty:
        validation['valid'] = False
        validation['errors'].append("CSV file is empty")
        return validation
    
    # Check for required patterns
    columns = [col.lower().strip() for col in df.columns]
    
    # Look for date column
    date_patterns = ['date', 'time', 'created', 'order']
    has_date = any(pattern in col for col in columns for pattern in date_patterns)
    
    if not has_date:
        validation['warnings'].append("No date column detected")
    
    # Look for sales/amount column
    amount_patterns = ['sales', 'amount', 'revenue', 'total', 'price']
    has_amount = any(pattern in col for col in columns for pattern in amount_patterns)
    
    if not has_amount:
        validation['errors'].append("No sales/amount column detected")
        validation['valid'] = False
    
    return validation

def validate_date_format(date_series: pd.Series) -> bool:
    """Validate date format in series"""
    try:
        pd.to_datetime(date_series, errors='coerce')
        return True
    except:
        return False

def validate_numeric_data(series: pd.Series) -> Dict[str, Any]:
    """Validate numeric data"""
    result = {
        'valid': True,
        'non_numeric_count': 0,
        'negative_count': 0,
        'zero_count': 0
    }
    
    # Convert to numeric
    numeric_series = pd.to_numeric(series, errors='coerce')
    result['non_numeric_count'] = numeric_series.isna().sum()
    
    # Check for negative values
    valid_numbers = numeric_series.dropna()
    result['negative_count'] = (valid_numbers < 0).sum()
    result['zero_count'] = (valid_numbers == 0).sum()
    
    if result['non_numeric_count'] > len(series) * 0.1:  # More than 10% invalid
        result['valid'] = False
    
    return result

