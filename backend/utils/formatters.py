from typing import Any, Dict
import json

def format_api_response(success: bool, data: Any = None, error: str = None, **kwargs) -> Dict:
    """Format standard API response"""
    response = {
        'success': success,
        **kwargs
    }
    
    if success:
        response['data'] = data
    else:
        response['error'] = error
    
    return response

def format_error_response(error: str, status_code: int = 400) -> Dict:
    """Format error response"""
    return {
        'success': False,
        'error': error,
        'status_code': status_code
    }

def format_currency_inr(amount: float) -> str:
    """Format amount in Indian Rupees"""
    if amount >= 10000000:  # 1 Crore
        return f"₹{amount/10000000:.2f}Cr"
    elif amount >= 100000:  # 1 Lakh
        return f"₹{amount/100000:.2f}L"
    elif amount >= 1000:  # 1 Thousand
        return f"₹{amount/1000:.2f}K"
    else:
        return f"₹{amount:,.2f}"

def clean_text_for_json(text: str) -> str:
    """Clean text for JSON serialization"""
    if not isinstance(text, str):
        return str(text)
    
    # Remove problematic characters
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = ' '.join(text.split())  # Normalize whitespace
    
    return text