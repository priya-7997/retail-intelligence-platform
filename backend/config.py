"""
Configuration settings for Retail Intelligence Platform
Optimized for Indian retail market with INR currency support
"""

import os
from pathlib import Path
from typing import List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings:
    # Application Info
    APP_NAME: str = os.getenv("APP_NAME", "Retail Intelligence Platform")
    APP_VERSION: str = os.getenv("APP_VERSION", "1.0.0")
    DEBUG: bool = os.getenv("DEBUG", "True").lower() == "true"
    SECRET_KEY: str = os.getenv("SECRET_KEY", "retail-intelligence-secret-key-2024")
    
    # Server Configuration
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", 8000))
    RELOAD: bool = os.getenv("RELOAD", "True").lower() == "true"
    
    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./retail_intelligence.db")
    
    # File Handling
    MAX_FILE_SIZE: int = int(os.getenv("MAX_FILE_SIZE", 52428800))  # 50MB
    ALLOWED_EXTENSIONS: List[str] = os.getenv("ALLOWED_EXTENSIONS", "csv,xlsx,xls").split(",")
    
    # Directories
    BASE_DIR: Path = Path(__file__).parent.parent
    UPLOAD_DIR: Path = BASE_DIR / os.getenv("UPLOAD_DIR", "data/uploads")
    PROCESSED_DIR: Path = BASE_DIR / os.getenv("PROCESSED_DIR", "data/processed")
    MODEL_SAVE_DIR: Path = BASE_DIR / os.getenv("MODEL_SAVE_DIR", "backend/models/saved_models")
    
    # Indian Localization
    CURRENCY: str = os.getenv("CURRENCY", "INR")
    CURRENCY_SYMBOL: str = os.getenv("CURRENCY_SYMBOL", "Rs.")
    LOCALE: str = os.getenv("LOCALE", "en_IN")
    TIMEZONE: str = os.getenv("TIMEZONE", "Asia/Kolkata")
    
    # Forecasting Configuration
    DEFAULT_FORECAST_DAYS: int = int(os.getenv("DEFAULT_FORECAST_DAYS", 30))
    MODEL_RETRAIN_THRESHOLD: float = float(os.getenv("MODEL_RETRAIN_THRESHOLD", 0.1))
    
    # API Configuration
    API_V1_PREFIX: str = os.getenv("API_V1_PREFIX", "/api/v1")
    CORS_ORIGINS: List[str] = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:8000").split(",")
    
    # Performance Settings
    CACHE_TTL: int = int(os.getenv("CACHE_TTL", 3600))  # 1 hour
    MAX_WORKERS: int = int(os.getenv("MAX_WORKERS", 4))
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", 1000))
    
    # Model Performance Thresholds
    MIN_ACCURACY_THRESHOLD: float = 0.75
    CONFIDENCE_THRESHOLD: float = 0.8
    
    # Indian Market Specific Settings
    BUSINESS_DAYS_INDIA = [0, 1, 2, 3, 4, 5]  # Monday to Saturday
    INDIAN_HOLIDAYS = [
        "2024-01-26",  # Republic Day
        "2024-03-29",  # Holi
        "2024-08-15",  # Independence Day
        "2024-10-02",  # Gandhi Jayanti
        "2024-11-12",  # Diwali
        "2025-01-26",  # Republic Day 2025
        "2025-03-14",  # Holi 2025
        "2025-08-15",  # Independence Day 2025
        "2025-10-02",  # Gandhi Jayanti 2025
        "2025-11-01",  # Diwali 2025 (estimated)
        # Add more Indian holidays as needed
    ]
    
    # Retail Categories (Indian Market)
    RETAIL_CATEGORIES = [
        "Grocery & Food",
        "Electronics",
        "Fashion & Apparel",
        "Home & Kitchen",
        "Healthcare & Beauty",
        "Books & Stationery",
        "Sports & Fitness",
        "Automotive",
        "Jewelry & Accessories",
        "Mobile & Accessories"
    ]
    
    # Business Intelligence Settings
    INSIGHTS_CONFIG = {
        "trend_threshold": 0.05,  # 5% change for trend detection
        "seasonality_periods": [7, 30, 365],  # Weekly, Monthly, Yearly
        "anomaly_threshold": 2.0,  # Standard deviations for anomaly detection
        "forecast_confidence": 0.80  # 80% confidence intervals
    }
    
    # Alert Thresholds
    ALERT_THRESHOLDS = {
        "stock_out_days": 7,      # Alert when stock will run out in 7 days
        "overstock_ratio": 3.0,   # Alert when stock is 3x normal levels
        "sales_drop_percent": 0.25,  # Alert on 25% sales drop
        "new_trend_strength": 0.3   # Threshold for detecting new trends
    }

# Global settings instance
settings = Settings()

# Currency formatting for Indian Rupees (Windows-safe)
def format_currency(amount: float) -> str:
    """Format amount in Indian Rupees with proper comma separation (Windows-safe)"""
    if amount >= 10000000:  # 1 Crore
        return f"Rs.{amount/10000000:.2f}Cr"
    elif amount >= 100000:  # 1 Lakh
        return f"Rs.{amount/100000:.2f}L"
    elif amount >= 1000:  # 1 Thousand
        return f"Rs.{amount/1000:.2f}K"
    else:
        return f"Rs.{amount:,.2f}"

# Indian number formatting (Windows-safe)
def format_indian_number(number: float) -> str:
    """Format numbers in Indian numbering system (Lakh, Crore) - Windows-safe"""
    if number >= 10000000:  # 1 Crore
        return f"{number/10000000:.2f}Cr"
    elif number >= 100000:  # 1 Lakh
        return f"{number/100000:.2f}L"
    elif number >= 1000:  # 1 Thousand
        return f"{number/1000:.2f}K"
    else:
        return f"{number:,.0f}"

# Validation functions
def validate_csv_columns(columns: List[str]) -> dict:
    """Validate and identify CSV columns for retail data"""
    column_mapping = {
        "date": None,
        "sales": None,
        "quantity": None,
        "product": None,
        "category": None
    }
    
    for col in columns:
        col_lower = col.lower().strip()
        
        # Date column detection
        if any(keyword in col_lower for keyword in ['date', 'time', 'day', 'month']):
            column_mapping["date"] = col
        
        # Sales/Revenue column detection
        elif any(keyword in col_lower for keyword in ['sales', 'revenue', 'amount', 'value', 'price']):
            column_mapping["sales"] = col
        
        # Quantity column detection
        elif any(keyword in col_lower for keyword in ['quantity', 'qty', 'units', 'count']):
            column_mapping["quantity"] = col
        
        # Product column detection
        elif any(keyword in col_lower for keyword in ['product', 'item', 'sku', 'name']):
            column_mapping["product"] = col
        
        # Category column detection
        elif any(keyword in col_lower for keyword in ['category', 'type', 'group', 'class']):
            column_mapping["category"] = col
    
    return column_mapping

# Create necessary directories
def ensure_directories():
    """Ensure all required directories exist"""
    try:
        settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        settings.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        settings.MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        print(f"Error creating directories: {e}")
        return False