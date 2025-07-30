"""
API modules for Retail Intelligence Platform
FastAPI routers for upload, forecast, insights, and dashboard endpoints
"""

from . import upload
from . import forecast
from . import insights
from . import dashboard

__all__ = [
    'upload',
    'forecast', 
    'insights',
    'dashboard'
]

__version__ = "1.0.0"