"""
Model storage and metadata management for Retail Intelligence Platform
Handles trained ML model persistence and performance tracking
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

def get_model_metadata_path() -> Path:
    """Get path to model metadata file"""
    return Path(__file__).parent / "model_metadata.json"

def load_model_metadata() -> Dict[str, Any]:
    """Load model metadata from JSON file"""
    metadata_path = get_model_metadata_path()
    
    if metadata_path.exists():
        try:
            with open(metadata_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    
    # Return default metadata structure
    return {
        "models": {},
        "last_updated": None,
        "version": "1.0.0",
        "model_performance": {}
    }

def save_model_metadata(metadata: Dict[str, Any]) -> bool:
    """Save model metadata to JSON file"""
    try:
        metadata_path = get_model_metadata_path()
        metadata["last_updated"] = datetime.now().isoformat()
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        return True
    except Exception:
        return False

def add_model_record(model_id: str, model_info: Dict[str, Any]) -> bool:
    """Add a new model record to metadata"""
    try:
        metadata = load_model_metadata()
        metadata["models"][model_id] = {
            **model_info,
            "created_at": datetime.now().isoformat()
        }
        return save_model_metadata(metadata)
    except Exception:
        return False

def get_model_record(model_id: str) -> Optional[Dict[str, Any]]:
    """Get model record by ID"""
    metadata = load_model_metadata()
    return metadata["models"].get(model_id)

def remove_model_record(model_id: str) -> bool:
    """Remove model record from metadata"""
    try:
        metadata = load_model_metadata()
        if model_id in metadata["models"]:
            del metadata["models"][model_id]
            return save_model_metadata(metadata)
        return False
    except Exception:
        return False

def list_models() -> Dict[str, Any]:
    """List all models in metadata"""
    metadata = load_model_metadata()
    return metadata["models"]

def update_model_performance(model_id: str, performance: Dict[str, Any]) -> bool:
    """Update model performance metrics"""
    try:
        metadata = load_model_metadata()
        if model_id in metadata["models"]:
            metadata["model_performance"][model_id] = {
                **performance,
                "updated_at": datetime.now().isoformat()
            }
            return save_model_metadata(metadata)
        return False
    except Exception:
        return False

__all__ = [
    'load_model_metadata',
    'save_model_metadata',
    'add_model_record',
    'get_model_record', 
    'remove_model_record',
    'list_models',
    'update_model_performance'
]

__version__ = "1.0.0"
