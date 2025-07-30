import os
import json
from datetime import datetime
from pathlib import Path

def ensure_directory_exists(path):
    """Ensure directory exists, create if not"""
    Path(path).mkdir(parents=True, exist_ok=True)

def get_file_size(file_path):
    """Get file size in bytes"""
    return os.path.getsize(file_path) if os.path.exists(file_path) else 0

def save_json(data, file_path):
    """Save data as JSON file"""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2, default=str)

def load_json(file_path):
    """Load data from JSON file"""
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    return {}

def get_timestamp():
    """Get current timestamp"""
    return datetime.now().isoformat()
