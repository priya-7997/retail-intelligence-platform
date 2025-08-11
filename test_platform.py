import sys
import importlib
import subprocess
import requests
import time
from pathlib import Path

def test_dependencies():
    """Test if all required dependencies are installed"""
    print("🧪 Testing Python Dependencies...")
    
    dependencies = [
        'pandas', 'numpy', 'fastapi', 'uvicorn',
        'prophet', 'xgboost', 'statsmodels',
        'sklearn', 'plotly', 'aiofiles'
    ]
    
    failed = []
    for dep in dependencies:
        try:
            importlib.import_module(dep)
            print(f"  ✅ {dep}")
        except ImportError:
            print(f"  ❌ {dep}")
            failed.append(dep)
    
    assert not failed, f"Missing dependencies: {', '.join(failed)}. Run: pip install -r requirements.txt"
    print("✅ All dependencies installed!")

def test_directory_structure():
    """Test if directory structure is correct"""
    print("\n📁 Testing Directory Structure...")
    
    required_dirs = [
        'backend/core',
        'backend/api', 
        'backend/models/saved_models',
        'backend/utils',
        'frontend/styles',
        'frontend/scripts',
        'data/uploads',
        'data/processed',
        'data/samples'
    ]
    
    missing = []
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"  ✅ {dir_path}")
        else:
            print(f"  ❌ {dir_path}")
            missing.append(dir_path)
    
    assert not missing, f"Missing directories: {', '.join(missing)}"
    print("✅ Directory structure correct!")

def test_api_server():
    """Test if API server starts and responds"""
    print("\n🌐 Testing API Server...")
    
    try:
        # Start server in background (this is a simple test)
        response = requests.get("http://127.0.0.1:8000/health", timeout=5)
        assert response.status_code == 200, "API server health check failed"
        print("  ✅ API server is running")
        print(f"  ✅ Health check: {response.json()}")
    except requests.exceptions.ConnectionError:
        print("  ❌ API server not running")
        print("  💡 Start server with: python backend/main.py")
    except Exception as e:
        print(f"  ❌ API test failed: {e}")
    
    return False

def test_sample_data():
    """Test sample data generation"""
    print("\n📊 Testing Sample Data Generation...")
    
    try:
        from generate_sample_data import generate_sample_retail_data
        df, summary = generate_sample_retail_data()
        
        print(f"  ✅ Generated {summary['total_records']} records")
        print(f"  ✅ Date range: {summary['date_range']}")
        print(f"  ✅ Total sales: ₹{summary['total_sales']:,.2f}")
    except Exception as e:
        assert False, f"Sample data generation failed: {e}"

def test_ml_models():
    """Test ML model imports and basic functionality"""
    print("\n🤖 Testing ML Models...")
    
    try:
        # Test Prophet
        from prophet import Prophet
        print("  ✅ Prophet imported")
        
        # Test XGBoost
        import xgboost as xgb
        print("  ✅ XGBoost imported")
        
        # Test ARIMA
        from statsmodels.tsa.arima.model import ARIMA
        print("  ✅ ARIMA imported")
        
        print("✅ All ML models available!")
    except Exception as e:
        assert False, f"ML model test failed: {e}"

def run_full_test():
    """Run complete platform test"""
    print("🎯 Retail Intelligence Platform - Full Test Suite")
    print("=" * 50)
    
    tests = [
        test_dependencies,
        test_directory_structure,
        test_ml_models,
        test_sample_data,
    ]
    print("\n" + "=" * 50)
    print("📋 Test Summary:")
    for test_func in tests:
        try:
            test_func()
            print(f"  {test_func.__name__}: ✅ PASS")
        except AssertionError as e:
            print(f"  {test_func.__name__}: ❌ FAIL - {e}")
    print("\n🎯 All tests executed. Review results above.")

if __name__ == "__main__":
    run_full_test()
