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
    
    if failed:
        print(f"\n❌ Missing dependencies: {', '.join(failed)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies installed!")
    return True

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
    
    if missing:
        print(f"\n❌ Missing directories: {', '.join(missing)}")
        return False
    
    print("✅ Directory structure correct!")
    return True

def test_api_server():
    """Test if API server starts and responds"""
    print("\n🌐 Testing API Server...")
    
    try:
        # Start server in background (this is a simple test)
        response = requests.get("http://127.0.0.1:8000/health", timeout=5)
        if response.status_code == 200:
            print("  ✅ API server is running")
            print(f"  ✅ Health check: {response.json()}")
            return True
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
        
        return True
    except Exception as e:
        print(f"  ❌ Sample data generation failed: {e}")
        return False

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
        return True
    except Exception as e:
        print(f"  ❌ ML model test failed: {e}")
        return False

def run_full_test():
    """Run complete platform test"""
    print("🎯 Retail Intelligence Platform - Full Test Suite")
    print("=" * 50)
    
    tests = [
        ("Dependencies", test_dependencies),
        ("Directory Structure", test_directory_structure),
        ("ML Models", test_ml_models),
        ("Sample Data", test_sample_data),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 Test Summary:")
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Tests Passed: {passed}/{len(tests)}")
    
    if passed == len(tests):
        print("🎉 All tests passed! Platform is ready to use.")
        print("\n🚀 Next steps:")
        print("  1. Run: python backend/main.py")
        print("  2. Open: http://127.0.0.1:8000")
        print("  3. Upload the generated sample data")
        print("  4. Generate insights and forecasts")
    else:
        print("⚠️  Some tests failed. Please fix issues before proceeding.")
    
    return passed == len(tests)

if __name__ == "__main__":
    run_full_test()
