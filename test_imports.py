#!/usr/bin/env python3
"""
Simple test script to validate that all modules can be imported correctly.
This helps catch any import errors before running the full pipeline.
"""

def test_imports():
    """Test that all modules can be imported without errors."""
    print("Testing module imports...")
    
    try:
        print("  ✓ config.py", end="")
        import config
        print(" - OK")
        
        print("  ✓ utils.py", end="")
        import utils
        print(" - OK")
        
        print("  ✓ data_ingestion.py", end="")
        import data_ingestion
        print(" - OK")
        
        print("  ✓ feature_engineering.py", end="")
        import feature_engineering
        print(" - OK")
        
        print("  ✓ labeling.py", end="")
        import labeling
        print(" - OK")
        
        print("  ✓ model_training.py", end="")
        import model_training
        print(" - OK")
        
        print("  ✓ model_monitoring.py", end="")
        import model_monitoring
        print(" - OK")
        
        print("\n✅ All modules imported successfully!")
        print("\nNote: Snowflake package warnings are expected in development environment.")
        return True
        
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return False

def test_config():
    """Test configuration validation."""
    print("\nTesting configuration...")
    
    try:
        from config import config, feature_config
        
        # Test basic config validation
        assert config.CHURN_WINDOW > 0, "CHURN_WINDOW must be positive"
        assert 0 < config.RETRAIN_THRESHOLD <= 1, "RETRAIN_THRESHOLD must be between 0 and 1"
        assert len(config.feature_cols) > 0, "Feature columns must be defined"
        
        print("  ✓ Configuration validation passed")
        return True
        
    except Exception as e:
        print(f"  ❌ Configuration error: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 50)
    print("ML Pipeline Module Test")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 2
    
    if test_imports():
        tests_passed += 1
    
    if test_config():
        tests_passed += 1
    
    print("\n" + "=" * 50)
    print(f"Test Results: {tests_passed}/{total_tests} passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! Pipeline is ready to run.")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    exit(main())
