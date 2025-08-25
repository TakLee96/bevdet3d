#!/usr/bin/env python3
"""
Test runner script to execute all tests in the project.
"""

import os
import sys
import subprocess

def run_test(test_file):
    """Run a single test file."""
    print(f"\n{'='*60}")
    print(f"Running: {test_file}")
    print('='*60)
    
    try:
        result = subprocess.run([sys.executable, test_file], 
                              cwd=os.path.dirname(os.path.dirname(__file__)),
                              capture_output=False)
        if result.returncode == 0:
            print(f"✅ {test_file} PASSED")
            return True
        else:
            print(f"❌ {test_file} FAILED")
            return False
    except Exception as e:
        print(f"❌ {test_file} ERROR: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Running all tests...")
    
    # List of test files
    test_files = [
        "tests/test_dataset.py",
        "tests/test_training.py", 
        "tests/test_petr_quick.py",
        "tests/test_bevformer_quick.py"
    ]
    
    results = []
    for test_file in test_files:
        success = run_test(test_file)
        results.append((test_file, success))
    
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print('='*60)
    
    passed = 0
    for test_file, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_file}: {status}")
        if success:
            passed += 1
    
    print(f"\nResults: {passed}/{len(test_files)} tests passed")
    
    if passed == len(test_files):
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n💥 {len(test_files) - passed} test(s) failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())