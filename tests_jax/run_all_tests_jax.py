#!/usr/bin/env python3

import os
import sys
import unittest
import traceback
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import test modules
from test_dataset_jax import TestWaymoDatasetJAX
from test_petr_jax import TestPETRJAX
from test_bevformer_jax import TestBEVFormerJAX


def run_test_suite(test_class, suite_name):
    """Run a specific test suite and return results."""
    print("=" * 80)
    print(f"RUNNING {suite_name}")
    print("=" * 80)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(test_class)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    start_time = time.time()
    result = runner.run(suite)
    end_time = time.time()
    
    # Print results
    print(f"\n{suite_name} Results:")
    print(f"  Tests run: {result.testsRun}")
    print(f"  Failures: {len(result.failures)}")
    print(f"  Errors: {len(result.errors)}")
    print(f"  Skipped: {len(result.skipped)}")
    print(f"  Time: {end_time - start_time:.2f}s")
    
    if result.failures:
        print("  FAILURES:")
        for test, traceback in result.failures:
            print(f"    - {test}")
    
    if result.errors:
        print("  ERRORS:")
        for test, traceback in result.errors:
            print(f"    - {test}")
    
    if result.skipped:
        print("  SKIPPED:")
        for test, reason in result.skipped:
            print(f"    - {test}: {reason}")
    
    return result


def check_jax_installation():
    """Check if JAX is properly installed and working."""
    try:
        import jax
        import jax.numpy as jnp
        import flax.linen as nn
        
        print(f"JAX version: {jax.__version__}")
        print(f"JAX backend: {jax.lib.xla_bridge.get_backend().platform}")
        
        # Test basic JAX operation
        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.sum(x)
        print(f"JAX basic test: sum([1, 2, 3]) = {y}")
        
        return True
    except Exception as e:
        print(f"JAX installation check failed: {e}")
        return False


def check_dataset_availability():
    """Check if Waymo dataset is available."""
    dataset_path = "waymo_open_dataset_v_1_4_3"
    if os.path.exists(dataset_path):
        print(f"✓ Waymo dataset found at: {dataset_path}")
        return True
    else:
        print(f"⚠ Waymo dataset not found at: {dataset_path}")
        print("  Dataset tests will be skipped")
        return False


def main():
    """Main test runner function."""
    print("=" * 80)
    print("JAX/FLAX IMPLEMENTATION TEST SUITE")
    print("=" * 80)
    
    # Environment checks
    print("\nEnvironment Checks:")
    print("-" * 40)
    
    if not check_jax_installation():
        print("❌ JAX installation check failed. Exiting.")
        return 1
    
    dataset_available = check_dataset_availability()
    
    print("\nStarting Test Execution...")
    print("-" * 40)
    
    # Track overall results
    total_tests = 0
    total_failures = 0
    total_errors = 0
    total_skipped = 0
    start_time = time.time()
    
    # Test suites to run
    test_suites = [
        (TestWaymoDatasetJAX, "WAYMO DATASET JAX TESTS"),
        (TestPETRJAX, "PETR JAX MODEL TESTS"),
        (TestBEVFormerJAX, "BEVFORMER JAX MODEL TESTS"),
    ]
    
    # Run each test suite
    for test_class, suite_name in test_suites:
        try:
            result = run_test_suite(test_class, suite_name)
            total_tests += result.testsRun
            total_failures += len(result.failures)
            total_errors += len(result.errors)
            total_skipped += len(result.skipped)
        except Exception as e:
            print(f"❌ Error running {suite_name}: {e}")
            traceback.print_exc()
            total_errors += 1
    
    # Print overall summary
    end_time = time.time()
    total_time = end_time - start_time
    
    print("\n" + "=" * 80)
    print("OVERALL TEST SUMMARY")
    print("=" * 80)
    print(f"Total tests run: {total_tests}")
    print(f"Failures: {total_failures}")
    print(f"Errors: {total_errors}")
    print(f"Skipped: {total_skipped}")
    print(f"Total time: {total_time:.2f}s")
    
    # Success/failure status
    if total_failures == 0 and total_errors == 0:
        print("\n🎉 ALL TESTS PASSED!")
        return_code = 0
    else:
        print(f"\n❌ TESTS FAILED ({total_failures} failures, {total_errors} errors)")
        return_code = 1
    
    # Performance summary
    if total_tests > 0:
        avg_time = total_time / total_tests
        print(f"Average time per test: {avg_time:.2f}s")
    
    # Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    
    if not dataset_available:
        print("• To run dataset tests, ensure Waymo dataset is available at 'waymo_open_dataset_v_1_4_3'")
    
    if total_failures > 0 or total_errors > 0:
        print("• Check the detailed error messages above")
        print("• Ensure all dependencies are properly installed")
        print("• Verify JAX/CUDA compatibility if using GPU")
    
    if total_tests > 0 and total_failures == 0 and total_errors == 0:
        print("• JAX implementation is working correctly!")
        print("• You can now run training with:")
        print("  python src_jax/training/train_petr_jax.py --help")
        print("  python src_jax/training/train_bevformer_jax.py --help")
    
    return return_code


if __name__ == "__main__":
    sys.exit(main())