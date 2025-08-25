# Tests

This directory contains all test files for the BEVDet3D project, organized with consistent naming conventions.

## Test Files

- **`test_dataset.py`** - Tests Waymo dataset loading and processing functionality
- **`test_training.py`** - Tests the complete PETR training pipeline step-by-step
- **`test_petr_quick.py`** - Quick PETR training test with 3 training steps
- **`test_bevformer_quick.py`** - Quick BEVFormer training test with 3 training steps
- **`run_all_tests.py`** - Test runner script to execute all tests

## Running Tests

### Individual Tests
```bash
# Run from project root directory
python tests/test_dataset.py
python tests/test_training.py
python tests/test_petr_quick.py
python tests/test_bevformer_quick.py
```

### All Tests
```bash
# Run all tests with summary
python tests/run_all_tests.py
```

## Test Coverage

- ✅ Dataset loading and processing (Waymo tfrecord files)
- ✅ Model creation (PETR and BEVFormer)
- ✅ Forward/backward passes
- ✅ Loss computation (Hungarian matching)
- ✅ Training pipeline functionality
- ✅ Evaluation metrics computation
- ✅ Multi-camera input processing
- ✅ 3D bounding box handling
- ✅ Temporal BEV features (BEVFormer)

## Requirements

All tests require:
- CUDA-capable GPU (tested on RTX 4060 Laptop GPU)
- Waymo dataset files in `waymo_open_dataset_v_1_4_3/`
- All dependencies from `requirements.txt`

## Expected Output

All tests should pass with output similar to:
```
🎉 All tests passed!
Results: 4/4 tests passed
```