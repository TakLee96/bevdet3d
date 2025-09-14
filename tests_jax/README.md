# JAX/Flax Implementation Tests

This directory contains comprehensive unit tests for the JAX/Flax implementation of PETR and BEVFormer models.

## Test Coverage

### 1. Dataset Tests (`test_dataset_jax.py`)
- ✅ Dataset creation and loading
- ✅ Sample format validation (NHWC format)
- ✅ Batch loading functionality
- ✅ Data loader iterator
- ✅ Value ranges and data quality
- ✅ Cache functionality

### 2. PETR Model Tests (`test_petr_jax.py`)
- ✅ 3D Positional Encoding
- ✅ Multi-Head Attention
- ✅ PETR Transformer
- ✅ PETR Detection Head
- ✅ Complete PETR Model
- ✅ Loss function computation
- ✅ Metrics computation (mAP)
- ✅ Gradient computation

### 3. BEVFormer Model Tests (`test_bevformer_jax.py`)
- ✅ 2D Learned Positional Encoding
- ✅ Spatial Cross-Attention
- ✅ Temporal Self-Attention
- ✅ BEVFormer Layer
- ✅ BEVFormer Encoder
- ✅ Complete BEVFormer Model
- ✅ Temporal modeling capability
- ✅ Gradient computation

## Running Tests

### Run All Tests
```bash
python tests_jax/run_all_tests_jax.py
```

### Run Individual Test Suites
```bash
# Dataset tests
python tests_jax/test_dataset_jax.py

# PETR model tests
python tests_jax/test_petr_jax.py

# BEVFormer model tests
python tests_jax/test_bevformer_jax.py
```

## Test Requirements

### Dependencies
- JAX >= 0.6.0
- Flax >= 0.10.0
- NumPy
- SciPy (for Hungarian matching)
- TensorFlow (for Waymo dataset loading)
- OpenCV (for image processing)

### Dataset (Optional)
- Waymo Open Dataset v1.4.3 at `waymo_open_dataset_v_1_4_3/`
- If dataset is not available, dataset-related tests will be skipped

### Hardware
- Tests are designed to run on both CPU and GPU
- GPU recommended for larger models and full training

## Test Structure

Each test file follows this pattern:
1. **Unit Tests**: Test individual components
2. **Integration Tests**: Test component interactions
3. **End-to-End Tests**: Test complete workflows
4. **Gradient Tests**: Verify backpropagation works

## Key Differences from PyTorch Tests

### 1. Data Format
- **PyTorch**: NCHW format (batch, channels, height, width)
- **JAX**: NHWC format (batch, height, width, channels)

### 2. Model Initialization
- **PyTorch**: `model = Model(); model(input)`
- **JAX**: `params = model.init(key, input); model.apply(params, input)`

### 3. Random Number Generation
- **PyTorch**: `torch.randn()` with global state
- **JAX**: `jax.random.normal(key)` with explicit keys

### 4. Gradients
- **PyTorch**: `loss.backward()` with implicit computation graph
- **JAX**: `jax.grad(loss_fn)(params)` with functional approach

## Performance Benchmarks

Expected test performance on different hardware:

| Hardware | Dataset Tests | PETR Tests | BEVFormer Tests | Total |
|----------|---------------|------------|-----------------|-------|
| CPU (8 cores) | ~30s | ~45s | ~60s | ~135s |
| GPU (RTX 4060) | ~15s | ~20s | ~25s | ~60s |
| GPU (V100) | ~10s | ~15s | ~18s | ~43s |

## Troubleshooting

### Common Issues

1. **JAX Installation**
   ```bash
   pip install --upgrade jax jaxlib
   # For GPU support:
   pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
   ```

2. **Memory Issues**
   - Reduce batch sizes in tests
   - Use smaller model dimensions
   - Enable XLA memory optimization:
     ```python
     os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.8'
     ```

3. **CUDA/GPU Issues**
   - Check JAX GPU installation: `jax.devices()`
   - Verify CUDA compatibility
   - Fall back to CPU if needed: `export JAX_PLATFORM_NAME=cpu`

4. **Dataset Issues**
   - Ensure Waymo dataset is properly downloaded
   - Check file permissions
   - Tests will skip if dataset unavailable

### Debug Mode
Run tests with detailed output:
```bash
python -m pytest tests_jax/ -v -s
```

## Contributing

When adding new tests:
1. Follow existing naming conventions
2. Include both positive and negative test cases
3. Test edge cases and error conditions
4. Verify gradient computation for trainable components
5. Add performance benchmarks for critical paths

## Test Data

Tests use synthetic data generated with JAX random number generators for reproducibility. Real Waymo data is only used in dataset tests when available.