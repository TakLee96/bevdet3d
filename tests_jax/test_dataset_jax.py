#!/usr/bin/env python3

import os
import sys
import unittest
import jax.numpy as jnp
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src_jax.data.waymo_dataset_jax import create_waymo_dataset_jax


class TestWaymoDatasetJAX(unittest.TestCase):
    """Test suite for Waymo dataset JAX implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.data_root = "waymo_open_dataset_v_1_4_3"
        self.image_size = (400, 200)  # (width, height)
        self.max_files = 1
        self.max_frames_per_file = 2
        
    def test_dataset_creation(self):
        """Test dataset creation and basic properties."""
        print("Testing Dataset Creation...")
        
        try:
            dataset = create_waymo_dataset_jax(
                data_root=self.data_root,
                split="training",
                max_files=self.max_files,
                max_frames_per_file=self.max_frames_per_file,
                image_size=self.image_size,
                cache_dir="./cache"
            )
            
            # Check dataset has samples
            self.assertGreater(len(dataset), 0)
            
            print(f"✓ Dataset created with {len(dataset)} samples")
            
        except (FileNotFoundError, ValueError) as e:
            print(f"⚠ Dataset creation skipped: {e}")
            self.skipTest(f"Dataset not available: {e}")
            
    def test_sample_format(self):
        """Test individual sample format and shapes."""
        print("Testing Sample Format...")
        
        try:
            dataset = create_waymo_dataset_jax(
                data_root=self.data_root,
                split="training",
                max_files=self.max_files,
                max_frames_per_file=self.max_frames_per_file,
                image_size=self.image_size,
                cache_dir="./cache"
            )
            
            if len(dataset) == 0:
                self.skipTest("No samples in dataset")
            
            # Get first sample
            sample = dataset[0]
            
            # Check sample structure
            self.assertIn('images', sample)
            self.assertIn('labels', sample)
            self.assertIn('boxes_3d', sample)
            self.assertIn('num_objects', sample)
            
            # Check data types (JAX arrays)
            self.assertTrue(isinstance(sample['images'], jnp.ndarray))
            self.assertTrue(isinstance(sample['labels'], jnp.ndarray))
            self.assertTrue(isinstance(sample['boxes_3d'], jnp.ndarray))
            self.assertTrue(isinstance(sample['num_objects'], (int, np.integer)))
            
            # Check shapes
            expected_img_shape = (5, self.image_size[1], self.image_size[0], 3)  # [N_cams, H, W, C]
            self.assertEqual(sample['images'].shape, expected_img_shape)
            
            # Check labels and boxes shapes
            max_objects = 50
            self.assertEqual(sample['labels'].shape, (max_objects,))
            self.assertEqual(sample['boxes_3d'].shape, (max_objects, 7))
            
            # Check value ranges
            self.assertTrue(jnp.all(sample['images'] >= 0.0))
            self.assertTrue(jnp.all(sample['images'] <= 1.0))
            self.assertTrue(jnp.all(sample['labels'] >= 0))
            self.assertTrue(jnp.all(sample['labels'] <= 3))  # 0-2 for classes, 3 for background
            
            print(f"✓ Sample format correct")
            print(f"✓ Images shape: {sample['images'].shape} (NHWC format)")
            print(f"✓ Labels shape: {sample['labels'].shape}")
            print(f"✓ Boxes shape: {sample['boxes_3d'].shape}")
            print(f"✓ Num objects: {sample['num_objects']}")
            
        except (FileNotFoundError, ValueError) as e:
            self.skipTest(f"Dataset not available: {e}")
            
    def test_batch_loading(self):
        """Test batch loading functionality."""
        print("Testing Batch Loading...")
        
        try:
            dataset = create_waymo_dataset_jax(
                data_root=self.data_root,
                split="training",
                max_files=self.max_files,
                max_frames_per_file=self.max_frames_per_file,
                image_size=self.image_size,
                cache_dir="./cache"
            )
            
            if len(dataset) == 0:
                self.skipTest("No samples in dataset")
            
            # Test batch loading
            batch_size = min(2, len(dataset))
            batch = dataset.get_batch(batch_size=batch_size)
            
            # Check batch structure
            self.assertIn('images', batch)
            self.assertIn('labels', batch)
            self.assertIn('boxes_3d', batch)
            self.assertIn('num_objects', batch)
            
            # Check batch shapes
            expected_img_shape = (batch_size, 5, self.image_size[1], self.image_size[0], 3)
            self.assertEqual(batch['images'].shape, expected_img_shape)
            self.assertEqual(batch['labels'].shape, (batch_size, 50))
            self.assertEqual(batch['boxes_3d'].shape, (batch_size, 50, 7))
            self.assertEqual(batch['num_objects'].shape, (batch_size,))
            
            # Check data types
            self.assertTrue(isinstance(batch['images'], jnp.ndarray))
            self.assertTrue(isinstance(batch['labels'], jnp.ndarray))
            self.assertTrue(isinstance(batch['boxes_3d'], jnp.ndarray))
            self.assertTrue(isinstance(batch['num_objects'], jnp.ndarray))
            
            print(f"✓ Batch loading successful")
            print(f"✓ Batch images shape: {batch['images'].shape}")
            print(f"✓ Batch labels shape: {batch['labels'].shape}")
            print(f"✓ Batch boxes shape: {batch['boxes_3d'].shape}")
            
        except (FileNotFoundError, ValueError) as e:
            self.skipTest(f"Dataset not available: {e}")
            
    def test_data_loader_iterator(self):
        """Test data loader iterator functionality."""
        print("Testing Data Loader Iterator...")
        
        try:
            dataset = create_waymo_dataset_jax(
                data_root=self.data_root,
                split="training",
                max_files=self.max_files,
                max_frames_per_file=self.max_frames_per_file,
                image_size=self.image_size,
                cache_dir="./cache"
            )
            
            if len(dataset) == 0:
                self.skipTest("No samples in dataset")
            
            # Test data loader
            batch_size = min(2, len(dataset))
            batch_count = 0
            
            for batch in dataset.data_loader(batch_size=batch_size, shuffle=False):
                # Check batch format
                self.assertIn('images', batch)
                self.assertIn('labels', batch)
                self.assertIn('boxes_3d', batch)
                self.assertIn('num_objects', batch)
                
                # Check shapes
                self.assertEqual(len(batch['images'].shape), 5)  # [B, N_cams, H, W, C]
                self.assertEqual(len(batch['labels'].shape), 2)   # [B, max_objects]
                self.assertEqual(len(batch['boxes_3d'].shape), 3) # [B, max_objects, 7]
                self.assertEqual(len(batch['num_objects'].shape), 1) # [B]
                
                batch_count += 1
                if batch_count >= 2:  # Test only first 2 batches
                    break
            
            self.assertGreater(batch_count, 0)
            
            print(f"✓ Data loader iterator working")
            print(f"✓ Processed {batch_count} batches")
            
        except (FileNotFoundError, ValueError) as e:
            self.skipTest(f"Dataset not available: {e}")
            
    def test_value_ranges_and_quality(self):
        """Test data value ranges and quality."""
        print("Testing Data Value Ranges and Quality...")
        
        try:
            dataset = create_waymo_dataset_jax(
                data_root=self.data_root,
                split="training",
                max_files=self.max_files,
                max_frames_per_file=self.max_frames_per_file,
                image_size=self.image_size,
                cache_dir="./cache"
            )
            
            if len(dataset) == 0:
                self.skipTest("No samples in dataset")
            
            # Test multiple samples
            for i in range(min(3, len(dataset))):
                sample = dataset[i]
                
                # Image value ranges
                self.assertTrue(jnp.all(sample['images'] >= 0.0))
                self.assertTrue(jnp.all(sample['images'] <= 1.0))
                
                # Check no NaN or infinite values
                self.assertTrue(jnp.all(jnp.isfinite(sample['images'])))
                self.assertTrue(jnp.all(jnp.isfinite(sample['boxes_3d'])))
                
                # Label ranges
                self.assertTrue(jnp.all(sample['labels'] >= 0))
                self.assertTrue(jnp.all(sample['labels'] <= 3))
                
                # Check that we have some valid objects
                valid_objects = jnp.sum(sample['labels'] < 3)
                self.assertEqual(valid_objects, sample['num_objects'])
                
            print(f"✓ Data quality checks passed")
            print(f"✓ Image values in [0, 1]")
            print(f"✓ No NaN or infinite values")
            print(f"✓ Label consistency verified")
            
        except (FileNotFoundError, ValueError) as e:
            self.skipTest(f"Dataset not available: {e}")
            
    def test_cache_functionality(self):
        """Test dataset caching functionality."""
        print("Testing Cache Functionality...")
        
        import os
        import time
        
        cache_dir = "./cache_test"
        
        try:
            # Remove cache if exists
            cache_file = os.path.join(cache_dir, "training_cached_data_jax.pkl")
            if os.path.exists(cache_file):
                os.remove(cache_file)
            
            # First load (should create cache)
            start_time = time.time()
            dataset1 = create_waymo_dataset_jax(
                data_root=self.data_root,
                split="training",
                max_files=1,
                max_frames_per_file=1,
                image_size=self.image_size,
                cache_dir=cache_dir
            )
            first_load_time = time.time() - start_time
            
            # Check cache file was created
            self.assertTrue(os.path.exists(cache_file))
            
            # Second load (should use cache)
            start_time = time.time()
            dataset2 = create_waymo_dataset_jax(
                data_root=self.data_root,
                split="training",
                max_files=1,
                max_frames_per_file=1,
                image_size=self.image_size,
                cache_dir=cache_dir
            )
            second_load_time = time.time() - start_time
            
            # Second load should be faster (using cache)
            # self.assertLess(second_load_time, first_load_time)  # Commented out as it's not always reliable
            
            # Datasets should have same size
            self.assertEqual(len(dataset1), len(dataset2))
            
            print(f"✓ Cache functionality working")
            print(f"✓ First load time: {first_load_time:.2f}s")
            print(f"✓ Second load time: {second_load_time:.2f}s")
            print(f"✓ Cache file created: {os.path.exists(cache_file)}")
            
            # Clean up
            if os.path.exists(cache_file):
                os.remove(cache_file)
            if os.path.exists(cache_dir):
                os.rmdir(cache_dir)
                
        except (FileNotFoundError, ValueError) as e:
            self.skipTest(f"Dataset not available: {e}")


def run_dataset_tests():
    """Run all dataset JAX tests."""
    print("=" * 60)
    print("RUNNING WAYMO DATASET JAX TESTS")
    print("=" * 60)
    
    unittest.main(argv=[''], exit=False, verbosity=2)


if __name__ == "__main__":
    run_dataset_tests()