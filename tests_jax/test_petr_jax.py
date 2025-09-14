#!/usr/bin/env python3

import os
import sys
import unittest
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src_jax.models.petr_jax import create_petr_model, PositionalEncoding3D, MultiHeadAttention, PETRTransformer, PETRHead
from src_jax.utils.losses_jax import create_petr_loss, hungarian_matcher_jax
from src_jax.utils.metrics_jax import compute_map_3d_jax


class TestPETRJAX(unittest.TestCase):
    """Test suite for PETR JAX implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.key = jrandom.PRNGKey(42)
        self.batch_size = 2
        self.num_cameras = 5
        self.image_height = 200
        self.image_width = 400
        self.embed_dims = 128
        self.num_queries = 25
        self.num_classes = 3
        
    def test_positional_encoding_3d(self):
        """Test 3D positional encoding module."""
        print("Testing 3D Positional Encoding...")
        
        pos_enc = PositionalEncoding3D(num_pos_feats=64)
        
        # Test input
        positions = jnp.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]])
        
        # Initialize and apply
        key, subkey = jrandom.split(self.key)
        params = pos_enc.init(subkey, positions)
        output = pos_enc.apply(params, positions)
        
        # Check output shape
        expected_shape = (3, 64 * 3)  # num_pos_feats * 3 for x,y,z
        self.assertEqual(output.shape, expected_shape)
        
        # Check output is finite
        self.assertTrue(jnp.all(jnp.isfinite(output)))
        
        print(f"✓ Positional encoding output shape: {output.shape}")
        
    def test_multi_head_attention(self):
        """Test multi-head attention module."""
        print("Testing Multi-Head Attention...")
        
        attention = MultiHeadAttention(
            embed_dims=self.embed_dims,
            num_heads=8,
            dropout_rate=0.1
        )
        
        # Test inputs
        key, subkey1, subkey2, subkey3 = jrandom.split(self.key, 4)
        query = jrandom.normal(subkey1, (self.batch_size, 10, self.embed_dims))
        key_tensor = jrandom.normal(subkey2, (self.batch_size, 20, self.embed_dims))
        value = jrandom.normal(subkey3, (self.batch_size, 20, self.embed_dims))
        
        # Initialize and apply
        params = attention.init(self.key, query, key_tensor, value, training=False)
        output = attention.apply(params, query, key_tensor, value, training=False)
        
        # Check output shape
        self.assertEqual(output.shape, query.shape)
        
        # Check output is finite
        self.assertTrue(jnp.all(jnp.isfinite(output)))
        
        print(f"✓ Multi-head attention output shape: {output.shape}")
        
    def test_petr_transformer(self):
        """Test PETR transformer module."""
        print("Testing PETR Transformer...")
        
        transformer = PETRTransformer(
            embed_dims=self.embed_dims,
            num_heads=4,
            num_layers=2,
            ffn_dims=256,
            dropout_rate=0.1
        )
        
        # Test inputs
        key, subkey1, subkey2, subkey3 = jrandom.split(self.key, 4)
        query = jrandom.normal(subkey1, (self.batch_size, self.num_queries, self.embed_dims))
        key_tensor = jrandom.normal(subkey2, (self.batch_size, 100, self.embed_dims))
        value = jrandom.normal(subkey3, (self.batch_size, 100, self.embed_dims))
        
        # Initialize and apply
        params = transformer.init(self.key, query, key_tensor, value, training=False)
        output = transformer.apply(params, query, key_tensor, value, training=False)
        
        # Check output shape: [num_layers, batch_size, num_queries, embed_dims]
        expected_shape = (2, self.batch_size, self.num_queries, self.embed_dims)
        self.assertEqual(output.shape, expected_shape)
        
        # Check output is finite
        self.assertTrue(jnp.all(jnp.isfinite(output)))
        
        print(f"✓ PETR transformer output shape: {output.shape}")
        
    def test_petr_head(self):
        """Test PETR detection head."""
        print("Testing PETR Head...")
        
        head = PETRHead(
            embed_dims=self.embed_dims,
            num_classes=self.num_classes,
            num_reg_fcs=2
        )
        
        # Test input
        features = jrandom.normal(self.key, (self.batch_size, self.num_queries, self.embed_dims))
        
        # Initialize and apply
        key, subkey = jrandom.split(self.key)
        params = head.init(subkey, features)
        cls_logits, bbox_preds = head.apply(params, features)
        
        # Check output shapes
        expected_cls_shape = (self.batch_size, self.num_queries, self.num_classes + 1)
        expected_bbox_shape = (self.batch_size, self.num_queries, 7)
        
        self.assertEqual(cls_logits.shape, expected_cls_shape)
        self.assertEqual(bbox_preds.shape, expected_bbox_shape)
        
        # Check outputs are finite
        self.assertTrue(jnp.all(jnp.isfinite(cls_logits)))
        self.assertTrue(jnp.all(jnp.isfinite(bbox_preds)))
        
        print(f"✓ PETR head cls_logits shape: {cls_logits.shape}")
        print(f"✓ PETR head bbox_preds shape: {bbox_preds.shape}")
        
    def test_petr_model(self):
        """Test complete PETR model."""
        print("Testing Complete PETR Model...")
        
        model = create_petr_model(
            embed_dims=self.embed_dims,
            num_queries=self.num_queries,
            num_classes=self.num_classes,
            num_cameras=self.num_cameras
        )
        
        # Test input (JAX uses NHWC format)
        images = jrandom.normal(
            self.key, 
            (self.batch_size, self.num_cameras, self.image_height, self.image_width, 3)
        )
        
        # Initialize model
        key, subkey = jrandom.split(self.key)
        params = model.init(subkey, images, training=False)
        
        # Forward pass
        outputs = model.apply(params, images, training=False)
        
        # Check output shapes
        expected_cls_shape = (self.batch_size, self.num_queries, self.num_classes + 1)
        expected_bbox_shape = (self.batch_size, self.num_queries, 7)
        expected_ref_shape = (self.batch_size, self.num_queries, 3)
        
        self.assertEqual(outputs['cls_logits'].shape, expected_cls_shape)
        self.assertEqual(outputs['bbox_preds'].shape, expected_bbox_shape)
        self.assertEqual(outputs['reference_points'].shape, expected_ref_shape)
        
        # Check outputs are finite
        self.assertTrue(jnp.all(jnp.isfinite(outputs['cls_logits'])))
        self.assertTrue(jnp.all(jnp.isfinite(outputs['bbox_preds'])))
        self.assertTrue(jnp.all(jnp.isfinite(outputs['reference_points'])))
        
        # Count parameters
        param_count = sum(x.size for x in jax.tree.leaves(params))
        
        print(f"✓ PETR model forward pass successful")
        print(f"✓ Model parameters: {param_count:,}")
        print(f"✓ Output shapes - cls: {outputs['cls_logits'].shape}, "
              f"bbox: {outputs['bbox_preds'].shape}, ref: {outputs['reference_points'].shape}")
        
    def test_petr_loss(self):
        """Test PETR loss function."""
        print("Testing PETR Loss Function...")
        
        # Create fake outputs
        key, subkey1, subkey2 = jrandom.split(self.key, 3)
        outputs = {
            'cls_logits': jrandom.normal(subkey1, (self.batch_size, self.num_queries, self.num_classes + 1)),
            'bbox_preds': jrandom.normal(subkey2, (self.batch_size, self.num_queries, 7))
        }
        
        # Create fake targets
        targets = {
            'labels': jnp.array([[0, 1, 2, 3, 3], [1, 0, 3, 3, 3]]),  # 3 = background
            'boxes_3d': jrandom.normal(self.key, (self.batch_size, 5, 7))
        }
        
        # Test loss computation
        loss_fn = create_petr_loss(num_classes=self.num_classes)
        losses = loss_fn(outputs, targets)
        
        # Check loss components
        self.assertTrue(jnp.isfinite(losses['total_loss']))
        self.assertTrue(jnp.isfinite(losses['cls_loss']))
        self.assertTrue(jnp.isfinite(losses['bbox_loss']))
        self.assertTrue(jnp.isfinite(losses['giou_loss']))
        
        # Check losses are positive
        self.assertTrue(losses['total_loss'] >= 0)
        self.assertTrue(losses['cls_loss'] >= 0)
        self.assertTrue(losses['bbox_loss'] >= 0)
        self.assertTrue(losses['giou_loss'] >= 0)
        
        print(f"✓ Loss computation successful")
        print(f"✓ Total loss: {losses['total_loss']:.4f}")
        print(f"✓ Individual losses - cls: {losses['cls_loss']:.4f}, "
              f"bbox: {losses['bbox_loss']:.4f}, giou: {losses['giou_loss']:.4f}")
        
    def test_petr_metrics(self):
        """Test PETR metrics computation."""
        print("Testing PETR Metrics...")
        
        # Create fake predictions with realistic scores
        key, subkey1, subkey2 = jrandom.split(self.key, 3)
        
        # Create logits that favor certain classes
        cls_logits = jrandom.normal(subkey1, (self.batch_size, self.num_queries, self.num_classes + 1))
        cls_logits = cls_logits.at[:, :5, :self.num_classes].add(2.0)  # Boost first few predictions
        
        predictions = {
            'cls_logits': cls_logits,
            'bbox_preds': jrandom.normal(subkey2, (self.batch_size, self.num_queries, 7)) * 5
        }
        
        # Create fake targets
        targets = {
            'labels': jnp.array([[0, 1, 2, 3, 3], [1, 0, 3, 3, 3]]),  # 3 = background
            'boxes_3d': jrandom.normal(self.key, (self.batch_size, 5, 7)) * 3
        }
        
        # Test metrics computation
        metrics = compute_map_3d_jax(predictions, targets, num_classes=self.num_classes)
        
        # Check metrics
        self.assertTrue(0 <= metrics['mAP'] <= 1)
        self.assertTrue(0 <= metrics['AP_vehicle'] <= 1)
        self.assertTrue(0 <= metrics['AP_pedestrian'] <= 1)
        self.assertTrue(0 <= metrics['AP_cyclist'] <= 1)
        
        print(f"✓ Metrics computation successful")
        print(f"✓ mAP: {metrics['mAP']:.4f}")
        print(f"✓ AP by class - vehicle: {metrics['AP_vehicle']:.4f}, "
              f"pedestrian: {metrics['AP_pedestrian']:.4f}, cyclist: {metrics['AP_cyclist']:.4f}")
        print(f"✓ Predictions: {metrics['num_predictions']}, GT: {metrics['num_gt_total']}")
        
    def test_gradients(self):
        """Test gradient computation through the model."""
        print("Testing Gradient Computation...")
        
        # Create model and loss
        model = create_petr_model(
            embed_dims=64,  # Smaller for testing
            num_queries=10,
            num_classes=self.num_classes,
            num_cameras=self.num_cameras
        )
        loss_fn = create_petr_loss(num_classes=self.num_classes)
        
        # Test input
        images = jrandom.normal(
            self.key, 
            (1, self.num_cameras, 100, 200, 3)  # Smaller images
        )
        
        # Targets
        targets = {
            'labels': jnp.array([[0, 1, 3, 3, 3]]),
            'boxes_3d': jrandom.normal(self.key, (1, 5, 7))
        }
        
        # Initialize model
        key, subkey = jrandom.split(self.key)
        params = model.init(subkey, images, training=True)
        
        # Define loss function
        def compute_loss(params):
            outputs = model.apply(params, images, training=True)
            loss_dict = loss_fn(outputs, targets)
            return loss_dict['total_loss']
        
        # Compute gradients
        loss, grads = jax.value_and_grad(compute_loss)(params)
        
        # Check gradients exist and are finite
        self.assertTrue(jnp.isfinite(loss))
        
        # Check all gradients are finite
        def check_finite(x):
            if isinstance(x, jnp.ndarray):
                return jnp.all(jnp.isfinite(x))
            return True
        
        all_finite = jax.tree_util.tree_all(jax.tree_map(check_finite, grads))
        self.assertTrue(all_finite)
        
        print(f"✓ Gradient computation successful")
        print(f"✓ Loss: {loss:.4f}")
        print(f"✓ All gradients finite: {all_finite}")


def run_petr_tests():
    """Run all PETR JAX tests."""
    print("=" * 60)
    print("RUNNING PETR JAX TESTS")
    print("=" * 60)
    
    unittest.main(argv=[''], exit=False, verbosity=2)


if __name__ == "__main__":
    run_petr_tests()