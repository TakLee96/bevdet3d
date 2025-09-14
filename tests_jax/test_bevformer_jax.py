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

from src_jax.models.bevformer_jax import (
    create_bevformer_model, PositionalEncodingLearned2D, 
    SpatialCrossAttention, TemporalSelfAttention, BEVFormerLayer, BEVFormerEncoder
)


class TestBEVFormerJAX(unittest.TestCase):
    """Test suite for BEVFormer JAX implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.key = jrandom.PRNGKey(42)
        self.batch_size = 1  # BEVFormer is memory intensive
        self.num_cameras = 5
        self.image_height = 200
        self.image_width = 400
        self.embed_dims = 64  # Smaller for testing
        self.num_queries = 10
        self.num_classes = 3
        self.bev_h = 8
        self.bev_w = 8
        
    def test_positional_encoding_learned_2d(self):
        """Test 2D learned positional encoding."""
        print("Testing 2D Learned Positional Encoding...")
        
        pos_enc = PositionalEncodingLearned2D(
            embed_dims=self.embed_dims,
            h=self.bev_h,
            w=self.bev_w
        )
        
        # Test input
        x = jrandom.normal(self.key, (self.batch_size, self.bev_h * self.bev_w, self.embed_dims))
        
        # Initialize and apply
        key, subkey = jrandom.split(self.key)
        params = pos_enc.init(subkey, x)
        output = pos_enc.apply(params, x)
        
        # Check output shape
        expected_shape = (self.batch_size, self.bev_h * self.bev_w, self.embed_dims)
        self.assertEqual(output.shape, expected_shape)
        
        # Check output is finite
        self.assertTrue(jnp.all(jnp.isfinite(output)))
        
        print(f"✓ 2D positional encoding output shape: {output.shape}")
        
    def test_spatial_cross_attention(self):
        """Test spatial cross-attention module."""
        print("Testing Spatial Cross-Attention...")
        
        spatial_attn = SpatialCrossAttention(
            embed_dims=self.embed_dims,
            num_heads=4,
            num_levels=2,
            num_points=2
        )
        
        # Test inputs
        key, subkey1, subkey2, subkey3 = jrandom.split(self.key, 4)
        bev_query = jrandom.normal(subkey1, (self.batch_size, self.bev_h * self.bev_w, self.embed_dims))
        reference_points = jrandom.uniform(subkey2, (self.batch_size, self.bev_h * self.bev_w, 2))
        img_feats = jrandom.normal(subkey3, (self.batch_size, 100, self.embed_dims))
        spatial_shapes = [(10, 10), (5, 5)]
        
        # Initialize and apply
        params = spatial_attn.init(self.key, bev_query, reference_points, img_feats, spatial_shapes, training=False)
        output = spatial_attn.apply(params, bev_query, reference_points, img_feats, spatial_shapes, training=False)
        
        # Check output shape
        self.assertEqual(output.shape, bev_query.shape)
        
        # Check output is finite
        self.assertTrue(jnp.all(jnp.isfinite(output)))
        
        print(f"✓ Spatial cross-attention output shape: {output.shape}")
        
    def test_temporal_self_attention(self):
        """Test temporal self-attention module."""
        print("Testing Temporal Self-Attention...")
        
        temporal_attn = TemporalSelfAttention(
            embed_dims=self.embed_dims,
            num_heads=4
        )
        
        # Test inputs
        key, subkey1, subkey2, subkey3 = jrandom.split(self.key, 4)
        current_bev = jrandom.normal(subkey1, (self.batch_size, self.bev_h * self.bev_w, self.embed_dims))
        prev_bev = jrandom.normal(subkey2, (self.batch_size, self.bev_h * self.bev_w, self.embed_dims))
        bev_pos = jrandom.normal(subkey3, (self.batch_size, self.bev_h * self.bev_w, self.embed_dims))
        
        # Initialize and apply
        params = temporal_attn.init(self.key, current_bev, prev_bev, bev_pos, training=False)
        output = temporal_attn.apply(params, current_bev, prev_bev, bev_pos, training=False)
        
        # Check output shape
        self.assertEqual(output.shape, current_bev.shape)
        
        # Check output is finite
        self.assertTrue(jnp.all(jnp.isfinite(output)))
        
        print(f"✓ Temporal self-attention output shape: {output.shape}")
        
        # Test without previous BEV
        output_no_prev = temporal_attn.apply(params, current_bev, None, bev_pos, training=False)
        self.assertEqual(output_no_prev.shape, current_bev.shape)
        self.assertTrue(jnp.all(jnp.isfinite(output_no_prev)))
        
        print(f"✓ Temporal self-attention without prev_bev: {output_no_prev.shape}")
        
    def test_bevformer_layer(self):
        """Test single BEVFormer encoder layer."""
        print("Testing BEVFormer Layer...")
        
        layer = BEVFormerLayer(
            embed_dims=self.embed_dims,
            num_heads=4,
            ffn_dims=128,
            dropout_rate=0.1
        )
        
        # Test inputs
        key, subkey1, subkey2, subkey3, subkey4 = jrandom.split(self.key, 5)
        bev_query = jrandom.normal(subkey1, (self.batch_size, self.bev_h * self.bev_w, self.embed_dims))
        reference_points = jrandom.uniform(subkey2, (self.batch_size, self.bev_h * self.bev_w, 2))
        img_feats = jrandom.normal(subkey3, (self.batch_size, 100, self.embed_dims))
        bev_pos = jrandom.normal(subkey4, (self.batch_size, self.bev_h * self.bev_w, self.embed_dims))
        spatial_shapes = [(10, 10)]
        
        # Initialize and apply
        params = layer.init(
            self.key, bev_query, reference_points, img_feats, spatial_shapes,
            prev_bev=None, bev_pos=bev_pos, training=False
        )
        output = layer.apply(
            params, bev_query, reference_points, img_feats, spatial_shapes,
            prev_bev=None, bev_pos=bev_pos, training=False
        )
        
        # Check output shape
        self.assertEqual(output.shape, bev_query.shape)
        
        # Check output is finite
        self.assertTrue(jnp.all(jnp.isfinite(output)))
        
        print(f"✓ BEVFormer layer output shape: {output.shape}")
        
    def test_bevformer_encoder(self):
        """Test BEVFormer encoder with multiple layers."""
        print("Testing BEVFormer Encoder...")
        
        encoder = BEVFormerEncoder(
            embed_dims=self.embed_dims,
            num_layers=2,
            num_heads=4,
            ffn_dims=128,
            dropout_rate=0.1
        )
        
        # Test inputs
        key, subkey1, subkey2, subkey3, subkey4 = jrandom.split(self.key, 5)
        bev_query = jrandom.normal(subkey1, (self.batch_size, self.bev_h * self.bev_w, self.embed_dims))
        reference_points = jrandom.uniform(subkey2, (self.batch_size, self.bev_h * self.bev_w, 2))
        img_feats = jrandom.normal(subkey3, (self.batch_size, 100, self.embed_dims))
        bev_pos = jrandom.normal(subkey4, (self.batch_size, self.bev_h * self.bev_w, self.embed_dims))
        spatial_shapes = [(10, 10)]
        
        # Initialize and apply
        params = encoder.init(
            self.key, bev_query, reference_points, img_feats, spatial_shapes,
            prev_bev=None, bev_pos=bev_pos, training=False
        )
        output = encoder.apply(
            params, bev_query, reference_points, img_feats, spatial_shapes,
            prev_bev=None, bev_pos=bev_pos, training=False
        )
        
        # Check output shape
        self.assertEqual(output.shape, bev_query.shape)
        
        # Check output is finite
        self.assertTrue(jnp.all(jnp.isfinite(output)))
        
        print(f"✓ BEVFormer encoder output shape: {output.shape}")
        
    def test_bevformer_model(self):
        """Test complete BEVFormer model."""
        print("Testing Complete BEVFormer Model...")
        
        model = create_bevformer_model(
            embed_dims=self.embed_dims,
            num_queries=self.num_queries,
            num_classes=self.num_classes,
            bev_h=self.bev_h,
            bev_w=self.bev_w,
            num_cameras=self.num_cameras,
            num_encoder_layers=2
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
        expected_bev_shape = (self.batch_size, self.bev_h * self.bev_w, self.embed_dims)
        
        self.assertEqual(outputs['cls_logits'].shape, expected_cls_shape)
        self.assertEqual(outputs['bbox_preds'].shape, expected_bbox_shape)
        self.assertEqual(outputs['reference_points'].shape, expected_ref_shape)
        self.assertEqual(outputs['bev_features'].shape, expected_bev_shape)
        
        # Check outputs are finite
        self.assertTrue(jnp.all(jnp.isfinite(outputs['cls_logits'])))
        self.assertTrue(jnp.all(jnp.isfinite(outputs['bbox_preds'])))
        self.assertTrue(jnp.all(jnp.isfinite(outputs['reference_points'])))
        self.assertTrue(jnp.all(jnp.isfinite(outputs['bev_features'])))
        
        # Count parameters
        param_count = sum(x.size for x in jax.tree_leaves(params))
        
        print(f"✓ BEVFormer model forward pass successful")
        print(f"✓ Model parameters: {param_count:,}")
        print(f"✓ Output shapes - cls: {outputs['cls_logits'].shape}, "
              f"bbox: {outputs['bbox_preds'].shape}, bev: {outputs['bev_features'].shape}")
        
    def test_temporal_modeling(self):
        """Test temporal modeling capability."""
        print("Testing Temporal Modeling...")
        
        model = create_bevformer_model(
            embed_dims=self.embed_dims,
            num_queries=self.num_queries,
            num_classes=self.num_classes,
            bev_h=self.bev_h,
            bev_w=self.bev_w,
            num_cameras=self.num_cameras,
            num_encoder_layers=2
        )
        
        # Test inputs
        images1 = jrandom.normal(
            self.key, 
            (self.batch_size, self.num_cameras, self.image_height, self.image_width, 3)
        )
        key2 = jrandom.split(self.key)[1]
        images2 = jrandom.normal(
            key2, 
            (self.batch_size, self.num_cameras, self.image_height, self.image_width, 3)
        )
        
        # Initialize model
        key, subkey = jrandom.split(self.key)
        params = model.init(subkey, images1, training=False)
        
        # First forward pass (no previous BEV)
        outputs1 = model.apply(params, images1, prev_bev=None, training=False)
        
        # Second forward pass (with previous BEV)
        outputs2 = model.apply(params, images2, prev_bev=outputs1['bev_features'], training=False)
        
        # Check that temporal modeling affects outputs
        bev_diff = jnp.mean(jnp.abs(outputs2['bev_features'] - outputs1['bev_features']))
        
        # BEV features should be different when using temporal modeling
        self.assertTrue(bev_diff > 0)
        
        # Check outputs are finite
        self.assertTrue(jnp.all(jnp.isfinite(outputs2['cls_logits'])))
        self.assertTrue(jnp.all(jnp.isfinite(outputs2['bbox_preds'])))
        self.assertTrue(jnp.all(jnp.isfinite(outputs2['bev_features'])))
        
        print(f"✓ Temporal modeling successful")
        print(f"✓ BEV feature difference: {bev_diff:.6f}")
        print(f"✓ Second pass outputs finite: True")
        
    def test_gradients(self):
        """Test gradient computation through the model."""
        print("Testing Gradient Computation...")
        
        # Import loss function
        from src_jax.utils.losses_jax import create_petr_loss
        
        # Create model and loss
        model = create_bevformer_model(
            embed_dims=32,  # Very small for testing
            num_queries=5,
            num_classes=self.num_classes,
            bev_h=4,
            bev_w=4,
            num_cameras=self.num_cameras,
            num_encoder_layers=1
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


def run_bevformer_tests():
    """Run all BEVFormer JAX tests."""
    print("=" * 60)
    print("RUNNING BEVFORMER JAX TESTS")
    print("=" * 60)
    
    unittest.main(argv=[''], exit=False, verbosity=2)


if __name__ == "__main__":
    run_bevformer_tests()