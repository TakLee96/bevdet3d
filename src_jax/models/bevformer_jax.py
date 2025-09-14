import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from functools import partial
from .petr_jax import ResNet50Backbone  # Reuse backbone


class PositionalEncodingLearned2D(nn.Module):
    """2D learned positional encoding for BEV queries in JAX/Flax."""
    
    embed_dims: int = 256
    h: int = 50
    w: int = 50
    
    @nn.compact
    def __call__(self, x):
        """
        Args:
            x: [B, H*W, D] or [B, H, W, D]
        Returns:
            pos: [B, H*W, D] positional encoding
        """
        # Learned positional embeddings
        row_embed = nn.Embed(self.h, self.embed_dims // 2, name='row_embed')
        col_embed = nn.Embed(self.w, self.embed_dims // 2, name='col_embed')
        
        # Create position indices
        i = jnp.arange(self.w)
        j = jnp.arange(self.h)
        
        # Get embeddings
        x_emb = col_embed(i)  # [W, D//2]
        y_emb = row_embed(j)  # [H, D//2]
        
        # Create 2D positional encoding
        x_emb_expanded = jnp.repeat(x_emb[None, :, :], self.h, axis=0)  # [H, W, D//2]
        y_emb_expanded = jnp.repeat(y_emb[:, None, :], self.w, axis=1)  # [H, W, D//2]
        
        pos = jnp.concatenate([x_emb_expanded, y_emb_expanded], axis=-1)  # [H, W, D]
        pos = pos.reshape(-1, self.embed_dims)  # [H*W, D]
        
        # Expand for batch
        batch_size = x.shape[0]
        pos = jnp.broadcast_to(pos[None, ...], (batch_size, self.h * self.w, self.embed_dims))
        
        return pos


class SpatialCrossAttention(nn.Module):
    """Spatial cross-attention for multi-camera feature aggregation in JAX/Flax."""
    
    embed_dims: int = 256
    num_heads: int = 8
    num_levels: int = 4
    num_points: int = 4
    
    @nn.compact
    def __call__(self, query, reference_points, value, spatial_shapes, training=True):
        """
        Args:
            query: [B, H*W, D] BEV queries
            reference_points: [B, H*W, 2] normalized 2D reference points
            value: [B, N_cams*H'*W', D] multi-camera features
            spatial_shapes: List of (H', W') for each level
        """
        batch_size, num_queries, _ = query.shape
        num_value = value.shape[1]
        
        # Get attention weights and sampling offsets
        attention_weights = nn.Dense(
            self.num_heads * self.num_levels * self.num_points,
            name='attention_weights'
        )(query)
        
        sampling_offsets = nn.Dense(
            self.num_heads * self.num_levels * self.num_points * 2,
            name='sampling_offsets'
        )(query)
        
        # Reshape
        attention_weights = attention_weights.reshape(
            batch_size, num_queries, self.num_heads, self.num_levels * self.num_points)
        sampling_offsets = sampling_offsets.reshape(
            batch_size, num_queries, self.num_heads, self.num_levels, self.num_points, 2)
        
        # Normalize attention weights
        attention_weights = nn.softmax(attention_weights, axis=-1)
        
        # Project values
        value_proj = nn.Dense(self.embed_dims, name='value_proj')(value)
        value_proj = value_proj.reshape(
            batch_size, num_value, self.num_heads, self.embed_dims // self.num_heads)
        
        # For simplicity, use average pooling instead of deformable attention
        # In a full implementation, this would do proper deformable sampling
        output = jnp.mean(value_proj, axis=1, keepdims=True)
        output = jnp.broadcast_to(output, (batch_size, num_queries, self.num_heads, self.embed_dims // self.num_heads))
        output = output.reshape(batch_size, num_queries, self.embed_dims)
        
        return nn.Dense(self.embed_dims, name='output_proj')(output)


class TemporalSelfAttention(nn.Module):
    """Temporal self-attention for BEV feature evolution in JAX/Flax."""
    
    embed_dims: int = 256
    num_heads: int = 8
    
    @nn.compact
    def __call__(self, query, prev_bev=None, bev_pos=None, training=True):
        """
        Args:
            query: [B, H*W, D] current BEV queries
            prev_bev: [B, H*W, D] previous BEV features (optional)
            bev_pos: [B, H*W, D] BEV positional encoding
        """
        if prev_bev is not None:
            # Concatenate current and previous BEV features
            key_value = jnp.concatenate([query, prev_bev], axis=1)  # [B, 2*H*W, D]
        else:
            key_value = query
            
        # Add positional encoding
        if bev_pos is not None:
            query = query + bev_pos
            if prev_bev is not None:
                key_value = key_value + jnp.concatenate([bev_pos, bev_pos], axis=1)
            else:
                key_value = key_value + bev_pos
        
        # Self-attention using Flax's MultiHeadDotProductAttention
        attn = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.embed_dims,
            name='self_attn'
        )
        
        output = attn(query, key_value, key_value)
        
        return nn.LayerNorm()(output + query)


class BEVFormerLayer(nn.Module):
    """Single BEVFormer encoder layer in JAX/Flax."""
    
    embed_dims: int = 256
    num_heads: int = 8
    ffn_dims: int = 1024
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, bev_query, reference_points, img_feats, spatial_shapes, 
                 prev_bev=None, bev_pos=None, training=True):
        """
        Args:
            bev_query: [B, H*W, D] BEV queries
            reference_points: [B, H*W, 2] 2D reference points
            img_feats: [B, N_cams*H'*W', D] image features
            spatial_shapes: Spatial shapes of image features
            prev_bev: Previous BEV features (optional)
            bev_pos: BEV positional encoding
        """
        # Temporal self-attention
        temporal_attn = TemporalSelfAttention(
            embed_dims=self.embed_dims,
            num_heads=self.num_heads,
            name='temporal_self_attn'
        )
        bev_query = temporal_attn(
            nn.LayerNorm()(bev_query), prev_bev, bev_pos, training)
        
        # Spatial cross-attention
        spatial_attn = SpatialCrossAttention(
            embed_dims=self.embed_dims,
            num_heads=self.num_heads,
            name='spatial_cross_attn'
        )
        spatial_out = spatial_attn(
            nn.LayerNorm()(bev_query), reference_points, img_feats, spatial_shapes, training)
        bev_query = bev_query + spatial_out
        
        # Feed-forward
        ffn_out = nn.LayerNorm()(bev_query)
        ffn_out = nn.Dense(self.ffn_dims)(ffn_out)
        ffn_out = nn.relu(ffn_out)
        ffn_out = nn.Dropout(self.dropout_rate)(ffn_out, deterministic=not training)
        ffn_out = nn.Dense(self.embed_dims)(ffn_out)
        ffn_out = nn.Dropout(self.dropout_rate)(ffn_out, deterministic=not training)
        bev_query = bev_query + ffn_out
        
        return bev_query


class BEVFormerEncoder(nn.Module):
    """BEVFormer encoder with multiple layers in JAX/Flax."""
    
    embed_dims: int = 256
    num_layers: int = 6
    num_heads: int = 8
    ffn_dims: int = 1024
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, bev_query, reference_points, img_feats, spatial_shapes, 
                 prev_bev=None, bev_pos=None, training=True):
        """Forward through all encoder layers."""
        output = bev_query
        
        for i in range(self.num_layers):
            layer = BEVFormerLayer(
                embed_dims=self.embed_dims,
                num_heads=self.num_heads,
                ffn_dims=self.ffn_dims,
                dropout_rate=self.dropout_rate,
                name=f'layer_{i}'
            )
            output = layer(output, reference_points, img_feats, spatial_shapes, 
                          prev_bev, bev_pos, training)
            
        return output


class BEVFormerModel(nn.Module):
    """
    BEVFormer: Learning Bird's-Eye-View Representation from Multi-Camera Images.
    
    JAX/Flax implementation without PyTorch dependencies.
    """
    
    embed_dims: int = 256
    num_queries: int = 100
    num_classes: int = 3
    bev_h: int = 50
    bev_w: int = 50
    num_cameras: int = 5
    num_encoder_layers: int = 6
    pc_range: Tuple = (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0)
    
    def setup(self):
        # Image backbone
        self.backbone = ResNet50Backbone()
        
        # Feature projection
        self.input_proj = nn.Conv(self.embed_dims, kernel_size=(1, 1))
        
        # BEV positional encoding
        self.bev_pos_encoding = PositionalEncodingLearned2D(
            embed_dims=self.embed_dims, h=self.bev_h, w=self.bev_w)
        
        # BEVFormer encoder
        self.encoder = BEVFormerEncoder(
            embed_dims=self.embed_dims,
            num_layers=self.num_encoder_layers
        )
        
        # Detection heads
        self.cls_head = nn.Sequential([
            nn.Dense(self.embed_dims),
            nn.relu,
            nn.Dense(self.num_classes + 1)
        ])
        
        self.reg_head = nn.Sequential([
            nn.Dense(self.embed_dims),
            nn.relu,
            nn.Dense(7)  # 3D bbox: x, y, z, l, w, h, rot
        ])
        
        # Cross-attention for detection (BEV features -> object queries)
        self.detection_cross_attn = nn.MultiHeadDotProductAttention(
            num_heads=8,
            qkv_features=self.embed_dims
        )
    
    def get_reference_points(self, batch_size):
        """Generate 2D reference points for BEV queries."""
        # Create grid of reference points in BEV space
        y_coords = jnp.linspace(0, 1, self.bev_h)
        x_coords = jnp.linspace(0, 1, self.bev_w)
        
        y_grid, x_grid = jnp.meshgrid(y_coords, x_coords, indexing='ij')
        
        # Stack and reshape to [H*W, 2]
        ref_points = jnp.stack([x_grid, y_grid], axis=-1).reshape(-1, 2)
        
        # Expand for batch
        ref_points = jnp.broadcast_to(ref_points[None, ...], (batch_size, self.bev_h * self.bev_w, 2))
        
        return ref_points
        
    def extract_img_features(self, images, training=True):
        """Extract features from multi-camera images."""
        batch_size, num_cams, H, W, C = images.shape
        
        # Flatten cameras and batch for backbone
        images_flat = images.reshape(batch_size * num_cams, H, W, C)
        
        # Extract features
        feats = self.backbone(images_flat, training)  # [B*N_cams, H', W', backbone_dims]
        feats = self.input_proj(feats)  # [B*N_cams, H', W', embed_dims]
        
        _, feat_h, feat_w, embed_dims = feats.shape
        
        # Reshape and flatten spatial dimensions
        feats = feats.reshape(batch_size, num_cams, feat_h, feat_w, embed_dims)
        feats = feats.reshape(batch_size, num_cams * feat_h * feat_w, embed_dims)
        
        spatial_shapes = [(feat_h, feat_w)] * num_cams
        
        return feats, spatial_shapes
        
    def __call__(self, images, prev_bev=None, training=True):
        """
        Args:
            images: [B, N_cams, H, W, 3] multi-camera images
            prev_bev: Previous BEV features for temporal modeling
        """
        batch_size = images.shape[0]
        
        # Extract image features
        img_feats, spatial_shapes = self.extract_img_features(images, training)
        
        # Get BEV queries and positional encoding
        bev_queries = self.param('bev_queries', 
                                lambda rng: jax.random.normal(rng, (self.bev_h * self.bev_w, self.embed_dims)))
        bev_queries = jnp.broadcast_to(bev_queries[None, ...], (batch_size, self.bev_h * self.bev_w, self.embed_dims))
        
        bev_pos = self.bev_pos_encoding(bev_queries)  # [B, H*W, D]
        
        # Get reference points
        reference_points = self.get_reference_points(batch_size)  # [B, H*W, 2]
        
        # BEVFormer encoder
        bev_features = self.encoder(
            bev_query=bev_queries,
            reference_points=reference_points,
            img_feats=img_feats,
            spatial_shapes=spatial_shapes,
            prev_bev=prev_bev,
            bev_pos=bev_pos,
            training=training
        )  # [B, H*W, D]
        
        # Object detection from BEV features
        object_queries = self.param('object_queries',
                                   lambda rng: jax.random.normal(rng, (self.num_queries, self.embed_dims)))
        object_queries = jnp.broadcast_to(object_queries[None, ...], (batch_size, self.num_queries, self.embed_dims))
        
        # Cross-attention: object queries attend to BEV features
        detection_features = self.detection_cross_attn(
            object_queries, bev_features, bev_features)  # [B, num_queries, D]
        
        # Generate predictions
        cls_logits = self.cls_head(detection_features)  # [B, num_queries, num_classes+1]
        bbox_preds = self.reg_head(detection_features)  # [B, num_queries, 7]
        
        # Reference points for 3D detection (simplified)
        reference_points_3d = jnp.zeros((batch_size, self.num_queries, 3))
        
        return {
            'cls_logits': cls_logits,
            'bbox_preds': bbox_preds,
            'reference_points': reference_points_3d,
            'bev_features': bev_features  # For temporal modeling
        }


def create_bevformer_model(
    embed_dims=256,
    num_queries=100,
    num_classes=3,
    bev_h=50,
    bev_w=50,
    num_cameras=5,
    num_encoder_layers=6,
    **kwargs
):
    """Factory function to create BEVFormer model."""
    return BEVFormerModel(
        embed_dims=embed_dims,
        num_queries=num_queries,
        num_classes=num_classes,
        bev_h=bev_h,
        bev_w=bev_w,
        num_cameras=num_cameras,
        num_encoder_layers=num_encoder_layers,
        **kwargs
    )


if __name__ == "__main__":
    # Test the JAX BEVFormer model
    import jax.random as jrandom
    
    model = create_bevformer_model(
        embed_dims=128,  # Smaller for testing
        num_queries=25,
        num_classes=3,
        bev_h=15,       # Smaller BEV grid
        bev_w=15,
        num_cameras=5,
        num_encoder_layers=3  # Fewer layers for testing
    )
    
    # Test input (JAX uses NHWC format)
    batch_size = 1
    key = jrandom.PRNGKey(0)
    images = jrandom.normal(key, (batch_size, 5, 200, 400, 3))
    
    # Initialize model
    key, subkey = jrandom.split(key)
    params = model.init(subkey, images, training=False)
    
    # Forward pass
    outputs = model.apply(params, images, training=False)
    
    print("JAX BEVFormer model test successful!")
    print(f"Class logits shape: {outputs['cls_logits'].shape}")
    print(f"Bbox predictions shape: {outputs['bbox_preds'].shape}")
    print(f"Reference points shape: {outputs['reference_points'].shape}")
    print(f"BEV features shape: {outputs['bev_features'].shape}")
    
    # Count parameters
    param_count = sum(x.size for x in jax.tree_leaves(params))
    print(f"Total parameters: {param_count:,}")
    
    # Test temporal modeling
    print("\nTesting temporal modeling...")
    prev_bev = outputs['bev_features']
    outputs2 = model.apply(params, images, prev_bev=prev_bev, training=False)
    print(f"Temporal BEV features shape: {outputs2['bev_features'].shape}")
    print("✓ Temporal modeling test passed")