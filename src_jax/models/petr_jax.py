import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from functools import partial


class PositionalEncoding3D(nn.Module):
    """3D positional encoding for PETR in JAX/Flax."""
    
    num_pos_feats: int = 128
    temperature: float = 10000.0
    
    @nn.compact
    def __call__(self, pos):
        """
        Args:
            pos: [N, 3] positions in 3D space
        Returns:
            pos_embed: [N, num_pos_feats*3] positional embeddings
        """
        scale = 2 * jnp.pi
        pos = pos * scale
        
        dim_t = jnp.arange(self.num_pos_feats, dtype=jnp.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        
        pos_x = pos[..., 0:1] / dim_t[None, :]
        pos_y = pos[..., 1:2] / dim_t[None, :]  
        pos_z = pos[..., 2:3] / dim_t[None, :]
        
        # Handle even/odd encoding
        def encode_dim(p):
            sin_part = jnp.sin(p[..., 0::2])
            cos_part = jnp.cos(p[..., 1::2])
            # Ensure same size by taking minimum
            min_size = min(sin_part.shape[-1], cos_part.shape[-1])
            sin_part = sin_part[..., :min_size]
            cos_part = cos_part[..., :min_size]
            return jnp.concatenate([sin_part, cos_part], axis=-1)
        
        pos_x = encode_dim(pos_x)
        pos_y = encode_dim(pos_y)
        pos_z = encode_dim(pos_z)
        
        pos_embed = jnp.concatenate([pos_y, pos_x, pos_z], axis=-1)
        return pos_embed


class MultiHeadAttention(nn.Module):
    """Multi-head attention module in JAX/Flax."""
    
    embed_dims: int = 256
    num_heads: int = 8
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, query, key, value, key_padding_mask=None, pos_embed=None, training=True):
        batch_size, seq_len = query.shape[:2]
        head_dims = self.embed_dims // self.num_heads
        
        # Project to Q, K, V
        q = nn.Dense(self.embed_dims, name='q_proj')(query)
        k = nn.Dense(self.embed_dims, name='k_proj')(key)
        v = nn.Dense(self.embed_dims, name='v_proj')(value)
        
        # Add positional encoding if provided
        if pos_embed is not None:
            k = k + pos_embed
            
        # Reshape for multi-head attention
        q = q.reshape(batch_size, seq_len, self.num_heads, head_dims).transpose(0, 2, 1, 3)
        k = k.reshape(batch_size, -1, self.num_heads, head_dims).transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, -1, self.num_heads, head_dims).transpose(0, 2, 1, 3)
        
        # Compute attention
        scale = head_dims ** -0.5
        attn_weights = jnp.matmul(q, k.transpose(0, 1, 3, 2)) * scale
        
        # Apply key padding mask
        if key_padding_mask is not None:
            attn_weights = jnp.where(
                key_padding_mask[:, None, None, :], 
                -jnp.inf, 
                attn_weights
            )
                
        attn_weights = nn.softmax(attn_weights, axis=-1)
        attn_weights = nn.Dropout(self.dropout_rate)(attn_weights, deterministic=not training)
        
        # Apply attention to values
        attn_output = jnp.matmul(attn_weights, v)
        
        # Concatenate heads
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(
            batch_size, seq_len, self.embed_dims)
            
        return nn.Dense(self.embed_dims, name='out_proj')(attn_output)


class TransformerLayer(nn.Module):
    """Transformer layer with self-attention and cross-attention."""
    
    embed_dims: int = 256
    num_heads: int = 8
    ffn_dims: int = 1024
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, query, key, value, query_pos=None, key_pos=None, 
                 key_padding_mask=None, training=True):
        # Self-attention
        q_norm = nn.LayerNorm()(query)
        q = MultiHeadAttention(
            embed_dims=self.embed_dims, 
            num_heads=self.num_heads, 
            dropout_rate=self.dropout_rate,
            name='self_attn'
        )(q_norm, q_norm, q_norm, pos_embed=query_pos, training=training)
        query = query + nn.Dropout(self.dropout_rate)(q, deterministic=not training)
        
        # Cross-attention
        q_norm = nn.LayerNorm()(query)
        q = MultiHeadAttention(
            embed_dims=self.embed_dims, 
            num_heads=self.num_heads, 
            dropout_rate=self.dropout_rate,
            name='cross_attn'
        )(q_norm, key, value, key_padding_mask=key_padding_mask, 
          pos_embed=key_pos, training=training)
        query = query + nn.Dropout(self.dropout_rate)(q, deterministic=not training)
        
        # Feed-forward
        q_norm = nn.LayerNorm()(query)
        q = nn.Dense(self.ffn_dims)(q_norm)
        q = nn.relu(q)
        q = nn.Dropout(self.dropout_rate)(q, deterministic=not training)
        q = nn.Dense(self.embed_dims)(q)
        q = nn.Dropout(self.dropout_rate)(q, deterministic=not training)
        query = query + q
        
        return query


class PETRTransformer(nn.Module):
    """PETR Transformer decoder in JAX/Flax."""
    
    embed_dims: int = 256
    num_heads: int = 8
    num_layers: int = 6
    ffn_dims: int = 1024
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, query, key, value, query_pos=None, key_pos=None, 
                 key_padding_mask=None, training=True):
        """
        Args:
            query: [B, N_queries, D] object queries
            key: [B, N_feat, D] image features 
            value: [B, N_feat, D] image features
            query_pos: [B, N_queries, D] query positional encoding
            key_pos: [B, N_feat, D] key positional encoding
            key_padding_mask: [B, N_feat] padding mask for features
        """
        output = query
        intermediate = []
        
        for i in range(self.num_layers):
            layer = TransformerLayer(
                embed_dims=self.embed_dims,
                num_heads=self.num_heads,
                ffn_dims=self.ffn_dims,
                dropout_rate=self.dropout_rate,
                name=f'layer_{i}'
            )
            output = layer(output, key, value, query_pos, key_pos, 
                         key_padding_mask, training)
            intermediate.append(nn.LayerNorm()(output))
            
        return jnp.stack(intermediate)  # [num_layers, B, N_queries, D]


class PETRHead(nn.Module):
    """PETR detection head in JAX/Flax."""
    
    embed_dims: int = 256
    num_classes: int = 3  # vehicle, pedestrian, cyclist
    num_reg_fcs: int = 2
    
    @nn.compact
    def __call__(self, features):
        """
        Args:
            features: [B, N_queries, D] decoder features
        Returns:
            cls_logits: [B, N_queries, num_classes+1] classification logits
            bbox_preds: [B, N_queries, 7] 3D bbox predictions
        """
        # Classification head
        cls_feat = features
        for i in range(self.num_reg_fcs):
            cls_feat = nn.Dense(self.embed_dims, name=f'cls_fc_{i}')(cls_feat)
            cls_feat = nn.relu(cls_feat)
        cls_logits = nn.Dense(self.num_classes + 1, name='cls_out')(cls_feat)
        
        # Regression head (3D bbox: center_x, center_y, center_z, length, width, height, heading)
        reg_feat = features
        for i in range(self.num_reg_fcs):
            reg_feat = nn.Dense(self.embed_dims, name=f'reg_fc_{i}')(reg_feat)
            reg_feat = nn.relu(reg_feat)
        bbox_preds = nn.Dense(7, name='reg_out')(reg_feat)
        
        return cls_logits, bbox_preds


class ResNet50Backbone(nn.Module):
    """Simplified ResNet50 backbone for JAX/Flax."""
    
    @nn.compact
    def __call__(self, x, training=True):
        # This is a simplified version - in production you'd want to use 
        # a proper ResNet50 implementation or load pretrained weights
        
        # Initial conv
        x = nn.Conv(64, kernel_size=(7, 7), strides=(2, 2), padding=3)(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding=1)
        
        # Simplified residual blocks
        for stage, (channels, num_blocks, stride) in enumerate([
            (64, 3, 1), (128, 4, 2), (256, 6, 2), (512, 3, 2)
        ]):
            for block in range(num_blocks):
                block_stride = stride if block == 0 else 1
                residual = x
                
                x = nn.Conv(channels, kernel_size=(3, 3), strides=(block_stride, block_stride), padding=1)(x)
                x = nn.BatchNorm(use_running_average=not training)(x)
                x = nn.relu(x)
                x = nn.Conv(channels, kernel_size=(3, 3), padding=1)(x)
                x = nn.BatchNorm(use_running_average=not training)(x)
                
                # Adjust residual if needed
                if block == 0 and (stride != 1 or residual.shape[-1] != channels):
                    residual = nn.Conv(channels, kernel_size=(1, 1), strides=(stride, stride))(residual)
                    residual = nn.BatchNorm(use_running_average=not training)(residual)
                
                x = nn.relu(x + residual)
        
        return x


class PETRModel(nn.Module):
    """
    PETR: Position Embedding Transformation for Multi-View 3D Object Detection.
    
    JAX/Flax implementation without PyTorch dependencies.
    """
    
    embed_dims: int = 256
    num_queries: int = 100
    num_classes: int = 3
    num_heads: int = 8
    num_layers: int = 6
    ffn_dims: int = 1024
    dropout_rate: float = 0.1
    point_cloud_range: Tuple = (-74.88, -74.88, -2, 74.88, 74.88, 4)
    num_cameras: int = 5
    
    def setup(self):
        # Image backbone
        self.backbone = ResNet50Backbone()
        
        # Feature projection
        self.input_proj = nn.Conv(self.embed_dims, kernel_size=(1, 1))
        
        # Positional encoding
        self.pos_encoding = nn.Dense(self.embed_dims)  # Simple linear projection of 3D positions
        
        # Transformer
        self.transformer = PETRTransformer(
            embed_dims=self.embed_dims,
            num_heads=self.num_heads, 
            num_layers=self.num_layers,
            ffn_dims=self.ffn_dims,
            dropout_rate=self.dropout_rate
        )
        
        # Detection head
        self.head = PETRHead(self.embed_dims, self.num_classes)
        
    def extract_features(self, images, training=True):
        """Extract features from multi-camera images.
        
        Args:
            images: [B, N_cams, H, W, 3] multi-camera images
        Returns:
            features: [B, N_cams*H*W, D] flattened features
        """
        batch_size, num_cams, H, W, C = images.shape
        
        # Flatten cameras and batch for backbone
        images_flat = images.reshape(batch_size * num_cams, H, W, C)
        
        # Extract features
        feats = self.backbone(images_flat, training)  # [B*N_cams, H', W', backbone_dims]
        feats = self.input_proj(feats)  # [B*N_cams, H', W', embed_dims]
        
        _, feat_h, feat_w, embed_dims = feats.shape
        
        # Reshape back and flatten spatial dimensions
        feats = feats.reshape(batch_size, num_cams, feat_h, feat_w, embed_dims)
        feats = feats.reshape(batch_size, num_cams * feat_h * feat_w, embed_dims)
        
        return feats
        
    def get_reference_points(self, batch_size):
        """Get 3D reference points for object queries."""
        # Initialize reference points uniformly in [0, 1]
        ref_points = self.param('reference_points', 
                               lambda rng: jax.random.uniform(rng, (self.num_queries, 3)))
        
        # Scale from [0, 1] to point cloud range
        pc_range = jnp.array(self.point_cloud_range)
        ref_points = ref_points * (pc_range[3:] - pc_range[:3]) + pc_range[:3]
        
        # Expand for batch
        ref_points = jnp.broadcast_to(ref_points[None, ...], (batch_size, self.num_queries, 3))
        
        return ref_points
        
    def __call__(self, images, training=True):
        """
        Args:
            images: [B, N_cams, H, W, 3] multi-camera images
        Returns:
            Dict containing:
                cls_logits: [B, num_queries, num_classes+1] 
                bbox_preds: [B, num_queries, 7]
                reference_points: [B, num_queries, 3]
        """
        batch_size = images.shape[0]
        
        # Extract image features
        img_feats = self.extract_features(images, training)  # [B, N_cams*H*W, D]
        
        # Get reference points for queries
        reference_points = self.get_reference_points(batch_size)  # [B, num_queries, 3]
        
        # Get object queries
        query_embed = self.param('query_embed', 
                                lambda rng: jax.random.normal(rng, (self.num_queries, self.embed_dims)))
        query_embed = jnp.broadcast_to(query_embed[None, ...], (batch_size, self.num_queries, self.embed_dims))
        
        # Get positional encodings
        query_pos = self.pos_encoding(reference_points)  # [B, num_queries, D]
        
        # For image features, we need to create dummy 3D positions
        # In a full implementation, this would use camera calibration to project 2D features to 3D
        img_pos = jnp.zeros((batch_size, img_feats.shape[1], 3))
        key_pos = self.pos_encoding(img_pos)  # [B, N_feat, D]
        
        # Transform through decoder
        decoder_out = self.transformer(
            query=query_embed,
            key=img_feats,
            value=img_feats,
            query_pos=query_pos,
            key_pos=key_pos,
            training=training
        )  # [num_layers, B, num_queries, D]
        
        # Use output from last layer
        features = decoder_out[-1]  # [B, num_queries, D]
        
        # Get predictions
        cls_logits, bbox_preds = self.head(features)
        
        return {
            'cls_logits': cls_logits,      # [B, num_queries, num_classes+1]
            'bbox_preds': bbox_preds,      # [B, num_queries, 7] 
            'reference_points': reference_points  # [B, num_queries, 3]
        }


def create_petr_model(
    embed_dims=256,
    num_queries=100,
    num_classes=3,
    num_cameras=5,
    **kwargs
):
    """Factory function to create PETR model."""
    return PETRModel(
        embed_dims=embed_dims,
        num_queries=num_queries,
        num_classes=num_classes,
        num_cameras=num_cameras,
        **kwargs
    )


if __name__ == "__main__":
    # Test the JAX model
    import jax.random as jrandom
    
    model = create_petr_model(
        embed_dims=256,
        num_queries=50,  # Smaller for testing
        num_classes=3,
        num_cameras=5
    )
    
    # Test input (JAX uses NHWC format)
    batch_size = 2
    key = jrandom.PRNGKey(0)
    images = jrandom.normal(key, (batch_size, 5, 200, 400, 3))  # 5 cameras, NHWC format
    
    # Initialize model
    key, subkey = jrandom.split(key)
    params = model.init(subkey, images, training=False)
    
    # Forward pass
    outputs = model.apply(params, images, training=False)
    
    print("JAX PETR model test successful!")
    print(f"Class logits shape: {outputs['cls_logits'].shape}")
    print(f"Bbox predictions shape: {outputs['bbox_preds'].shape}")
    print(f"Reference points shape: {outputs['reference_points'].shape}")
    
    # Count parameters
    param_count = sum(x.size for x in jax.tree_leaves(params))
    print(f"Total parameters: {param_count:,}")