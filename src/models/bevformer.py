import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from einops import rearrange, repeat
import torchvision


class PositionalEncodingLearned2D(nn.Module):
    """2D learned positional encoding for BEV queries."""
    
    def __init__(self, embed_dims=256, h=50, w=50):
        super().__init__()
        self.embed_dims = embed_dims
        self.h = h
        self.w = w
        
        # Learned positional embeddings
        self.row_embed = nn.Embedding(h, embed_dims // 2)
        self.col_embed = nn.Embedding(w, embed_dims // 2)
        
    def forward(self, x):
        """
        Args:
            x: [B, H, W, D] or [B, H*W, D]
        Returns:
            pos: [B, H*W, D] positional encoding
        """
        h, w = self.h, self.w
        
        # Create position indices
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        
        # Get embeddings
        x_emb = self.col_embed(i)  # [W, D//2]
        y_emb = self.row_embed(j)  # [H, D//2]
        
        # Create 2D positional encoding
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),  # [H, W, D//2]
            y_emb.unsqueeze(1).repeat(1, w, 1),  # [H, W, D//2]
        ], dim=-1).flatten(0, 1)  # [H*W, D]
        
        # Expand for batch
        batch_size = x.shape[0]
        pos = pos.unsqueeze(0).expand(batch_size, -1, -1)  # [B, H*W, D]
        
        return pos


class SpatialCrossAttention(nn.Module):
    """Spatial cross-attention for multi-camera feature aggregation."""
    
    def __init__(self, embed_dims=256, num_heads=8, num_levels=4, num_points=4):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.head_dims = embed_dims // num_heads
        
        # Attention weights
        self.attention_weights = nn.Linear(embed_dims, num_heads * num_levels * num_points)
        
        # Value projection  
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        
        # Sampling offset
        self.sampling_offsets = nn.Linear(embed_dims, num_heads * num_levels * num_points * 2)
        
    def forward(self, query, reference_points, value, spatial_shapes):
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
        attention_weights = self.attention_weights(query)  # [B, H*W, num_heads*num_levels*num_points]
        sampling_offsets = self.sampling_offsets(query)    # [B, H*W, num_heads*num_levels*num_points*2]
        
        # Reshape
        attention_weights = attention_weights.view(
            batch_size, num_queries, self.num_heads, self.num_levels * self.num_points)
        sampling_offsets = sampling_offsets.view(
            batch_size, num_queries, self.num_heads, self.num_levels, self.num_points, 2)
        
        # Normalize attention weights
        attention_weights = F.softmax(attention_weights, dim=-1)
        
        # Project values
        value = self.value_proj(value).view(
            batch_size, num_value, self.num_heads, self.head_dims)
        
        # For simplicity, use average pooling instead of deformable attention
        # In a full implementation, this would do proper deformable sampling
        output = torch.mean(value, dim=1, keepdim=True).expand(-1, num_queries, -1, -1)
        output = output.contiguous().view(batch_size, num_queries, self.embed_dims)
        
        return self.output_proj(output)


class TemporalSelfAttention(nn.Module):
    """Temporal self-attention for BEV feature evolution."""
    
    def __init__(self, embed_dims=256, num_heads=8):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        
        self.self_attn = nn.MultiheadAttention(embed_dims, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dims)
        
    def forward(self, query, prev_bev=None, bev_pos=None):
        """
        Args:
            query: [B, H*W, D] current BEV queries
            prev_bev: [B, H*W, D] previous BEV features (optional)
            bev_pos: [B, H*W, D] BEV positional encoding
        """
        if prev_bev is not None:
            # Concatenate current and previous BEV features
            key_value = torch.cat([query, prev_bev], dim=1)  # [B, 2*H*W, D]
        else:
            key_value = query
            
        # Add positional encoding
        if bev_pos is not None:
            query = query + bev_pos
            if prev_bev is not None:
                key_value = key_value + torch.cat([bev_pos, bev_pos], dim=1)
            else:
                key_value = key_value + bev_pos
        
        # Self-attention
        output, _ = self.self_attn(query, key_value, key_value)
        
        return self.norm(output + query)


class BEVFormerLayer(nn.Module):
    """Single BEVFormer encoder layer."""
    
    def __init__(self, embed_dims=256, num_heads=8, ffn_dims=1024, dropout=0.1):
        super().__init__()
        
        # Temporal self-attention
        self.temporal_self_attn = TemporalSelfAttention(embed_dims, num_heads)
        
        # Spatial cross-attention
        self.spatial_cross_attn = SpatialCrossAttention(embed_dims, num_heads)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dims, ffn_dims),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ffn_dims, embed_dims),
            nn.Dropout(dropout)
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)
        self.norm3 = nn.LayerNorm(embed_dims)
        
    def forward(self, bev_query, reference_points, img_feats, spatial_shapes, prev_bev=None, bev_pos=None):
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
        bev_query = self.temporal_self_attn(
            self.norm1(bev_query), prev_bev, bev_pos)
        
        # Spatial cross-attention
        bev_query = bev_query + self.spatial_cross_attn(
            self.norm2(bev_query), reference_points, img_feats, spatial_shapes)
        
        # Feed-forward
        bev_query = bev_query + self.ffn(self.norm3(bev_query))
        
        return bev_query


class BEVFormerEncoder(nn.Module):
    """BEVFormer encoder with multiple layers."""
    
    def __init__(self, embed_dims=256, num_layers=6, **kwargs):
        super().__init__()
        
        self.layers = nn.ModuleList([
            BEVFormerLayer(embed_dims, **kwargs)
            for _ in range(num_layers)
        ])
        
    def forward(self, bev_query, reference_points, img_feats, spatial_shapes, prev_bev=None, bev_pos=None):
        """Forward through all encoder layers."""
        output = bev_query
        
        for layer in self.layers:
            output = layer(output, reference_points, img_feats, spatial_shapes, prev_bev, bev_pos)
            
        return output


class BEVFormerModel(nn.Module):
    """
    BEVFormer: Learning Bird's-Eye-View Representation from Multi-Camera Images.
    
    This is a pure PyTorch implementation without mmdet3d dependencies.
    """
    
    def __init__(self,
                 backbone_name='resnet50',
                 embed_dims=256,
                 num_queries=100,
                 num_classes=3,
                 bev_h=50,
                 bev_w=50,
                 num_cameras=5,
                 num_encoder_layers=6,
                 pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]):
        super().__init__()
        
        self.embed_dims = embed_dims
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.num_cameras = num_cameras
        self.pc_range = np.array(pc_range)
        
        # Image backbone
        if backbone_name == 'resnet50':
            backbone = torchvision.models.resnet50(pretrained=True)
            self.backbone = nn.Sequential(*list(backbone.children())[:-2])
            backbone_dims = 2048
        else:
            raise NotImplementedError(f"Backbone {backbone_name} not implemented")
            
        # Feature projection
        self.input_proj = nn.Conv2d(backbone_dims, embed_dims, kernel_size=1)
        
        # BEV queries
        self.bev_queries = nn.Embedding(bev_h * bev_w, embed_dims)
        
        # BEV positional encoding
        self.bev_pos_encoding = PositionalEncodingLearned2D(embed_dims, bev_h, bev_w)
        
        # Reference points for BEV queries (normalized coordinates)
        self.register_buffer('reference_points', self._get_reference_points())
        
        # BEVFormer encoder
        self.encoder = BEVFormerEncoder(
            embed_dims=embed_dims,
            num_layers=num_encoder_layers
        )
        
        # Object queries for detection
        self.object_queries = nn.Embedding(num_queries, embed_dims)
        
        # Detection head (similar to PETR)
        self.cls_head = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, num_classes + 1)
        )
        
        self.reg_head = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, 7)  # 3D bbox: x, y, z, l, w, h, rot
        )
        
        # Cross-attention for detection (BEV features -> object queries)
        self.detection_cross_attn = nn.MultiheadAttention(embed_dims, num_heads=8, batch_first=True)
        
    def _get_reference_points(self):
        """Generate 2D reference points for BEV queries."""
        # Create grid of reference points in BEV space
        y_coords = torch.linspace(0, 1, self.bev_h)
        x_coords = torch.linspace(0, 1, self.bev_w)
        
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        # Stack and reshape to [H*W, 2]
        ref_points = torch.stack([x_grid, y_grid], dim=-1).reshape(-1, 2)
        
        return ref_points
        
    def extract_img_features(self, images):
        """Extract features from multi-camera images."""
        batch_size, num_cams, C, H, W = images.shape
        
        # Flatten cameras and batch for backbone
        images_flat = images.view(batch_size * num_cams, C, H, W)
        
        # Extract features
        feats = self.backbone(images_flat)  # [B*N_cams, backbone_dims, H', W']
        feats = self.input_proj(feats)      # [B*N_cams, embed_dims, H', W']
        
        _, embed_dims, feat_h, feat_w = feats.shape
        
        # Reshape and flatten spatial dimensions
        feats = feats.view(batch_size, num_cams, embed_dims, feat_h, feat_w)
        feats = feats.permute(0, 1, 3, 4, 2).contiguous()  # [B, N_cams, H', W', D]
        feats = feats.view(batch_size, num_cams * feat_h * feat_w, embed_dims)
        
        spatial_shapes = [(feat_h, feat_w)] * num_cams
        
        return feats, spatial_shapes
        
    def forward(self, images, camera_infos=None, prev_bev=None):
        """
        Args:
            images: [B, N_cams, 3, H, W] multi-camera images
            camera_infos: Camera calibration info (not used in this simplified version)
            prev_bev: Previous BEV features for temporal modeling
        """
        batch_size = images.shape[0]
        device = images.device
        
        # Extract image features
        img_feats, spatial_shapes = self.extract_img_features(images)
        
        # Get BEV queries and positional encoding
        bev_queries = self.bev_queries.weight.unsqueeze(0).expand(batch_size, -1, -1)  # [B, H*W, D]
        bev_pos = self.bev_pos_encoding(bev_queries)  # [B, H*W, D]
        
        # Get reference points
        reference_points = self.reference_points.unsqueeze(0).expand(batch_size, -1, -1)  # [B, H*W, 2]
        
        # BEVFormer encoder
        bev_features = self.encoder(
            bev_query=bev_queries,
            reference_points=reference_points,
            img_feats=img_feats,
            spatial_shapes=spatial_shapes,
            prev_bev=prev_bev,
            bev_pos=bev_pos
        )  # [B, H*W, D]
        
        # Object detection from BEV features
        object_queries = self.object_queries.weight.unsqueeze(0).expand(batch_size, -1, -1)  # [B, num_queries, D]
        
        # Cross-attention: object queries attend to BEV features
        detection_features, _ = self.detection_cross_attn(
            object_queries, bev_features, bev_features)  # [B, num_queries, D]
        
        # Generate predictions
        cls_logits = self.cls_head(detection_features)  # [B, num_queries, num_classes+1]
        bbox_preds = self.reg_head(detection_features)  # [B, num_queries, 7]
        
        # Reference points for 3D detection (simplified)
        reference_points_3d = torch.zeros(batch_size, self.num_queries, 3, device=device)
        
        return {
            'cls_logits': cls_logits,
            'bbox_preds': bbox_preds,
            'reference_points': reference_points_3d,
            'bev_features': bev_features  # For temporal modeling
        }


if __name__ == "__main__":
    # Test the full BEVFormer model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = BEVFormerModel(
        backbone_name='resnet50',
        embed_dims=128,  # Smaller for testing
        num_queries=50,
        num_classes=3,
        bev_h=25,       # Smaller BEV grid
        bev_w=25,
        num_cameras=5,
        num_encoder_layers=3  # Fewer layers for testing
    ).to(device)
    
    # Test input
    batch_size = 1
    images = torch.randn(batch_size, 5, 3, 200, 400).to(device)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(images)
        
    print("BEVFormer test successful!")
    print(f"Class logits shape: {outputs['cls_logits'].shape}")
    print(f"Bbox predictions shape: {outputs['bbox_preds'].shape}")
    print(f"Reference points shape: {outputs['reference_points'].shape}")
    print(f"BEV features shape: {outputs['bev_features'].shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test temporal modeling
    print("\nTesting temporal modeling...")
    prev_bev = outputs['bev_features']
    with torch.no_grad():
        outputs2 = model(images, prev_bev=prev_bev)
    print(f"Temporal BEV features shape: {outputs2['bev_features'].shape}")
    print("✓ Temporal modeling test passed")