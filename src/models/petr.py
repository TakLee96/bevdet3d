import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from einops import rearrange
import torchvision


class PositionalEncoding3D(nn.Module):
    """3D positional encoding for PETR."""
    
    def __init__(self, num_pos_feats=128, temperature=10000):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        
    def forward(self, pos):
        """
        Args:
            pos: [N, 3] positions in 3D space
        Returns:
            pos_embed: [N, num_pos_feats*3] positional embeddings
        """
        scale = 2 * math.pi
        pos = pos * scale
        
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=pos.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        
        pos_x = pos[..., 0, None] / dim_t
        pos_y = pos[..., 1, None] / dim_t  
        pos_z = pos[..., 2, None] / dim_t
        
        # Handle odd dimensions by ensuring even number of features
        pos_x_sin = pos_x[..., 0::2].sin()
        pos_x_cos = pos_x[..., 1::2].cos()
        if pos_x_sin.shape[-1] != pos_x_cos.shape[-1]:
            # Trim the larger one
            min_dim = min(pos_x_sin.shape[-1], pos_x_cos.shape[-1])
            pos_x_sin = pos_x_sin[..., :min_dim]
            pos_x_cos = pos_x_cos[..., :min_dim]
        pos_x = torch.stack((pos_x_sin, pos_x_cos), dim=-1).flatten(-2)
        
        pos_y_sin = pos_y[..., 0::2].sin()
        pos_y_cos = pos_y[..., 1::2].cos()
        if pos_y_sin.shape[-1] != pos_y_cos.shape[-1]:
            min_dim = min(pos_y_sin.shape[-1], pos_y_cos.shape[-1])
            pos_y_sin = pos_y_sin[..., :min_dim]
            pos_y_cos = pos_y_cos[..., :min_dim]
        pos_y = torch.stack((pos_y_sin, pos_y_cos), dim=-1).flatten(-2)
        
        pos_z_sin = pos_z[..., 0::2].sin()
        pos_z_cos = pos_z[..., 1::2].cos()
        if pos_z_sin.shape[-1] != pos_z_cos.shape[-1]:
            min_dim = min(pos_z_sin.shape[-1], pos_z_cos.shape[-1])
            pos_z_sin = pos_z_sin[..., :min_dim]
            pos_z_cos = pos_z_cos[..., :min_dim]
        pos_z = torch.stack((pos_z_sin, pos_z_cos), dim=-1).flatten(-2)
        
        pos_embed = torch.cat((pos_y, pos_x, pos_z), dim=-1)
        return pos_embed


class MultiHeadAttention(nn.Module):
    """Multi-head attention module."""
    
    def __init__(self, embed_dims=256, num_heads=8, dropout=0.1):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.head_dims = embed_dims // num_heads
        
        assert embed_dims % num_heads == 0
        
        self.q_proj = nn.Linear(embed_dims, embed_dims)
        self.k_proj = nn.Linear(embed_dims, embed_dims)
        self.v_proj = nn.Linear(embed_dims, embed_dims)
        self.out_proj = nn.Linear(embed_dims, embed_dims)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dims ** -0.5
        
    def forward(self, query, key, value, key_padding_mask=None, pos_embed=None):
        batch_size, seq_len = query.shape[:2]
        
        # Project to Q, K, V
        q = self.q_proj(query)  # [B, N, D]
        k = self.k_proj(key)    # [B, M, D]
        v = self.v_proj(value)  # [B, M, D]
        
        # Add positional encoding if provided
        if pos_embed is not None:
            k = k + pos_embed
            
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dims).transpose(1, 2)  # [B, H, N, D_h]
        k = k.view(batch_size, -1, self.num_heads, self.head_dims).transpose(1, 2)       # [B, H, M, D_h]  
        v = v.view(batch_size, -1, self.num_heads, self.head_dims).transpose(1, 2)       # [B, H, M, D_h]
        
        # Compute attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, N, M]
        
        # Apply key padding mask
        if key_padding_mask is not None:
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
                
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)  # [B, H, N, D_h]
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.embed_dims)
            
        return self.out_proj(attn_output)


class TransformerLayer(nn.Module):
    """Transformer layer with self-attention and cross-attention."""
    
    def __init__(self, embed_dims=256, num_heads=8, ffn_dims=1024, dropout=0.1):
        super().__init__()
        
        # Self-attention
        self.self_attn = MultiHeadAttention(embed_dims, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dims)
        self.dropout1 = nn.Dropout(dropout)
        
        # Cross-attention
        self.cross_attn = MultiHeadAttention(embed_dims, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dims)
        self.dropout2 = nn.Dropout(dropout)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dims, ffn_dims),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ffn_dims, embed_dims)
        )
        self.norm3 = nn.LayerNorm(embed_dims)
        self.dropout3 = nn.Dropout(dropout)
        
    def forward(self, query, key, value, query_pos=None, key_pos=None, key_padding_mask=None):
        # Self-attention
        q = self.norm1(query)
        q = self.self_attn(q, q, q, pos_embed=query_pos)
        query = query + self.dropout1(q)
        
        # Cross-attention
        q = self.norm2(query)
        q = self.cross_attn(q, key, value, key_padding_mask=key_padding_mask, pos_embed=key_pos)
        query = query + self.dropout2(q)
        
        # Feed-forward
        q = self.norm3(query)
        q = self.ffn(q)
        query = query + self.dropout3(q)
        
        return query


class PETRTransformer(nn.Module):
    """PETR Transformer decoder."""
    
    def __init__(self, 
                 embed_dims=256,
                 num_heads=8,
                 num_layers=6,
                 ffn_dims=1024,
                 dropout=0.1):
        super().__init__()
        
        self.embed_dims = embed_dims
        self.num_layers = num_layers
        
        self.layers = nn.ModuleList([
            TransformerLayer(embed_dims, num_heads, ffn_dims, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dims)
        
    def forward(self, query, key, value, query_pos=None, key_pos=None, key_padding_mask=None):
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
        
        for layer in self.layers:
            output = layer(output, key, value, query_pos, key_pos, key_padding_mask)
            intermediate.append(self.norm(output))
            
        return torch.stack(intermediate)  # [num_layers, B, N_queries, D]


class PETRHead(nn.Module):
    """PETR detection head."""
    
    def __init__(self, 
                 embed_dims=256,
                 num_classes=3,  # vehicle, pedestrian, cyclist
                 num_reg_fcs=2):
        super().__init__()
        
        self.embed_dims = embed_dims
        self.num_classes = num_classes
        
        # Classification head
        cls_layers = []
        for _ in range(num_reg_fcs):
            cls_layers.extend([
                nn.Linear(embed_dims, embed_dims),
                nn.ReLU(inplace=True)
            ])
        cls_layers.append(nn.Linear(embed_dims, num_classes + 1))  # +1 for background
        self.cls_head = nn.Sequential(*cls_layers)
        
        # Regression head (3D bbox: center_x, center_y, center_z, length, width, height, heading)
        reg_layers = []
        for _ in range(num_reg_fcs):
            reg_layers.extend([
                nn.Linear(embed_dims, embed_dims),
                nn.ReLU(inplace=True)
            ])
        reg_layers.append(nn.Linear(embed_dims, 7))
        self.reg_head = nn.Sequential(*reg_layers)
        
    def forward(self, features):
        """
        Args:
            features: [B, N_queries, D] decoder features
        Returns:
            cls_logits: [B, N_queries, num_classes+1] classification logits
            bbox_preds: [B, N_queries, 7] 3D bbox predictions
        """
        cls_logits = self.cls_head(features)
        bbox_preds = self.reg_head(features)
        
        return cls_logits, bbox_preds


class PETRModel(nn.Module):
    """
    PETR: Position Embedding Transformation for Multi-View 3D Object Detection.
    
    This is a pure PyTorch implementation without mmdet3d dependencies.
    """
    
    def __init__(self,
                 backbone_name='resnet50',
                 embed_dims=256,
                 num_queries=100,
                 num_classes=3,
                 num_heads=8,
                 num_layers=6,
                 ffn_dims=1024,
                 dropout=0.1,
                 point_cloud_range=[-74.88, -74.88, -2, 74.88, 74.88, 4],
                 num_cameras=5):
        super().__init__()
        
        self.embed_dims = embed_dims
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.num_cameras = num_cameras
        self.point_cloud_range = np.array(point_cloud_range)
        
        # Image backbone
        if backbone_name == 'resnet50':
            backbone = torchvision.models.resnet50(pretrained=True)
            self.backbone = nn.Sequential(*list(backbone.children())[:-2])  # Remove FC and avgpool
            backbone_dims = 2048
        else:
            raise NotImplementedError(f"Backbone {backbone_name} not implemented")
            
        # Feature projection
        self.input_proj = nn.Conv2d(backbone_dims, embed_dims, kernel_size=1)
        
        # Positional encoding - simplified to match embed_dims exactly
        self.pos_encoding = nn.Linear(3, embed_dims)  # Simple linear projection of 3D positions
        
        # Object queries
        self.query_embed = nn.Embedding(num_queries, embed_dims)
        
        # Position reference points for queries
        self.reference_points = nn.Embedding(num_queries, 3)
        nn.init.uniform_(self.reference_points.weight.data, 0, 1)
        
        # Transformer
        self.transformer = PETRTransformer(
            embed_dims=embed_dims,
            num_heads=num_heads, 
            num_layers=num_layers,
            ffn_dims=ffn_dims,
            dropout=dropout
        )
        
        # Detection head
        self.head = PETRHead(embed_dims, num_classes)
        
    def extract_features(self, images):
        """Extract features from multi-camera images.
        
        Args:
            images: [B, N_cams, 3, H, W] multi-camera images
        Returns:
            features: [B, N_cams*H*W, D] flattened features
            spatial_shapes: List of (H, W) for each camera
        """
        batch_size, num_cams, C, H, W = images.shape
        
        # Flatten cameras and batch for backbone
        images_flat = images.view(batch_size * num_cams, C, H, W)
        
        # Extract features
        feats = self.backbone(images_flat)  # [B*N_cams, backbone_dims, H', W']
        feats = self.input_proj(feats)      # [B*N_cams, embed_dims, H', W']
        
        _, embed_dims, feat_h, feat_w = feats.shape
        
        # Reshape back and flatten spatial dimensions
        feats = feats.view(batch_size, num_cams, embed_dims, feat_h, feat_w)
        feats = feats.permute(0, 1, 3, 4, 2).contiguous()  # [B, N_cams, H', W', D]
        feats = feats.view(batch_size, num_cams * feat_h * feat_w, embed_dims)
        
        return feats, [(feat_h, feat_w)] * num_cams
        
    def get_reference_points(self, batch_size, device):
        """Get 3D reference points for object queries."""
        # Scale reference points to point cloud range
        ref_points = self.reference_points.weight.clone()  # [num_queries, 3]
        
        # Scale from [0, 1] to point cloud range
        pc_range = torch.tensor(self.point_cloud_range, device=device, dtype=torch.float32)
        ref_points = ref_points * (pc_range[3:] - pc_range[:3]) + pc_range[:3]
        
        # Expand for batch
        ref_points = ref_points.unsqueeze(0).expand(batch_size, -1, -1)  # [B, num_queries, 3]
        
        return ref_points
        
    def forward(self, images, camera_infos=None):
        """
        Args:
            images: [B, N_cams, 3, H, W] multi-camera images
            camera_infos: List of camera calibration info (not used in this simple version)
        Returns:
            Dict containing:
                cls_logits: [B, num_queries, num_classes+1] 
                bbox_preds: [B, num_queries, 7]
                reference_points: [B, num_queries, 3]
        """
        batch_size = images.shape[0]
        device = images.device
        
        # Extract image features
        img_feats, spatial_shapes = self.extract_features(images)  # [B, N_cams*H*W, D]
        
        # Get reference points for queries
        reference_points = self.get_reference_points(batch_size, device)  # [B, num_queries, 3]
        
        # Get object queries
        query_embed = self.query_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)  # [B, num_queries, D]
        
        # Get positional encodings
        query_pos = self.pos_encoding(reference_points)  # [B, num_queries, D]
        
        # For image features, we need to create dummy 3D positions
        # In a full implementation, this would use camera calibration to project 2D features to 3D
        img_pos = torch.zeros(batch_size, img_feats.shape[1], 3, device=device)
        key_pos = self.pos_encoding(img_pos)  # [B, N_feat, D]
        
        # Transform through decoder
        decoder_out = self.transformer(
            query=query_embed,
            key=img_feats,
            value=img_feats,
            query_pos=query_pos,
            key_pos=key_pos
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


if __name__ == "__main__":
    # Test the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = PETRModel(
        backbone_name='resnet50',
        embed_dims=256,
        num_queries=100,
        num_classes=3,
        num_cameras=5
    ).to(device)
    
    # Test input
    batch_size = 2
    images = torch.randn(batch_size, 5, 3, 320, 800).to(device)  # 5 cameras
    
    # Forward pass
    with torch.no_grad():
        outputs = model(images)
        
    print("Model test successful!")
    print(f"Class logits shape: {outputs['cls_logits'].shape}")
    print(f"Bbox predictions shape: {outputs['bbox_preds'].shape}")
    print(f"Reference points shape: {outputs['reference_points'].shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")