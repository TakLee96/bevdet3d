#!/usr/bin/env python3
"""
Example usage of PETR training on Waymo Open Dataset.

To run training:
python train_petr.py --data_root /path/to/waymo_open_dataset_v_1_4_3 --batch_size 2 --num_epochs 10 --max_frames 50

This example shows how to use the implemented components.
"""

import sys
import os
sys.path.append('src')

import torch
from data.waymo_dataset import WaymoDataset
from models.petr import PETRModel
from utils.losses import PETRLoss
from utils.metrics import compute_metrics, filter_predictions


def main():
    """Example usage of the PETR implementation."""
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create a small dataset for testing
    dataset = WaymoDataset(
        data_root="waymo_open_dataset_v_1_4_3",
        split="training", 
        num_frames_per_segment=5,  # Small number for testing
        image_size=(800, 320),
        load_point_cloud=False
    )
    
    print(f"Dataset loaded: {len(dataset)} samples")
    
    # Create model
    model = PETRModel(
        backbone_name='resnet50',
        embed_dims=256,
        num_queries=100, 
        num_classes=3,
        num_cameras=5
    ).to(device)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create loss function
    criterion = PETRLoss(num_classes=3).to(device)
    
    # Test with a single batch
    sample = dataset[0]
    batch = {
        'images': sample['images'].unsqueeze(0),  # Add batch dimension
        'gt_labels_3d': [sample['gt_labels_3d']],
        'gt_bboxes_3d': [sample['gt_bboxes_3d']]
    }
    
    # Move to device
    images = batch['images'].to(device)
    targets = [{
        'labels': batch['gt_labels_3d'][0].to(device),
        'boxes_3d': batch['gt_bboxes_3d'][0].to(device)
    }]
    
    print(f"Input images shape: {images.shape}")
    print(f"Target labels: {targets[0]['labels'].shape}")
    print(f"Target boxes: {targets[0]['boxes_3d'].shape}")
    
    # Forward pass
    with torch.no_grad():
        outputs = model(images)
        
    print(f"Output shapes:")
    print(f"  cls_logits: {outputs['cls_logits'].shape}")
    print(f"  bbox_preds: {outputs['bbox_preds'].shape}")
    print(f"  reference_points: {outputs['reference_points'].shape}")
    
    # Compute loss
    losses = criterion(outputs, targets)
    print(f"Losses: {losses}")
    
    # Convert to predictions and compute metrics
    predictions = filter_predictions(outputs, score_threshold=0.1)
    metrics = compute_metrics(predictions, targets, num_classes=3)
    print(f"Metrics: {metrics}")
    
    print("Example completed successfully!")


if __name__ == "__main__":
    main()