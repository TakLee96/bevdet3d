#!/usr/bin/env python3
"""
Quick BEVFormer training test to verify it works with a few training steps.
"""

import os
import sys
import warnings

# Suppress all warnings and TensorFlow output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
from torch.utils.data import DataLoader
from data.simple_waymo_dataset import SimpleCachedWaymoDataset, collate_fn
from models.bevformer import BEVFormerModel
from utils.losses import PETRLoss
from utils.metrics import compute_metrics, filter_predictions


def prepare_targets(batch):
    """Convert batch data to target format for loss computation."""
    targets = []
    
    for i in range(len(batch['gt_labels_3d'])):
        gt_labels = batch['gt_labels_3d'][i]
        gt_boxes = batch['gt_bboxes_3d'][i]
        
        # Filter out invalid boxes (all zeros)
        valid_mask = (gt_boxes.sum(dim=-1) != 0)
        
        targets.append({
            'labels': gt_labels[valid_mask],
            'boxes_3d': gt_boxes[valid_mask]
        })
    
    return targets


def main():
    """Quick BEVFormer training test."""
    print("Quick BEVFormer Training Test\n")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create cached dataset
    print("Using cached dataset...")
    dataset = SimpleCachedWaymoDataset(
        data_root=os.path.join(os.path.dirname(__file__), "..", "waymo_open_dataset_v_1_4_3"),
        split="training",
        max_files=2,
        max_frames_per_file=3,
        image_size=(400, 200)
    )
    
    print(f"Dataset: {len(dataset)} samples")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    # Create BEVFormer model
    print("Creating BEVFormer model...")
    model = BEVFormerModel(
        backbone_name='resnet50',
        embed_dims=64,       # Very small for fast testing
        num_queries=25,      # Few queries
        num_classes=3,
        bev_h=15,           # Small BEV grid
        bev_w=15,
        num_cameras=5,
        num_encoder_layers=2  # Very few layers
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model created with {total_params:,} parameters")
    
    # Create loss and optimizer
    criterion = PETRLoss(num_classes=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    print("\nStarting BEVFormer training test...")
    
    model.train()
    prev_bev = None  # For temporal modeling
    
    # Train for a few steps
    for step, batch in enumerate(dataloader):
        if step >= 3:  # Only do 3 steps
            break
            
        print(f"\nStep {step + 1}/3:")
        
        # Move data to device
        images = batch['images'].to(device)
        targets = prepare_targets(batch)
        
        # Move targets to device
        for target in targets:
            for k, v in target.items():
                target[k] = v.to(device)
        
        print(f"  Input: {images.shape}, Targets: {len(targets)} objects")
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass with temporal modeling
        outputs = model(images, prev_bev=prev_bev)
        
        # Store BEV features for next iteration
        prev_bev = outputs['bev_features'].detach()
        
        print(f"  Forward pass: cls={outputs['cls_logits'].shape}, bbox={outputs['bbox_preds'].shape}")
        print(f"  BEV features: {outputs['bev_features'].shape}")
        
        # Compute loss
        losses = criterion(outputs, targets)
        total_loss = sum(losses.values())
        
        print(f"  Losses: total={total_loss.item():.4f}, ce={losses['loss_ce'].item():.4f}, bbox={losses['loss_bbox'].item():.4f}")
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        
        # Update weights
        optimizer.step()
        
        print(f"  ✓ Step completed successfully (with temporal BEV)")
    
    print("\n🎉 Quick BEVFormer training test completed successfully!")
    print("BEVFormer is ready for full training.")
    
    # Test evaluation
    print("\nTesting BEVFormer evaluation...")
    model.eval()
    
    with torch.no_grad():
        batch = next(iter(dataloader))
        images = batch['images'].to(device)
        targets = prepare_targets(batch)
        
        # Move targets to device
        for target in targets:
            for k, v in target.items():
                target[k] = v.to(device)
        
        outputs = model(images, prev_bev=prev_bev)
        
        # Convert to predictions
        predictions = filter_predictions(outputs, score_threshold=0.1)
        
        # Compute metrics
        metrics = compute_metrics(predictions, targets, num_classes=3)
        
        print(f"Evaluation metrics: {metrics}")
        print("✓ BEVFormer evaluation test passed")


if __name__ == "__main__":
    main()