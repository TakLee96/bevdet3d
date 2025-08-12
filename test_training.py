#!/usr/bin/env python3
"""
Simple test script to verify PETR training works step by step.
"""

import os
import sys
import warnings

# Suppress all warnings and TensorFlow output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

sys.path.append('src')

import torch
from torch.utils.data import DataLoader
from data.waymo_dataset import WaymoDataset, collate_fn
from models.petr import PETRModel
from utils.losses import PETRLoss


def test_dataset_loading():
    """Test dataset loading."""
    print("1. Testing dataset loading...")
    
    try:
        dataset = WaymoDataset(
            data_root="waymo_open_dataset_v_1_4_3",
            split="training",
            num_frames_per_segment=2,  # Very small for testing
            image_size=(400, 200),  # Smaller images for faster processing
            load_point_cloud=False
        )
        
        print(f"   ✓ Dataset created with {len(dataset)} samples")
        
        # Test loading one sample
        sample = dataset[0]
        print(f"   ✓ Sample loaded: images={sample['images'].shape}, "
              f"labels={sample['gt_labels_3d'].shape}, boxes={sample['gt_bboxes_3d'].shape}")
        
        return dataset
        
    except Exception as e:
        print(f"   ✗ Dataset loading failed: {e}")
        return None


def test_model_creation():
    """Test model creation and forward pass."""
    print("2. Testing model creation...")
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = PETRModel(
            backbone_name='resnet50',
            embed_dims=128,  # Smaller for faster testing
            num_queries=50,   # Fewer queries
            num_classes=3,
            num_cameras=5
        ).to(device)
        
        # Test forward pass
        batch_size = 1
        images = torch.randn(batch_size, 5, 3, 200, 400).to(device)
        
        with torch.no_grad():
            outputs = model(images)
            
        print(f"   ✓ Model created and forward pass successful")
        print(f"   ✓ Output shapes: cls={outputs['cls_logits'].shape}, bbox={outputs['bbox_preds'].shape}")
        
        return model, device
        
    except Exception as e:
        print(f"   ✗ Model creation failed: {e}")
        return None, None


def test_loss_computation(model, device):
    """Test loss computation."""
    print("3. Testing loss computation...")
    
    try:
        criterion = PETRLoss(num_classes=3).to(device)
        
        # Create dummy batch
        batch_size = 1
        outputs = {
            'cls_logits': torch.randn(batch_size, 50, 4, requires_grad=True).to(device),
            'bbox_preds': torch.randn(batch_size, 50, 7, requires_grad=True).to(device),
        }
        
        targets = [{
            'labels': torch.tensor([0, 1], dtype=torch.long).to(device),
            'boxes_3d': torch.randn(2, 7).to(device)
        }]
        
        losses = criterion(outputs, targets)
        total_loss = sum(losses.values())
        
        print(f"   ✓ Loss computation successful: {total_loss.item():.4f}")
        print(f"   ✓ Individual losses: {[(k, v.item()) for k, v in losses.items()]}")
        
        # Test backward pass
        total_loss.backward()
        print(f"   ✓ Backward pass successful")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Loss computation failed: {e}")
        return False


def test_dataloader(dataset):
    """Test dataloader with actual data."""
    print("4. Testing dataloader...")
    
    try:
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,  # Single thread for debugging
            collate_fn=collate_fn
        )
        
        # Get one batch
        batch = next(iter(dataloader))
        
        print(f"   ✓ Dataloader working")
        print(f"   ✓ Batch keys: {list(batch.keys())}")
        print(f"   ✓ Images shape: {batch['images'].shape}")
        print(f"   ✓ Number of targets: {len(batch['gt_labels_3d'])}")
        
        return dataloader
        
    except Exception as e:
        print(f"   ✗ Dataloader failed: {e}")
        return None


def test_full_training_step(model, dataloader, device):
    """Test a full training step."""
    print("5. Testing full training step...")
    
    try:
        model.train()
        criterion = PETRLoss(num_classes=3).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        # Get one batch
        batch = next(iter(dataloader))
        
        # Prepare data
        images = batch['images'].to(device)
        targets = []
        for i in range(len(batch['gt_labels_3d'])):
            gt_labels = batch['gt_labels_3d'][i].to(device)
            gt_boxes = batch['gt_bboxes_3d'][i].to(device)
            
            # Filter valid boxes
            valid_mask = (gt_boxes.sum(dim=-1) != 0)
            targets.append({
                'labels': gt_labels[valid_mask],
                'boxes_3d': gt_boxes[valid_mask]
            })
        
        print(f"   ✓ Data prepared: {images.shape}, {len(targets)} targets")
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        
        print(f"   ✓ Forward pass completed")
        
        # Loss computation
        losses = criterion(outputs, targets)
        total_loss = sum(losses.values())
        
        print(f"   ✓ Loss computed: {total_loss.item():.4f}")
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        print(f"   ✓ Training step completed successfully!")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Training step failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("Testing PETR training pipeline step by step...\n")
    
    # Test 1: Dataset loading
    dataset = test_dataset_loading()
    if dataset is None:
        return
    
    # Test 2: Model creation
    model, device = test_model_creation()
    if model is None:
        return
        
    # Test 3: Loss computation
    if not test_loss_computation(model, device):
        return
        
    # Test 4: Dataloader
    dataloader = test_dataloader(dataset)
    if dataloader is None:
        return
        
    # Test 5: Full training step
    if not test_full_training_step(model, dataloader, device):
        return
        
    print("\n🎉 All tests passed! PETR training pipeline is working correctly.")


if __name__ == "__main__":
    main()