#!/usr/bin/env python3

import os
import sys
import argparse
import logging
import json
from datetime import datetime

# Suppress TensorFlow warnings before importing
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data.simple_waymo_dataset import SimpleCachedWaymoDataset, collate_fn
from models.bevformer import BEVFormerModel
from utils.losses import PETRLoss  # Reuse PETR loss for BEVFormer
from utils.metrics import compute_metrics, filter_predictions


def setup_logging(log_dir):
    """Setup logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'train.log')),
            logging.StreamHandler()
        ]
    )


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


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, writer=None):
    """Train for one epoch."""
    model.train()
    
    running_loss = 0.0
    num_batches = len(dataloader)
    prev_bev = None  # For temporal modeling
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for batch_idx, batch in enumerate(pbar):
        # Move data to device
        images = batch['images'].to(device)  # [B, N_cams, 3, H, W]
        targets = prepare_targets(batch)
        
        # Move targets to device
        for target in targets:
            for k, v in target.items():
                target[k] = v.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass (with temporal modeling)
        outputs = model(images, prev_bev=prev_bev)
        
        # Store BEV features for next iteration (detach to avoid gradient accumulation)
        prev_bev = outputs['bev_features'].detach() if 'bev_features' in outputs else None
        
        # Compute loss
        losses = criterion(outputs, targets)
        total_loss = sum(losses.values())
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        
        # Update weights
        optimizer.step()
        
        # Update running loss
        running_loss += total_loss.item()
        
        # Log to tensorboard
        if writer is not None:
            global_step = epoch * num_batches + batch_idx
            writer.add_scalar('Loss/Total', total_loss.item(), global_step)
            for k, v in losses.items():
                writer.add_scalar(f'Loss/{k}', v.item(), global_step)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{total_loss.item():.4f}',
            'avg_loss': f'{running_loss / (batch_idx + 1):.4f}'
        })
    
    return running_loss / num_batches


def validate_epoch(model, dataloader, criterion, device, epoch, writer=None):
    """Validate for one epoch."""
    model.eval()
    
    running_loss = 0.0
    all_predictions = []
    all_targets = []
    prev_bev = None
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f'Val Epoch {epoch}')
        
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            images = batch['images'].to(device)
            targets = prepare_targets(batch)
            
            # Move targets to device
            for target in targets:
                for k, v in target.items():
                    target[k] = v.to(device)
            
            # Forward pass
            outputs = model(images, prev_bev=prev_bev)
            prev_bev = outputs['bev_features'].detach() if 'bev_features' in outputs else None
            
            # Compute loss
            losses = criterion(outputs, targets)
            total_loss = sum(losses.values())
            
            running_loss += total_loss.item()
            
            # Convert outputs to predictions for metrics
            predictions = filter_predictions(outputs, score_threshold=0.1)
            all_predictions.extend(predictions)
            all_targets.extend(targets)
            
            pbar.set_postfix({'val_loss': f'{total_loss.item():.4f}'})
    
    # Compute metrics
    metrics = compute_metrics(all_predictions, all_targets, num_classes=3)
    
    avg_loss = running_loss / len(dataloader)
    
    # Log metrics
    if writer is not None:
        writer.add_scalar('Val/Loss', avg_loss, epoch)
        for k, v in metrics.items():
            writer.add_scalar(f'Val/{k}', v, epoch)
    
    return avg_loss, metrics


def save_checkpoint(model, optimizer, epoch, loss, metrics, checkpoint_dir):
    """Save model checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'metrics': metrics,
    }
    
    # Save latest checkpoint
    torch.save(checkpoint, os.path.join(checkpoint_dir, 'latest.pth'))
    
    # Save best checkpoint (based on mAP)
    best_path = os.path.join(checkpoint_dir, 'best.pth')
    if not os.path.exists(best_path):
        torch.save(checkpoint, best_path)
    else:
        best_checkpoint = torch.load(best_path)
        if metrics['mAP'] > best_checkpoint['metrics']['mAP']:
            torch.save(checkpoint, best_path)
            logging.info(f"New best model saved with mAP: {metrics['mAP']:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Train BEVFormer on Waymo Dataset')
    parser.add_argument('--data_root', type=str, required=True,
                       help='Path to Waymo dataset root')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='Output directory for logs and checkpoints')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=0,
                       help='Number of dataloader workers')
    parser.add_argument('--val_freq', type=int, default=5,
                       help='Validation frequency (epochs)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--max_files', type=int, default=3,
                       help='Maximum tfrecord files to use')
    parser.add_argument('--max_frames', type=int, default=5,
                       help='Maximum frames per tfrecord file')
    
    args = parser.parse_args()
    
    # Setup directories
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = os.path.join(args.output_dir, f'bevformer_waymo_{timestamp}')
    log_dir = os.path.join(exp_dir, 'logs')
    checkpoint_dir = os.path.join(exp_dir, 'checkpoints')
    
    # Setup logging
    setup_logging(log_dir)
    logging.info(f"Starting BEVFormer training experiment: {exp_dir}")
    logging.info(f"Arguments: {args}")
    
    # Save config
    config = vars(args)
    with open(os.path.join(exp_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    # Create datasets
    logging.info("Creating datasets...")
    train_dataset = SimpleCachedWaymoDataset(
        data_root=args.data_root,
        split='training',
        max_files=args.max_files,
        max_frames_per_file=args.max_frames,
        image_size=(400, 200),  # Smaller for faster training
        cache_dir=os.path.join(exp_dir, 'cache')
    )
    
    # Use same dataset for validation but smaller subset
    val_dataset = SimpleCachedWaymoDataset(
        data_root=args.data_root,
        split='training',
        max_files=2,
        max_frames_per_file=2,
        image_size=(400, 200),
        cache_dir=os.path.join(exp_dir, 'cache_val')
    )
    
    logging.info(f"Train dataset: {len(train_dataset)} samples")
    logging.info(f"Val dataset: {len(val_dataset)} samples")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Create model
    logging.info("Creating BEVFormer model...")
    model = BEVFormerModel(
        backbone_name='resnet50',
        embed_dims=128,      # Smaller for faster training
        num_queries=50,      # Fewer queries
        num_classes=3,       # vehicle, pedestrian, cyclist
        bev_h=25,           # Smaller BEV grid
        bev_w=25,
        num_cameras=5,
        num_encoder_layers=3  # Fewer layers
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Create loss function (reuse PETR loss)
    criterion = PETRLoss(num_classes=3).to(device)
    
    # Create optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Create scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    
    # Setup tensorboard
    writer = SummaryWriter(log_dir)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        logging.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
    
    # Training loop
    logging.info("Starting BEVFormer training...")
    best_map = 0.0
    
    for epoch in range(start_epoch, args.num_epochs):
        logging.info(f"Epoch {epoch}/{args.num_epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch, writer)
        
        # Update scheduler
        scheduler.step()
        
        logging.info(f"Train Loss: {train_loss:.4f}")
        
        # Validate
        if epoch % args.val_freq == 0:
            val_loss, metrics = validate_epoch(model, val_loader, criterion, device, epoch, writer)
            
            logging.info(f"Val Loss: {val_loss:.4f}")
            logging.info(f"Metrics: {metrics}")
            
            # Save checkpoint
            save_checkpoint(model, optimizer, epoch, val_loss, metrics, checkpoint_dir)
            
            if metrics['mAP'] > best_map:
                best_map = metrics['mAP']
                logging.info(f"New best mAP: {best_map:.4f}")
    
    logging.info("Training completed!")
    writer.close()


if __name__ == "__main__":
    main()