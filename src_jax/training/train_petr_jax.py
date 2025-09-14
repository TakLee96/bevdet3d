#!/usr/bin/env python3

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import numpy as np
import os
import sys
import argparse
from functools import partial
from typing import Dict, Any
import pickle
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src_jax.models.petr_jax import create_petr_model
from src_jax.data.waymo_dataset_jax import create_waymo_dataset_jax
from src_jax.utils.losses_jax import create_petr_loss
from src_jax.utils.metrics_jax import compute_map_3d_jax


class TrainingState:
    """Training state for JAX training."""
    
    def __init__(self, params, optimizer_state, step=0):
        self.params = params
        self.optimizer_state = optimizer_state
        self.step = step


def create_train_step(model, loss_fn, optimizer):
    """Create JAX training step function."""
    
    @jax.jit
    def train_step(state: TrainingState, batch: Dict) -> tuple:
        """Execute one training step."""
        
        def loss_function(params):
            # Forward pass
            outputs = model.apply(params, batch['images'], training=True)
            
            # Prepare targets
            targets = {
                'labels': batch['labels'],
                'boxes_3d': batch['boxes_3d']
            }
            
            # Compute loss
            loss_dict = loss_fn(outputs, targets)
            return loss_dict['total_loss'], (loss_dict, outputs)
        
        # Compute gradients
        (loss, (loss_dict, outputs)), grads = jax.value_and_grad(
            loss_function, has_aux=True)(state.params)
        
        # Update parameters
        updates, new_optimizer_state = optimizer.update(
            grads, state.optimizer_state, state.params)
        new_params = optax.apply_updates(state.params, updates)
        
        # Create new state
        new_state = TrainingState(
            params=new_params,
            optimizer_state=new_optimizer_state,
            step=state.step + 1
        )
        
        return new_state, loss_dict, outputs
    
    return train_step


def create_eval_step(model, loss_fn):
    """Create JAX evaluation step function."""
    
    @jax.jit
    def eval_step(params, batch: Dict) -> tuple:
        """Execute one evaluation step."""
        # Forward pass
        outputs = model.apply(params, batch['images'], training=False)
        
        # Prepare targets
        targets = {
            'labels': batch['labels'],
            'boxes_3d': batch['boxes_3d']
        }
        
        # Compute loss
        loss_dict = loss_fn(outputs, targets)
        
        return loss_dict, outputs
    
    return eval_step


def train_petr_jax(args):
    """Main training function for PETR in JAX."""
    
    print("Starting PETR JAX training...")
    print(f"Using JAX backend: {jax.lib.xla_bridge.get_backend().platform}")
    
    # Create dataset
    print("Creating dataset...")
    dataset = create_waymo_dataset_jax(
        data_root=args.data_root,
        split='training',
        max_files=args.max_files,
        max_frames_per_file=args.max_frames_per_file,
        image_size=(args.image_width, args.image_height),
        cache_dir=args.cache_dir
    )
    
    print(f"Dataset created with {len(dataset)} samples")
    
    # Create model
    print("Creating PETR model...")
    model = create_petr_model(
        embed_dims=args.embed_dims,
        num_queries=args.num_queries,
        num_classes=args.num_classes,
        num_cameras=5
    )
    
    # Initialize model parameters
    key = jax.random.PRNGKey(42)
    sample_batch = dataset.get_batch(1)
    params = model.init(key, sample_batch['images'], training=True)
    
    # Count parameters
    param_count = sum(x.size for x in jax.tree_leaves(params))
    print(f"Model initialized with {param_count:,} parameters")
    
    # Create optimizer
    learning_rate = optax.cosine_decay_schedule(
        init_value=args.learning_rate,
        decay_steps=args.num_epochs * (len(dataset) // args.batch_size),
        alpha=0.1
    )
    
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),  # Gradient clipping
        optax.adamw(learning_rate=learning_rate, weight_decay=args.weight_decay)
    )
    
    optimizer_state = optimizer.init(params)
    
    # Create loss function
    loss_fn = create_petr_loss(
        cost_class=1.0,
        cost_bbox=5.0,
        cost_giou=2.0,
        loss_class=1.0,
        loss_bbox=5.0,
        loss_giou=2.0,
        num_classes=args.num_classes
    )
    
    # Create training and evaluation steps
    train_step = create_train_step(model, loss_fn, optimizer)
    eval_step = create_eval_step(model, loss_fn)
    
    # Initialize training state
    state = TrainingState(
        params=params,
        optimizer_state=optimizer_state,
        step=0
    )
    
    # Training loop
    print(f"Starting training for {args.num_epochs} epochs...")
    
    best_loss = float('inf')
    
    for epoch in range(args.num_epochs):
        epoch_start_time = time.time()
        epoch_losses = []
        
        # Training
        for batch_idx, batch in enumerate(dataset.data_loader(
            batch_size=args.batch_size, shuffle=True)):
            
            # Training step
            state, loss_dict, outputs = train_step(state, batch)
            epoch_losses.append(loss_dict['total_loss'])
            
            # Log progress
            if batch_idx % args.log_interval == 0:
                print(f"Epoch {epoch+1}/{args.num_epochs}, "
                      f"Batch {batch_idx}, "
                      f"Loss: {loss_dict['total_loss']:.4f} "
                      f"(CE: {loss_dict['cls_loss']:.4f}, "
                      f"BBox: {loss_dict['bbox_loss']:.4f}, "
                      f"GIoU: {loss_dict['giou_loss']:.4f})")
        
        # Compute epoch statistics
        avg_loss = jnp.mean(jnp.array(epoch_losses))
        epoch_time = time.time() - epoch_start_time
        
        print(f"Epoch {epoch+1} completed in {epoch_time:.2f}s, "
              f"Average loss: {avg_loss:.4f}")
        
        # Evaluation
        if (epoch + 1) % args.eval_interval == 0:
            print("Running evaluation...")
            eval_losses = []
            eval_outputs = []
            eval_targets = []
            
            for batch_idx, batch in enumerate(dataset.data_loader(
                batch_size=args.batch_size, shuffle=False)):
                
                if batch_idx >= args.max_eval_batches:
                    break
                
                loss_dict, outputs = eval_step(state.params, batch)
                eval_losses.append(loss_dict['total_loss'])
                
                # Collect for metrics
                eval_outputs.append(outputs)
                eval_targets.append({
                    'labels': batch['labels'],
                    'boxes_3d': batch['boxes_3d']
                })
            
            # Compute evaluation metrics
            avg_eval_loss = jnp.mean(jnp.array(eval_losses))
            
            # Compute mAP
            if len(eval_outputs) > 0:
                # Concatenate all evaluation results
                all_outputs = {
                    'cls_logits': jnp.concatenate([o['cls_logits'] for o in eval_outputs]),
                    'bbox_preds': jnp.concatenate([o['bbox_preds'] for o in eval_outputs])
                }
                all_targets = {
                    'labels': jnp.concatenate([t['labels'] for t in eval_targets]),
                    'boxes_3d': jnp.concatenate([t['boxes_3d'] for t in eval_targets])
                }
                
                metrics = compute_map_3d_jax(all_outputs, all_targets)
                
                print(f"Evaluation - Loss: {avg_eval_loss:.4f}, "
                      f"mAP: {metrics['mAP']:.4f}, "
                      f"AP_vehicle: {metrics['AP_vehicle']:.4f}, "
                      f"AP_pedestrian: {metrics['AP_pedestrian']:.4f}, "
                      f"AP_cyclist: {metrics['AP_cyclist']:.4f}")
        
        # Save checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint_path = os.path.join(args.output_dir, 'best_model_jax.pkl')
            os.makedirs(args.output_dir, exist_ok=True)
            
            with open(checkpoint_path, 'wb') as f:
                pickle.dump({
                    'params': state.params,
                    'optimizer_state': state.optimizer_state,
                    'step': state.step,
                    'loss': avg_loss,
                    'epoch': epoch
                }, f)
            
            print(f"Saved best model checkpoint to {checkpoint_path}")
    
    print("Training completed!")
    return state.params


def main():
    parser = argparse.ArgumentParser(description='Train PETR model with JAX/Flax')
    
    # Data parameters
    parser.add_argument('--data_root', type=str, default='waymo_open_dataset_v_1_4_3',
                        help='Path to Waymo dataset')
    parser.add_argument('--cache_dir', type=str, default='./cache',
                        help='Directory for cached data')
    parser.add_argument('--max_files', type=int, default=2,
                        help='Maximum number of tfrecord files')
    parser.add_argument('--max_frames_per_file', type=int, default=3,
                        help='Maximum frames per file')
    
    # Model parameters
    parser.add_argument('--embed_dims', type=int, default=256,
                        help='Embedding dimensions')
    parser.add_argument('--num_queries', type=int, default=50,
                        help='Number of object queries')
    parser.add_argument('--num_classes', type=int, default=3,
                        help='Number of object classes')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    
    # Image parameters
    parser.add_argument('--image_height', type=int, default=200,
                        help='Image height')
    parser.add_argument('--image_width', type=int, default=400,
                        help='Image width')
    
    # Logging parameters
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Logging interval')
    parser.add_argument('--eval_interval', type=int, default=2,
                        help='Evaluation interval (epochs)')
    parser.add_argument('--max_eval_batches', type=int, default=5,
                        help='Maximum evaluation batches')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='./outputs_jax',
                        help='Output directory')
    
    args = parser.parse_args()
    
    # Run training
    train_petr_jax(args)


if __name__ == "__main__":
    main()