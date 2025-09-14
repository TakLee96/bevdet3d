import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Tuple, Optional


def compute_3d_iou_jax(pred_boxes, gt_boxes, threshold=0.5):
    """
    Compute 3D IoU between predicted and ground truth boxes in JAX.
    Simplified version using center distance.
    
    Args:
        pred_boxes: [N, 7] predicted boxes (x, y, z, l, w, h, rot)
        gt_boxes: [M, 7] ground truth boxes
        threshold: IoU threshold for positive matches
    
    Returns:
        iou_matrix: [N, M] IoU matrix
        matches: [N] boolean array indicating if each prediction has a match
    """
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return jnp.zeros((len(pred_boxes), len(gt_boxes))), jnp.zeros(len(pred_boxes), dtype=bool)
    
    # Extract centers and sizes
    pred_centers = pred_boxes[:, :3]  # [N, 3]
    pred_sizes = pred_boxes[:, 3:6]   # [N, 3]
    gt_centers = gt_boxes[:, :3]      # [M, 3]
    gt_sizes = gt_boxes[:, 3:6]       # [M, 3]
    
    # Compute pairwise distances between centers
    pred_centers_expanded = pred_centers[:, None, :]  # [N, 1, 3]
    gt_centers_expanded = gt_centers[None, :, :]      # [1, M, 3]
    center_dist = jnp.linalg.norm(pred_centers_expanded - gt_centers_expanded, axis=-1)  # [N, M]
    
    # Compute characteristic sizes (average of l, w, h)
    pred_char_size = jnp.mean(pred_sizes, axis=-1, keepdims=True)  # [N, 1]
    gt_char_size = jnp.mean(gt_sizes, axis=-1, keepdims=True)      # [M, 1]
    avg_char_size = (pred_char_size + gt_char_size.T) / 2         # [N, M]
    
    # Simple IoU approximation: higher overlap for smaller distances relative to size
    # IoU ≈ max(0, 1 - distance / characteristic_size)
    iou_matrix = jnp.maximum(0, 1 - center_dist / (avg_char_size + 1e-8))
    
    # Find matches above threshold
    matches = jnp.max(iou_matrix, axis=1) > threshold
    
    return iou_matrix, matches


def compute_ap_jax(pred_scores, pred_matches, num_gt):
    """
    Compute Average Precision (AP) in JAX.
    
    Args:
        pred_scores: [N] prediction confidence scores
        pred_matches: [N] boolean array indicating true positives
        num_gt: number of ground truth boxes
    
    Returns:
        ap: Average Precision score
    """
    if num_gt == 0:
        return 0.0
    
    if len(pred_scores) == 0:
        return 0.0
    
    # Sort by confidence scores (descending)
    sorted_indices = jnp.argsort(-pred_scores)
    sorted_matches = pred_matches[sorted_indices]
    
    # Compute precision and recall
    tp = jnp.cumsum(sorted_matches.astype(jnp.float32))
    fp = jnp.cumsum(1 - sorted_matches.astype(jnp.float32))
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / num_gt
    
    # Compute AP using 11-point interpolation
    recall_thresholds = jnp.linspace(0, 1, 11)
    ap = 0.0
    
    for r_thresh in recall_thresholds:
        # Find precisions at recalls >= r_thresh
        valid_mask = recall >= r_thresh
        if jnp.any(valid_mask):
            max_precision = jnp.max(jnp.where(valid_mask, precision, 0))
            ap += max_precision
    
    return ap / 11.0


def compute_map_3d_jax(predictions: Dict, targets: Dict, 
                       confidence_threshold: float = 0.1,
                       iou_threshold: float = 0.5,
                       num_classes: int = 3) -> Dict:
    """
    Compute mean Average Precision (mAP) for 3D object detection in JAX.
    
    Args:
        predictions: Dict with 'cls_logits' [B, N, num_classes+1] and 'bbox_preds' [B, N, 7]
        targets: Dict with 'labels' [B, M] and 'boxes_3d' [B, M, 7]
        confidence_threshold: Minimum confidence for predictions
        iou_threshold: IoU threshold for positive matches
        num_classes: Number of object classes
    
    Returns:
        Dict with mAP and per-class AP scores
    """
    batch_size = predictions['cls_logits'].shape[0]
    
    # Convert logits to probabilities
    pred_probs = jax.nn.softmax(predictions['cls_logits'], axis=-1)  # [B, N, num_classes+1]
    pred_boxes = predictions['bbox_preds']  # [B, N, 7]
    
    target_labels = targets['labels']    # [B, M]
    target_boxes = targets['boxes_3d']   # [B, M, 7]
    
    # Collect predictions and targets for each class
    class_predictions = {i: {'scores': [], 'boxes': [], 'matches': []} for i in range(num_classes)}
    class_num_gt = {i: 0 for i in range(num_classes)}
    
    for batch_idx in range(batch_size):
        # Get valid targets (exclude background class)
        valid_target_mask = target_labels[batch_idx] < num_classes
        valid_target_labels = target_labels[batch_idx][valid_target_mask]
        valid_target_boxes = target_boxes[batch_idx][valid_target_mask]
        
        # Count ground truth for each class
        for class_id in range(num_classes):
            class_num_gt[class_id] += jnp.sum(valid_target_labels == class_id)
        
        # Process predictions
        for class_id in range(num_classes):
            # Get predictions for this class
            class_scores = pred_probs[batch_idx, :, class_id]  # [N]
            
            # Filter by confidence threshold
            confident_mask = class_scores >= confidence_threshold
            if not jnp.any(confident_mask):
                continue
            
            confident_scores = class_scores[confident_mask]
            confident_boxes = pred_boxes[batch_idx][confident_mask]
            
            # Get ground truth for this class
            gt_class_mask = valid_target_labels == class_id
            if not jnp.any(gt_class_mask):
                # No ground truth for this class, all predictions are false positives
                class_predictions[class_id]['scores'].extend(confident_scores.tolist())
                class_predictions[class_id]['boxes'].extend(confident_boxes.tolist())
                class_predictions[class_id]['matches'].extend([False] * len(confident_scores))
                continue
            
            gt_class_boxes = valid_target_boxes[gt_class_mask]
            
            # Compute IoU and matches
            iou_matrix, matches = compute_3d_iou_jax(
                confident_boxes, gt_class_boxes, iou_threshold
            )
            
            # Store results
            class_predictions[class_id]['scores'].extend(confident_scores.tolist())
            class_predictions[class_id]['boxes'].extend(confident_boxes.tolist())
            class_predictions[class_id]['matches'].extend(matches.tolist())
    
    # Compute AP for each class
    class_aps = {}
    for class_id in range(num_classes):
        if len(class_predictions[class_id]['scores']) == 0:
            class_aps[class_id] = 0.0
        else:
            scores = jnp.array(class_predictions[class_id]['scores'])
            matches = jnp.array(class_predictions[class_id]['matches'])
            num_gt = class_num_gt[class_id]
            
            class_aps[class_id] = compute_ap_jax(scores, matches, num_gt)
    
    # Compute mAP
    mAP = jnp.mean(jnp.array(list(class_aps.values())))
    
    return {
        'mAP': float(mAP),
        'AP_vehicle': float(class_aps[0]),
        'AP_pedestrian': float(class_aps[1]), 
        'AP_cyclist': float(class_aps[2]),
        'num_predictions': sum(len(class_predictions[i]['scores']) for i in range(num_classes)),
        'num_gt_total': sum(class_num_gt.values())
    }


def evaluate_model_jax(model, params, dataset, num_batches: int = 10):
    """
    Evaluate model on dataset and compute metrics.
    
    Args:
        model: JAX model
        params: Model parameters
        dataset: Dataset with data_loader method
        num_batches: Number of batches to evaluate
    
    Returns:
        Dict with evaluation metrics
    """
    all_predictions = {'cls_logits': [], 'bbox_preds': []}
    all_targets = {'labels': [], 'boxes_3d': []}
    
    # Collect predictions and targets
    for batch_idx, batch in enumerate(dataset.data_loader(batch_size=2, shuffle=False)):
        if batch_idx >= num_batches:
            break
        
        # Forward pass
        outputs = model.apply(params, batch['images'], training=False)
        
        # Store results
        all_predictions['cls_logits'].append(outputs['cls_logits'])
        all_predictions['bbox_preds'].append(outputs['bbox_preds'])
        all_targets['labels'].append(batch['labels'])
        all_targets['boxes_3d'].append(batch['boxes_3d'])
    
    # Concatenate all batches
    predictions = {
        'cls_logits': jnp.concatenate(all_predictions['cls_logits'], axis=0),
        'bbox_preds': jnp.concatenate(all_predictions['bbox_preds'], axis=0)
    }
    targets = {
        'labels': jnp.concatenate(all_targets['labels'], axis=0),
        'boxes_3d': jnp.concatenate(all_targets['boxes_3d'], axis=0)
    }
    
    # Compute metrics
    metrics = compute_map_3d_jax(predictions, targets)
    
    return metrics


if __name__ == "__main__":
    # Test the JAX metrics
    import jax.random as jrandom
    
    key = jrandom.PRNGKey(42)
    batch_size = 2
    num_queries = 10
    num_classes = 3
    
    # Create fake predictions
    key, subkey1, subkey2 = jrandom.split(key, 3)
    predictions = {
        'cls_logits': jrandom.normal(subkey1, (batch_size, num_queries, num_classes + 1)),
        'bbox_preds': jrandom.normal(subkey2, (batch_size, num_queries, 7)) * 10  # Larger boxes
    }
    
    # Create fake targets
    targets = {
        'labels': jnp.array([[0, 1, 2, 3, 3], [1, 0, 3, 3, 3]]),  # 3 = background
        'boxes_3d': jrandom.normal(key, (batch_size, 5, 7)) * 5
    }
    
    # Test metrics computation
    metrics = compute_map_3d_jax(predictions, targets)
    
    print("JAX metrics test successful!")
    print(f"mAP: {metrics['mAP']:.4f}")
    print(f"AP Vehicle: {metrics['AP_vehicle']:.4f}")
    print(f"AP Pedestrian: {metrics['AP_pedestrian']:.4f}")
    print(f"AP Cyclist: {metrics['AP_cyclist']:.4f}")
    print(f"Num predictions: {metrics['num_predictions']}")
    print(f"Num ground truth: {metrics['num_gt_total']}")
    
    print("✓ JAX metrics test passed!")