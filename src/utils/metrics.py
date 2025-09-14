import torch
import numpy as np
from typing import Dict, List, Tuple, Optional


def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """
    Compute average precision from precision-recall curve.
    
    Args:
        recalls: Array of recall values
        precisions: Array of precision values
    Returns:
        Average precision
    """
    # Add endpoints
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))
    
    # Compute precision envelope
    for i in range(precisions.size - 1, 0, -1):
        precisions[i - 1] = max(precisions[i - 1], precisions[i])
    
    # Find points where recall changes
    indices = np.where(recalls[1:] != recalls[:-1])[0]
    
    # Compute AP as area under curve
    ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])
    
    return ap


def compute_precision_recall(scores: np.ndarray, 
                           labels: np.ndarray, 
                           num_positives: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute precision and recall curves.
    
    Args:
        scores: Prediction scores
        labels: Binary labels (1 for positive, 0 for negative)
        num_positives: Total number of positive samples
    Returns:
        Tuple of (precisions, recalls)
    """
    # Sort by score
    sorted_indices = np.argsort(scores)[::-1]
    sorted_labels = labels[sorted_indices]
    
    # Compute cumulative counts
    tp = np.cumsum(sorted_labels)
    fp = np.cumsum(1 - sorted_labels)
    
    # Compute precision and recall
    precisions = tp / (tp + fp + 1e-8)
    recalls = tp / (num_positives + 1e-8)
    
    return precisions, recalls


def box_iou_3d(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Simplified 3D IoU computation based on center distance.
    
    Args:
        boxes1: [N, 7] tensor (x, y, z, l, w, h, rot)
        boxes2: [M, 7] tensor (x, y, z, l, w, h, rot)
    Returns:
        IoU matrix [N, M]
    """
    # Simplified IoU - use center distance as proxy
    centers1 = boxes1[:, :3]  # [N, 3]
    centers2 = boxes2[:, :3]  # [M, 3]
    
    # Compute pairwise distances
    dist = torch.cdist(centers1, centers2)  # [N, M]
    
    # Convert to similarity (IoU-like)
    max_dist = dist.max()
    iou = (max_dist - dist) / (max_dist + 1e-8)
    
    return iou


def compute_metrics(predictions: List[Dict], 
                   targets: List[Dict],
                   num_classes: int = 3,
                   iou_threshold: float = 0.5) -> Dict[str, float]:
    """
    Compute detection metrics (mAP).
    
    Args:
        predictions: List of prediction dicts with 'scores', 'labels', 'boxes_3d'
        targets: List of target dicts with 'labels', 'boxes_3d'  
        num_classes: Number of object classes
        iou_threshold: IoU threshold for positive detection
    Returns:
        Dict with metrics
    """
    aps = []
    
    for class_id in range(num_classes):
        # Collect predictions and targets for this class
        pred_scores = []
        pred_boxes = []
        gt_boxes = []
        
        for pred, target in zip(predictions, targets):
            # Get predictions for this class
            class_mask = pred['labels'] == class_id
            if class_mask.sum() > 0:
                pred_scores.append(pred['scores'][class_mask])
                pred_boxes.append(pred['boxes_3d'][class_mask])
            
            # Get ground truth for this class  
            gt_class_mask = target['labels'] == class_id
            if gt_class_mask.sum() > 0:
                gt_boxes.append(target['boxes_3d'][gt_class_mask])
        
        if len(pred_scores) == 0 or len(gt_boxes) == 0:
            aps.append(0.0)
            continue
            
        # Concatenate all predictions and targets
        all_pred_scores = torch.cat(pred_scores)
        all_pred_boxes = torch.cat(pred_boxes)
        all_gt_boxes = torch.cat(gt_boxes)
        
        num_gt = len(all_gt_boxes)
        
        # Compute IoU between all predictions and ground truth
        if len(all_pred_boxes) > 0 and len(all_gt_boxes) > 0:
            ious = box_iou_3d(all_pred_boxes, all_gt_boxes)  # [N_pred, N_gt]
            
            # For each prediction, find best matching ground truth
            max_ious, _ = ious.max(dim=1)  # [N_pred]
            labels = (max_ious >= iou_threshold).float()
            
            # Compute precision-recall curve
            scores_np = all_pred_scores.cpu().numpy()
            labels_np = labels.cpu().numpy()
            
            precisions, recalls = compute_precision_recall(scores_np, labels_np, num_gt)
            ap = compute_ap(recalls, precisions)
        else:
            ap = 0.0
            
        aps.append(ap)
    
    # Compute mean AP
    map_score = np.mean(aps)
    
    metrics = {
        'mAP': map_score,
        'mAP_50': map_score,  # Since we only use one IoU threshold
    }
    
    # Add per-class APs
    class_names = ['Vehicle', 'Pedestrian', 'Cyclist']
    for i, class_name in enumerate(class_names[:num_classes]):
        metrics[f'AP_{class_name}'] = aps[i] if i < len(aps) else 0.0
    
    return metrics


def filter_predictions(outputs: Dict, 
                      score_threshold: float = 0.1,
                      max_detections: int = 100) -> List[Dict]:
    """
    Filter and convert model outputs to predictions.
    
    Args:
        outputs: Model outputs with 'cls_logits' and 'bbox_preds'
        score_threshold: Score threshold for filtering
        max_detections: Maximum number of detections per image
    Returns:
        List of prediction dicts
    """
    batch_size = outputs['cls_logits'].shape[0]
    predictions = []
    
    for i in range(batch_size):
        cls_logits = outputs['cls_logits'][i]  # [N, num_classes+1]
        bbox_preds = outputs['bbox_preds'][i]  # [N, 7]
        
        # Convert logits to probabilities and get scores
        probs = torch.softmax(cls_logits, dim=-1)
        scores, labels = probs[:, :-1].max(dim=-1)  # Exclude background class
        
        # Filter by score threshold
        keep = scores >= score_threshold
        scores = scores[keep]
        labels = labels[keep]
        bbox_preds = bbox_preds[keep]
        
        # Keep top detections
        if len(scores) > max_detections:
            _, top_indices = scores.topk(max_detections)
            scores = scores[top_indices]
            labels = labels[top_indices]
            bbox_preds = bbox_preds[top_indices]
        
        predictions.append({
            'scores': scores,
            'labels': labels,
            'boxes_3d': bbox_preds
        })
    
    return predictions


if __name__ == "__main__":
    # Test metrics computation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create dummy predictions and targets
    batch_size = 2
    num_classes = 3
    
    # Dummy model outputs
    outputs = {
        'cls_logits': torch.randn(batch_size, 100, num_classes + 1).to(device),
        'bbox_preds': torch.randn(batch_size, 100, 7).to(device),
    }
    
    # Convert to predictions
    predictions = filter_predictions(outputs, score_threshold=0.1)
    
    # Create dummy targets
    targets = []
    for i in range(batch_size):
        num_targets = torch.randint(1, 5, (1,)).item()
        targets.append({
            'labels': torch.randint(0, num_classes, (num_targets,)).to(device),
            'boxes_3d': torch.randn(num_targets, 7).to(device)
        })
    
    # Compute metrics
    metrics = compute_metrics(predictions, targets, num_classes=num_classes)
    
    print("Metrics computation test successful!")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
        
    print(f"Number of predictions: {[len(p['scores']) for p in predictions]}")
    print(f"Number of targets: {[len(t['labels']) for t in targets]}")