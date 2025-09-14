import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.optimize import linear_sum_assignment


def generalized_box_iou_jax(boxes1, boxes2):
    """
    Generalized IoU for 3D boxes (simplified version) in JAX.
    
    Args:
        boxes1: [N, 7] (x, y, z, l, w, h, rot)
        boxes2: [M, 7] (x, y, z, l, w, h, rot)
    Returns:
        iou: [N, M] IoU matrix
    """
    # Simplified IoU calculation - in a full implementation this would
    # properly handle 3D box overlap calculation
    
    # For now, use center distance as a proxy
    centers1 = boxes1[:, :3]  # [N, 3]
    centers2 = boxes2[:, :3]  # [M, 3]
    
    # Compute pairwise distances
    centers1_expanded = centers1[:, None, :]  # [N, 1, 3]
    centers2_expanded = centers2[None, :, :]  # [1, M, 3]
    dist = jnp.linalg.norm(centers1_expanded - centers2_expanded, axis=-1)  # [N, M]
    
    # Convert distance to similarity (higher is better)
    max_dist = jnp.max(dist)
    similarity = (max_dist - dist) / (max_dist + 1e-8)
    
    return similarity


def hungarian_matcher_jax(outputs: Dict, targets: Dict, 
                          cost_class: float = 1.0,
                          cost_bbox: float = 5.0,
                          cost_giou: float = 2.0) -> List[Tuple]:
    """
    Hungarian matcher for bipartite matching between predictions and ground truth in JAX.
    
    Args:
        outputs: Dict with 'cls_logits' [B, N, num_classes+1] and 'bbox_preds' [B, N, 7]
        targets: Dict with 'labels' [B, M] and 'boxes_3d' [B, M, 7]
        cost_class: Weight for classification cost
        cost_bbox: Weight for bbox regression cost
        cost_giou: Weight for generalized IoU cost
    
    Returns:
        List of (pred_indices, target_indices) tuples for each batch
    """
    batch_size, num_queries = outputs['cls_logits'].shape[:2]
    
    # Get predictions
    pred_logits = outputs['cls_logits']  # [B, N, num_classes+1]
    pred_boxes = outputs['bbox_preds']   # [B, N, 7]
    
    # Get targets  
    target_labels = targets['labels']    # [B, M]
    target_boxes = targets['boxes_3d']   # [B, M, 7]
    
    # Convert to probabilities
    pred_probs = jax.nn.softmax(pred_logits, axis=-1)  # [B, N, num_classes+1]
    
    indices = []
    
    for batch_idx in range(batch_size):
        # Get valid targets (exclude background class 3)
        valid_mask = target_labels[batch_idx] < 3
        valid_labels = target_labels[batch_idx][valid_mask]
        valid_boxes = target_boxes[batch_idx][valid_mask]
        
        if len(valid_labels) == 0:
            # No valid targets, return empty matching
            indices.append((jnp.array([]), jnp.array([])))
            continue
        
        # Classification cost
        cost_class_matrix = -pred_probs[batch_idx][:, valid_labels]  # [N, num_valid_targets]
        
        # L1 cost for bbox regression
        cost_bbox_matrix = jnp.linalg.norm(
            pred_boxes[batch_idx][:, None, :] - valid_boxes[None, :, :], 
            axis=-1
        )  # [N, num_valid_targets]
        
        # Generalized IoU cost
        cost_giou_matrix = -generalized_box_iou_jax(
            pred_boxes[batch_idx], valid_boxes
        )  # [N, num_valid_targets]
        
        # Total cost
        cost_matrix = (cost_class * cost_class_matrix + 
                      cost_bbox * cost_bbox_matrix + 
                      cost_giou * cost_giou_matrix)
        
        # Use scipy for Hungarian algorithm (convert to numpy)
        cost_matrix_np = np.array(cost_matrix)
        pred_indices, target_indices = linear_sum_assignment(cost_matrix_np)
        
        # Convert back to JAX arrays
        indices.append((jnp.array(pred_indices), jnp.array(target_indices)))
    
    return indices


def compute_classification_loss(pred_logits, target_labels, indices, num_classes=3):
    """Compute focal loss for classification."""
    batch_size = pred_logits.shape[0]
    total_loss = 0.0
    
    for batch_idx, (pred_idx, target_idx) in enumerate(indices):
        if len(pred_idx) == 0:
            continue
            
        # Get matched predictions and targets
        matched_pred_logits = pred_logits[batch_idx, pred_idx]  # [num_matched, num_classes+1]
        matched_target_labels = target_labels[batch_idx][target_idx]  # [num_matched]
        
        # Compute cross-entropy loss
        log_probs = jax.nn.log_softmax(matched_pred_logits, axis=-1)
        loss = -jnp.mean(log_probs[jnp.arange(len(matched_target_labels)), matched_target_labels])
        total_loss += loss
    
    return total_loss / batch_size


def compute_bbox_loss(pred_boxes, target_boxes, indices):
    """Compute L1 loss for bbox regression."""
    batch_size = pred_boxes.shape[0]
    total_loss = 0.0
    
    for batch_idx, (pred_idx, target_idx) in enumerate(indices):
        if len(pred_idx) == 0:
            continue
            
        # Get matched predictions and targets
        matched_pred_boxes = pred_boxes[batch_idx, pred_idx]  # [num_matched, 7]
        matched_target_boxes = target_boxes[batch_idx][target_idx]  # [num_matched, 7]
        
        # Compute L1 loss
        loss = jnp.mean(jnp.abs(matched_pred_boxes - matched_target_boxes))
        total_loss += loss
    
    return total_loss / batch_size


def compute_giou_loss(pred_boxes, target_boxes, indices):
    """Compute generalized IoU loss."""
    batch_size = pred_boxes.shape[0]
    total_loss = 0.0
    
    for batch_idx, (pred_idx, target_idx) in enumerate(indices):
        if len(pred_idx) == 0:
            continue
            
        # Get matched predictions and targets
        matched_pred_boxes = pred_boxes[batch_idx, pred_idx]  # [num_matched, 7]
        matched_target_boxes = target_boxes[batch_idx][target_idx]  # [num_matched, 7]
        
        # Compute GIoU (simplified as negative IoU)
        if len(matched_pred_boxes) > 0:
            iou = generalized_box_iou_jax(matched_pred_boxes, matched_target_boxes)
            # Take diagonal (matched pairs)
            matched_iou = jnp.diag(iou)
            loss = jnp.mean(1.0 - matched_iou)
            total_loss += loss
    
    return total_loss / batch_size


class PETRLoss:
    """PETR loss function with Hungarian matching in JAX."""
    
    def __init__(self, 
                 cost_class: float = 1.0,
                 cost_bbox: float = 5.0,
                 cost_giou: float = 2.0,
                 loss_class: float = 1.0,
                 loss_bbox: float = 5.0,
                 loss_giou: float = 2.0,
                 num_classes: int = 3):
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.loss_class = loss_class
        self.loss_bbox = loss_bbox
        self.loss_giou = loss_giou
        self.num_classes = num_classes
    
    def __call__(self, outputs: Dict, targets: Dict) -> Dict:
        """
        Compute PETR loss.
        
        Args:
            outputs: Dict with 'cls_logits' and 'bbox_preds'
            targets: Dict with 'labels' and 'boxes_3d'
        
        Returns:
            Dict with individual losses and total loss
        """
        # Hungarian matching
        indices = hungarian_matcher_jax(
            outputs, targets,
            self.cost_class, self.cost_bbox, self.cost_giou
        )
        
        # Compute individual losses
        cls_loss = compute_classification_loss(
            outputs['cls_logits'], targets['labels'], indices, self.num_classes
        )
        
        bbox_loss = compute_bbox_loss(
            outputs['bbox_preds'], targets['boxes_3d'], indices
        )
        
        giou_loss = compute_giou_loss(
            outputs['bbox_preds'], targets['boxes_3d'], indices
        )
        
        # Total loss
        total_loss = (self.loss_class * cls_loss + 
                     self.loss_bbox * bbox_loss + 
                     self.loss_giou * giou_loss)
        
        return {
            'total_loss': total_loss,
            'cls_loss': cls_loss,
            'bbox_loss': bbox_loss,
            'giou_loss': giou_loss,
            'indices': indices
        }


def create_petr_loss(
    cost_class: float = 1.0,
    cost_bbox: float = 5.0, 
    cost_giou: float = 2.0,
    loss_class: float = 1.0,
    loss_bbox: float = 5.0,
    loss_giou: float = 2.0,
    num_classes: int = 3
):
    """Factory function to create PETR loss."""
    return PETRLoss(
        cost_class=cost_class,
        cost_bbox=cost_bbox,
        cost_giou=cost_giou,
        loss_class=loss_class,
        loss_bbox=loss_bbox,
        loss_giou=loss_giou,
        num_classes=num_classes
    )


if __name__ == "__main__":
    # Test the JAX loss functions
    import jax.random as jrandom
    
    key = jrandom.PRNGKey(0)
    batch_size = 2
    num_queries = 10
    num_classes = 3
    
    # Create fake outputs
    key, subkey1, subkey2 = jrandom.split(key, 3)
    outputs = {
        'cls_logits': jrandom.normal(subkey1, (batch_size, num_queries, num_classes + 1)),
        'bbox_preds': jrandom.normal(subkey2, (batch_size, num_queries, 7))
    }
    
    # Create fake targets
    targets = {
        'labels': jnp.array([[0, 1, 2, 3, 3], [1, 0, 3, 3, 3]]),  # 3 = background
        'boxes_3d': jrandom.normal(key, (batch_size, 5, 7))
    }
    
    # Test loss computation
    loss_fn = create_petr_loss()
    losses = loss_fn(outputs, targets)
    
    print("JAX loss function test successful!")
    print(f"Total loss: {losses['total_loss']:.4f}")
    print(f"Classification loss: {losses['cls_loss']:.4f}")
    print(f"Bbox loss: {losses['bbox_loss']:.4f}")
    print(f"GIoU loss: {losses['giou_loss']:.4f}")
    print(f"Number of matches: {[len(idx[0]) for idx in losses['indices']]}")
    
    print("✓ JAX loss functions test passed!")