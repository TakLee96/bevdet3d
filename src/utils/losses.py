import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import Dict, List, Tuple, Optional


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU for 3D boxes (simplified version).
    
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
    dist = torch.cdist(centers1, centers2)  # [N, M]
    
    # Convert distance to similarity (higher is better)
    max_dist = dist.max()
    similarity = (max_dist - dist) / (max_dist + 1e-8)
    
    return similarity


class HungarianMatcher(nn.Module):
    """Hungarian matcher for bipartite matching between predictions and ground truth."""
    
    def __init__(self, 
                 cost_class: float = 1.0,
                 cost_bbox: float = 5.0,
                 cost_giou: float = 2.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        
    @torch.no_grad()
    def forward(self, outputs: Dict, targets: List[Dict]) -> List[Tuple]:
        """
        Perform bipartite matching.
        
        Args:
            outputs: Dict with 'cls_logits' [B, N, num_classes+1] and 'bbox_preds' [B, N, 7]
            targets: List of target dicts with 'labels' and 'boxes_3d'
        Returns:
            List of (src_idx, tgt_idx) tuples for each batch element
        """
        batch_size, num_queries = outputs['cls_logits'].shape[:2]
        
        # Flatten predictions
        out_prob = outputs['cls_logits'].flatten(0, 1).sigmoid()  # [B*N, num_classes+1]
        out_bbox = outputs['bbox_preds'].flatten(0, 1)  # [B*N, 7]
        
        # Concatenate targets
        tgt_ids = torch.cat([v['labels'] for v in targets])  # [total_targets]
        tgt_bbox = torch.cat([v['boxes_3d'] for v in targets])  # [total_targets, 7]
        
        # Classification cost
        cost_class = -out_prob[:, tgt_ids]  # [B*N, total_targets]
        
        # L1 cost for bbox
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)  # [B*N, total_targets]
        
        # GIoU cost
        cost_giou = -generalized_box_iou(out_bbox, tgt_bbox)  # [B*N, total_targets]
        
        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(batch_size, num_queries, -1).cpu()
        
        # Perform Hungarian matching for each batch element
        indices = []
        start_idx = 0
        for i, targets_i in enumerate(targets):
            num_targets = len(targets_i['labels'])
            if num_targets == 0:
                indices.append((torch.tensor([]), torch.tensor([])))
                continue
                
            cost_matrix = C[i, :, start_idx:start_idx + num_targets]
            src_idx, tgt_idx = linear_sum_assignment(cost_matrix)
            
            indices.append((torch.tensor(src_idx), torch.tensor(tgt_idx)))
            start_idx += num_targets
            
        return indices


class SetCriterion(nn.Module):
    """DETR-style set criterion with classification and regression losses."""
    
    def __init__(self, 
                 num_classes: int,
                 matcher: HungarianMatcher,
                 weight_dict: Dict[str, float],
                 eos_coef: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        
        # Background class weight
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)
        
    def loss_labels(self, outputs: Dict, targets: List[Dict], indices: List[Tuple]) -> Dict:
        """Classification loss."""
        assert 'cls_logits' in outputs
        src_logits = outputs['cls_logits']  # [B, N, num_classes+1]
        
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                   dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}
        
        return losses
        
    def loss_boxes(self, outputs: Dict, targets: List[Dict], indices: List[Tuple]) -> Dict:
        """Regression loss for 3D boxes."""
        assert 'bbox_preds' in outputs
        
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['bbox_preds'][idx]  # [num_matched, 7]
        target_boxes = torch.cat([t['boxes_3d'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        
        # L1 loss
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        
        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / len(src_boxes) if len(src_boxes) > 0 else torch.tensor(0.0, device=src_boxes.device)
        
        return losses
        
    def _get_src_permutation_idx(self, indices: List[Tuple]) -> Tuple:
        """Get source permutation indices."""
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx
        
    def forward(self, outputs: Dict, targets: List[Dict]) -> Dict:
        """
        Compute losses.
        
        Args:
            outputs: Dict with model outputs
            targets: List of target dicts
        Returns:
            Dict with losses
        """
        # Get matching indices
        indices = self.matcher(outputs, targets)
        
        # Compute losses
        losses = {}
        losses.update(self.loss_labels(outputs, targets, indices))
        losses.update(self.loss_boxes(outputs, targets, indices))
        
        # Apply weights
        for k in losses.keys():
            if k in self.weight_dict:
                losses[k] *= self.weight_dict[k]
                
        return losses


class PETRLoss(nn.Module):
    """PETR loss function wrapper."""
    
    def __init__(self, 
                 num_classes: int = 3,
                 weight_dict: Optional[Dict[str, float]] = None):
        super().__init__()
        
        if weight_dict is None:
            weight_dict = {
                'loss_ce': 1.0,
                'loss_bbox': 5.0,
            }
            
        self.matcher = HungarianMatcher(
            cost_class=1.0,
            cost_bbox=5.0, 
            cost_giou=2.0
        )
        
        self.criterion = SetCriterion(
            num_classes=num_classes,
            matcher=self.matcher,
            weight_dict=weight_dict,
            eos_coef=0.1
        )
        
    def forward(self, outputs: Dict, targets: List[Dict]) -> Dict:
        """
        Compute PETR losses.
        
        Args:
            outputs: Model outputs with 'cls_logits' and 'bbox_preds'
            targets: List of target dicts with 'labels' and 'boxes_3d'
        Returns:
            Dict with losses
        """
        return self.criterion(outputs, targets)


if __name__ == "__main__":
    # Test the loss function
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create dummy data
    batch_size = 2
    num_queries = 100
    num_classes = 3
    
    # Dummy outputs (require grad for testing backward pass)
    outputs = {
        'cls_logits': torch.randn(batch_size, num_queries, num_classes + 1, requires_grad=True).to(device),
        'bbox_preds': torch.randn(batch_size, num_queries, 7, requires_grad=True).to(device),
    }
    
    # Dummy targets
    targets = []
    for i in range(batch_size):
        num_targets = torch.randint(1, 10, (1,)).item()
        targets.append({
            'labels': torch.randint(0, num_classes, (num_targets,)).to(device),
            'boxes_3d': torch.randn(num_targets, 7).to(device)
        })
    
    # Test loss computation
    loss_fn = PETRLoss(num_classes=num_classes).to(device)
    losses = loss_fn(outputs, targets)
    
    print("Loss computation test successful!")
    for k, v in losses.items():
        print(f"{k}: {v.item():.4f}")
        
    # Test backward pass
    total_loss = sum(losses.values())
    total_loss.backward()
    print("Backward pass successful!")