import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SupervisedContrastiveLoss(nn.Module):
    """Supervised contrastive loss over top-class labels.
    Pulls samples with the same top-level parent together in latent space.
    """
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        Args:
            features: (batch, dim) — the z output from the model
            labels: (batch,) — top-class indices
        """
        features = F.normalize(features, dim=1)
        batch_size = features.shape[0]

        # Similarity matrix
        sim = torch.matmul(features, features.T) / self.temperature

        # Mask: 1 where same top-class (excluding self)
        labels = labels.unsqueeze(1)
        pos_mask = (labels == labels.T).float()
        # Self-exclusion mask (0 on diagonal, 1 elsewhere)
        self_mask = 1.0 - torch.eye(batch_size, device=features.device)
        pos_mask = pos_mask * self_mask

        # For numerical stability
        logits_max, _ = sim.max(dim=1, keepdim=True)
        logits = sim - logits_max.detach()

        # Exclude self from denominator (multiply by self_mask instead of in-place fill_diagonal_)
        exp_logits = torch.exp(logits) * self_mask

        # Log prob
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)

        # Mean of log prob over positive pairs
        mask_sum = pos_mask.sum(dim=1)
        mean_log_prob = (pos_mask * log_prob).sum(dim=1) / (mask_sum + 1e-8)

        # Only compute for samples that have at least one positive pair
        valid = mask_sum > 0
        if valid.sum() == 0:
            return torch.tensor(0.0, device=features.device, requires_grad=True)
        loss = -mean_log_prob[valid].mean()

        return loss


class CrossEntropyLoss(nn.Module):
    def __init__(self, class_weights=None, label_smoothing=0.01):
        super(CrossEntropyLoss, self).__init__()

        if class_weights is not None:
            self.cross_entropy = nn.CrossEntropyLoss(
                weight=class_weights, label_smoothing=label_smoothing, reduction='none')
        else:
            self.cross_entropy = nn.CrossEntropyLoss(
                label_smoothing=label_smoothing, reduction='none')

    def forward(self, logits, labels, sample_weights=None):
        per_sample_loss = self.cross_entropy(logits, labels)
        if sample_weights is not None:
            per_sample_loss = per_sample_loss * sample_weights
        return per_sample_loss.mean()


def compute_class_weights(train_df, num_classes, method='inverse_sqrt'):
    """Compute per-class weights from training data distribution.
    
    Methods:
        'inverse_freq': weight = N / (num_classes * count_per_class)
        'inverse_sqrt': weight = sqrt(N / (num_classes * count_per_class))  [less aggressive]
        'effective':    weight based on effective number of samples (beta=0.999)
    """
    class_counts = train_df['class_idx'].value_counts().sort_index()
    
    # Ensure all classes are present
    counts = np.zeros(num_classes)
    for idx, count in class_counts.items():
        counts[int(idx)] = count
    counts = np.maximum(counts, 1)  # avoid division by zero
    
    N = counts.sum()
    
    if method == 'inverse_freq':
        weights = N / (num_classes * counts)
    elif method == 'inverse_sqrt':
        weights = np.sqrt(N / (num_classes * counts))
    elif method == 'effective':
        beta = 0.999
        effective_num = 1.0 - np.power(beta, counts)
        weights = (1.0 - beta) / effective_num
        weights = weights / weights.sum() * num_classes  # normalize
    else:
        weights = np.ones(num_classes)
    
    # Normalize so mean weight = 1.0 (preserves loss scale)
    weights = weights / weights.mean()
    
    return torch.tensor(weights, dtype=torch.float32)