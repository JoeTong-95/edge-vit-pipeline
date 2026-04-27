"""Ordinal regression losses for axle counting.

Treats axle count as ordered categories (0, 1, 2, ..., 12) rather than
continuous regression, which helps the model learn that predicting 6 when
the truth is 7 is better than predicting 3.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftOrdinalLoss(nn.Module):
    """Soft-label ordinal regression via Gaussian label smoothing.

    Instead of one-hot labels:
        7 axles → [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]

    Uses soft Gaussian distribution:
        7 axles → [0, 0, 0, 0, 0.02, 0.10, 0.26, 0.24, 0.26, 0.10, 0.02, 0, 0]
                                    5     6     7     8     9

    This encodes ordering: predicting 6 or 8 gets partial credit,
    predicting 3 gets heavily penalized.

    Args:
        num_classes: Number of ordinal classes (13 for axles 0-12)
        sigma: Width of Gaussian distribution (default 1.0)
               - sigma=1.0 → neighbors get ~24% probability
               - sigma=0.5 → tighter distribution, stricter
               - sigma=1.5 → wider tolerance
    """

    def __init__(self, num_classes: int = 13, sigma: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        self.sigma = sigma
        self.register_buffer('class_indices', torch.arange(num_classes))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B, num_classes] raw class logits
            targets: [B] integer labels 0 to num_classes-1

        Returns:
            Scalar cross-entropy loss with soft ordinal labels
        """
        soft_labels = self._make_soft_labels(targets)

        # Standard cross-entropy with soft targets
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(soft_labels * log_probs).sum(dim=-1).mean()

        return loss

    def _make_soft_labels(self, targets: torch.Tensor) -> torch.Tensor:
        """Create soft Gaussian labels centered at each target.

        Args:
            targets: [B] integer labels

        Returns:
            [B, num_classes] soft label distributions
        """
        B = targets.shape[0]
        device = targets.device

        # Broadcast: [B, 1] vs [1, num_classes] → [B, num_classes]
        # Ensure class_indices is on the same device as targets
        class_indices = self.class_indices.to(device)
        distances = (class_indices.unsqueeze(0) - targets.unsqueeze(1)).float()

        # Gaussian kernel
        soft_labels = torch.exp(-0.5 * (distances / self.sigma) ** 2)

        # Normalize to sum to 1
        soft_labels = soft_labels / soft_labels.sum(dim=1, keepdim=True)

        return soft_labels


class FocalOrdinalLoss(nn.Module):
    """Soft ordinal loss + focal weighting for hard examples.

    Combines soft ordinal labels with focal loss reweighting to focus
    more on hard-to-classify examples (like 7-axle B-trains).

    Args:
        num_classes: Number of ordinal classes
        sigma: Gaussian width for soft labels
        gamma: Focal loss focusing parameter (default 2.0)
               - gamma=0 → no focal weighting (same as SoftOrdinalLoss)
               - gamma=2 → standard focal loss
    """

    def __init__(self, num_classes: int = 13, sigma: float = 1.0, gamma: float = 2.0):
        super().__init__()
        self.soft_ordinal = SoftOrdinalLoss(num_classes, sigma)
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        soft_labels = self.soft_ordinal._make_soft_labels(targets)

        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)

        # Focal weight: (1 - p_correct)^gamma
        # p_correct = sum of probabilities assigned to soft-label distribution
        p_correct = (probs * soft_labels).sum(dim=-1)
        focal_weight = (1 - p_correct) ** self.gamma

        # Weighted cross-entropy
        ce_loss = -(soft_labels * log_probs).sum(dim=-1)
        loss = (focal_weight * ce_loss).mean()

        return loss


def predict_from_ordinal_logits(logits: torch.Tensor, method: str = 'expectation') -> torch.Tensor:
    """Convert ordinal classification logits to scalar predictions.

    Args:
        logits: [B, num_classes] raw logits
        method: 'argmax' or 'expectation'
                - 'argmax': take most likely class (standard)
                - 'expectation': weighted average (smoother, better for regression metrics)

    Returns:
        [B] predicted axle counts (floats if expectation, ints if argmax)
    """
    if method == 'argmax':
        return logits.argmax(dim=-1).float()

    elif method == 'expectation':
        probs = F.softmax(logits, dim=-1)
        num_classes = logits.shape[-1]
        class_indices = torch.arange(num_classes, device=logits.device, dtype=torch.float32)
        # Expectation: E[y] = sum(p(y) * y)
        return (probs * class_indices).sum(dim=-1)

    else:
        raise ValueError(f"Unknown method: {method}")


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing SoftOrdinalLoss...\n")

    # Create loss
    loss_fn = SoftOrdinalLoss(num_classes=13, sigma=1.0)

    # Dummy batch: 4 samples
    logits = torch.randn(4, 13)  # random predictions
    targets = torch.tensor([2, 5, 7, 9])  # true axle counts

    # Show soft labels
    print("Soft label distribution for 7 axles:")
    soft_labels = loss_fn._make_soft_labels(torch.tensor([7]))
    for i, prob in enumerate(soft_labels[0]):
        if prob > 0.01:
            print(f"  {i} axles: {prob:.3f}")

    # Compute loss
    loss = loss_fn(logits, targets)
    print(f"\nLoss: {loss.item():.4f}")

    # Predictions
    pred_argmax = predict_from_ordinal_logits(logits, 'argmax')
    pred_expect = predict_from_ordinal_logits(logits, 'expectation')

    print(f"\nTrue:              {targets.tolist()}")
    print(f"Pred (argmax):     {pred_argmax.tolist()}")
    print(f"Pred (expectation): {[f'{p:.2f}' for p in pred_expect.tolist()]}")

    print("\n✓ Test passed")
