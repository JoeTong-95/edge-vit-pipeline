"""
Rotated Gaussian Axle Head - Full 2D Gaussians with orientation.

Predicts 5 parameters per Gaussian instead of 4:
- μ (mu): Center position [y, x] in normalized coords [0, 1]
- σ (sigma): Scale along major/minor axes [σ₁, σ₂]
- θ (theta): Rotation angle
- confidence: Presence probability [0, 1]
"""

import torch
import torch.nn as nn
import math


class RotatedGaussianHead(nn.Module):
    """
    Predicts rotated 2D Gaussian parameters for each axle.

    Key difference from axis-aligned: Can orient ellipses at any angle,
    allowing tighter fit to diagonal/rotated vehicle structures.

    Each axle is represented as:
    - μ (mu): Center position [y, x] in [0, 1]
    - σ (sigma): Scale [σ₁, σ₂] along major/minor axes
    - θ (theta): Rotation angle (predicted as sin/cos to avoid discontinuity)
    - confidence: Presence probability [0, 1]
    """

    def __init__(
        self,
        in_channels: int = 128,
        hidden_dim: int = 256,
        max_axles: int = 12,
        min_sigma: float = 0.01,
        max_sigma: float = 0.15
    ):
        super().__init__()
        self.max_axles = max_axles
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma

        # Shared feature processing
        self.feature_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU()
        )

        # Global pooling for axle-level predictions
        self.pool = nn.AdaptiveAvgPool2d((7, 7))
        flattened_dim = hidden_dim * 7 * 7

        # μ head: [y, x] center position per axle
        self.mu_head = nn.Sequential(
            nn.Linear(flattened_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, max_axles * 2),  # [y, x] for each axle
            nn.Sigmoid()  # Normalized coords [0, 1]
        )

        # σ head: [σ₁, σ₂] major/minor axis scales per axle
        self.sigma_head = nn.Sequential(
            nn.Linear(flattened_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, max_axles * 2),  # [σ₁, σ₂] for each
        )

        # Initialize sigma head to predict log(σ) ≈ log(0.05)
        with torch.no_grad():
            target_sigma = (min_sigma + max_sigma) / 2  # 0.08
            init_log_sigma = math.log(target_sigma)
            self.sigma_head[-1].bias.fill_(init_log_sigma)

        # θ head: rotation angle per axle
        # Predict (sin θ, cos θ) instead of θ directly to avoid discontinuity at 0/2π
        self.theta_head = nn.Sequential(
            nn.Linear(flattened_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, max_axles * 2),  # [sin θ, cos θ] for each
        )

        # Confidence head
        self.confidence_head = nn.Sequential(
            nn.Linear(flattened_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, max_axles),
            nn.Sigmoid()
        )

    def forward(self, features: torch.Tensor) -> dict:
        """
        Args:
            features: [B, C, H, W] fused backbone features

        Returns:
            mu: [B, max_axles, 2] - center positions (y, x) in [0, 1]
            sigma: [B, max_axles, 2] - scales (σ₁, σ₂) along major/minor axes
            theta: [B, max_axles] - rotation angles in radians
            confidence: [B, max_axles] - presence probabilities
        """
        B = features.size(0)

        # Process features
        feat = self.feature_conv(features)  # [B, hidden_dim, H, W]
        feat_pooled = self.pool(feat)  # [B, hidden_dim, 7, 7]
        feat_flat = feat_pooled.flatten(1)  # [B, hidden_dim * 49]

        # Predict Gaussian parameters
        mu = self.mu_head(feat_flat).view(B, self.max_axles, 2)  # [B, K, 2]

        # Sigma with log-space prediction for stability
        log_sigma = self.sigma_head(feat_flat).view(B, self.max_axles, 2)
        sigma = torch.exp(log_sigma)
        sigma = torch.clamp(sigma, min=self.min_sigma, max=self.max_sigma)

        # Rotation: predict sin/cos, then compute angle
        # This avoids discontinuity at θ = 0/2π
        sincos = self.theta_head(feat_flat).view(B, self.max_axles, 2)

        # Normalize to unit circle (make it a valid rotation)
        sincos = sincos / (sincos.norm(dim=-1, keepdim=True) + 1e-6)

        # Compute angle from normalized sin/cos
        theta = torch.atan2(sincos[..., 0], sincos[..., 1])  # [B, K]

        confidence = self.confidence_head(feat_flat)  # [B, K]

        return {
            'mu': mu,
            'sigma': sigma,
            'theta': theta,
            'confidence': confidence,
            'intermediate_features': feat_flat  # For reconstruction loss
        }


def reconstruct_rotated_heatmap(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    theta: torch.Tensor,
    confidence: torch.Tensor,
    height: int = 56,
    width: int = 56
) -> torch.Tensor:
    """
    Reconstruct heatmap as sum of rotated Gaussians.

    Each Gaussian is an oriented ellipse that can rotate to match
    the vehicle structure (e.g., diagonal axle lines in perspective).

    Args:
        mu: [B, K, 2] Gaussian centers (y, x)
        sigma: [B, K, 2] Gaussian scales (σ₁, σ₂)
        theta: [B, K] Rotation angles in radians
        confidence: [B, K] Gaussian amplitudes
        height, width: Output heatmap resolution

    Returns:
        heatmap: [B, 1, H, W] - sum of all rotated Gaussians
    """
    device = mu.device
    B, K = mu.shape[:2]

    # Create coordinate grid [0, 1]
    y = torch.linspace(0, 1, height, device=device)
    x = torch.linspace(0, 1, width, device=device)
    yy, xx = torch.meshgrid(y, x, indexing='ij')  # [H, W]

    # Collect all Gaussian contributions (avoid inplace for DDP)
    gaussians = []

    for k in range(K):
        # Offset from Gaussian center
        dy = yy - mu[:, k, 0].view(B, 1, 1)  # [B, H, W]
        dx = xx - mu[:, k, 1].view(B, 1, 1)

        # Rotate into Gaussian's local coordinate frame
        cos_t = torch.cos(theta[:, k]).view(B, 1, 1)
        sin_t = torch.sin(theta[:, k]).view(B, 1, 1)

        # Local coordinates (aligned with Gaussian axes)
        local_y = cos_t * dy + sin_t * dx   # [B, H, W]
        local_x = -sin_t * dy + cos_t * dx

        # Axis-aligned Gaussian in local frame
        s1 = sigma[:, k, 0].view(B, 1, 1)  # Major axis
        s2 = sigma[:, k, 1].view(B, 1, 1)  # Minor axis

        # Gaussian evaluation
        dist_sq = (local_y ** 2) / (s1 ** 2 + 1e-6) + (local_x ** 2) / (s2 ** 2 + 1e-6)
        gaussian = torch.exp(-0.5 * dist_sq)  # [B, H, W]

        # Scale by confidence
        conf = confidence[:, k].view(B, 1, 1)
        gaussians.append(conf * gaussian)

    # Stack and take maximum across all Gaussians (DDP-safe)
    all_gaussians = torch.stack(gaussians, dim=0)  # [K, B, H, W]
    heatmap = all_gaussians.max(dim=0)[0].unsqueeze(1)  # [B, 1, H, W]

    return heatmap
