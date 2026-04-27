"""
Axle Graph Model V25 - Gaussian Axle Transformer

Key innovations over V11:
1. Gaussian representation: Each axle = (μ, σ, confidence) instead of discrete point
2. Transformer over axle tokens: Self-attention learns relationships automatically
3. σ as positional encoding: Spacing geometry encoded in attention mechanism
4. No manual edge construction: Attention learns axle grouping patterns

Architecture:
    Image → Backbone → Fused features
                ↓
          Gaussian Axle Head
             ↓   ↓   ↓
            μ   σ   conf
             ↓
    Gaussian Axle Transformer
    - Sample CNN at μ locations
    - Encode σ as position
    - Self-attention learns grouping
             ↓
        Classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import math
import timm

from .multiscale_gnn import MultiScaleGraphNetwork
from .gaussian_transformer import GaussianAxleTransformer
from grace.losses.ordinal import predict_from_ordinal_logits


class GaussianAxleHead(nn.Module):
    """
    Predicts Gaussian parameters for each axle.

    Each axle is represented as a 2D Gaussian with:
    - μ (mu): Center position [y, x] in normalized coords [0, 1]
    - σ (sigma): Spread [σ_y, σ_x] in normalized coords
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

        # Predict Gaussian parameters for each of max_axles slots
        flattened_dim = hidden_dim * 7 * 7

        self.mu_head = nn.Sequential(
            nn.Linear(flattened_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, max_axles * 2),  # [y, x] for each axle
            nn.Sigmoid()  # Normalized coords [0, 1]
        )

        self.sigma_head = nn.Sequential(
            nn.Linear(flattened_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, max_axles * 2),  # [σ_y, σ_x] for each
        )

        # Initialize sigma head to predict log(σ) ≈ log(0.05) initially
        # This ensures σ starts in reasonable range [0.01, 0.15]
        with torch.no_grad():
            target_sigma = (min_sigma + max_sigma) / 2  # 0.08
            init_log_sigma = math.log(target_sigma)
            self.sigma_head[-1].bias.fill_(init_log_sigma)

        self.confidence_head = nn.Sequential(
            nn.Linear(flattened_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, max_axles),
            nn.Sigmoid()
        )

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: [B, C, H, W] fused backbone features

        Returns:
            mu: [B, max_axles, 2] - center positions (y, x) in [0, 1]
            sigma: [B, max_axles, 2] - spreads (σ_y, σ_x) in [0, 1]
            confidence: [B, max_axles] - presence probabilities
        """
        B = features.size(0)

        # Process features
        feat = self.feature_conv(features)  # [B, hidden_dim, H, W]
        feat_pooled = self.pool(feat)  # [B, hidden_dim, 7, 7]
        feat_flat = feat_pooled.flatten(1)  # [B, hidden_dim * 49]

        # Predict Gaussian parameters
        mu = self.mu_head(feat_flat).view(B, self.max_axles, 2)  # [B, K, 2]

        # Sigma with reparameterization trick (log-space prediction)
        # Predicting log(σ) is more stable than predicting σ directly
        log_sigma = self.sigma_head(feat_flat).view(B, self.max_axles, 2)

        # Reparameterization: σ = exp(log_σ)
        # Scale to [min_sigma, max_sigma] range
        sigma = torch.exp(log_sigma)
        sigma = torch.clamp(sigma, min=self.min_sigma, max=self.max_sigma)

        confidence = self.confidence_head(feat_flat)  # [B, K]

        return {
            'mu': mu,
            'sigma': sigma,
            'confidence': confidence,
            'intermediate_features': feat_flat  # [B, hidden_dim * 49] for reconstruction
        }


def gaussian_2d(
    coords: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    confidence: torch.Tensor
) -> torch.Tensor:
    """
    Evaluate 2D Gaussian at given coordinates.

    Args:
        coords: [H, W, 2] grid coordinates (y, x) in [0, 1]
        mu: [B, K, 2] Gaussian centers
        sigma: [B, K, 2] Gaussian spreads
        confidence: [B, K] Gaussian amplitudes

    Returns:
        heatmap: [B, K, H, W] - each Gaussian evaluated on grid
    """
    B, K = mu.shape[:2]
    H, W = coords.shape[:2]

    # Reshape for broadcasting
    coords = coords.view(1, 1, H, W, 2)  # [1, 1, H, W, 2]
    mu = mu.view(B, K, 1, 1, 2)  # [B, K, 1, 1, 2]
    sigma = sigma.view(B, K, 1, 1, 2)  # [B, K, 1, 1, 2]

    # Squared distance
    diff = coords - mu  # [B, K, H, W, 2]
    dist_sq = (diff ** 2) / (2 * sigma ** 2 + 1e-6)  # [B, K, H, W, 2]
    dist_sq = dist_sq.sum(dim=-1)  # [B, K, H, W]

    # Gaussian evaluation
    gaussian = torch.exp(-dist_sq)  # [B, K, H, W]

    # Scale by confidence
    confidence = confidence.view(B, K, 1, 1)  # [B, K, 1, 1]
    heatmap = gaussian * confidence  # [B, K, H, W]

    return heatmap


def reconstruct_heatmap(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    confidence: torch.Tensor,
    height: int = 56,
    width: int = 56
) -> torch.Tensor:
    """
    Reconstruct heatmap as sum of Gaussians.

    Args:
        mu: [B, K, 2] Gaussian centers
        sigma: [B, K, 2] Gaussian spreads
        confidence: [B, K] Gaussian amplitudes
        height, width: Output heatmap resolution

    Returns:
        heatmap: [B, 1, H, W] - sum of all Gaussians
    """
    device = mu.device
    B, K = mu.shape[:2]

    # Create coordinate grid [0, 1]
    y = torch.linspace(0, 1, height, device=device)
    x = torch.linspace(0, 1, width, device=device)
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    coords = torch.stack([yy, xx], dim=-1)  # [H, W, 2]

    # Evaluate all Gaussians
    gaussians = gaussian_2d(coords, mu, sigma, confidence)  # [B, K, H, W]

    # Sum (with max to handle overlaps properly)
    heatmap = gaussians.max(dim=1, keepdim=True)[0]  # [B, 1, H, W]

    return heatmap


class AxleGraphModelV25(nn.Module):
    """
    V25: Geometrically-grounded Gaussian axle detection.

    Key features:
    - Gaussian representation (μ, σ, confidence)
    - Geometric self-supervision (no position labels needed)
    - Explicit spacing information for GNN
    """

    def __init__(
        self,
        backbone_name: str = 'convnextv2_tiny',
        pretrained: bool = True,
        freeze_backbone: bool = False,  # NEW: Option to freeze backbone
        num_fhwa_classes: int = 13,
        num_primary_classes: int = 5,
        num_trailer_classes: int = 4,
        max_axles: int = 12,
        max_segments: int = 5,
        axle_feature_dim: int = 128,
        segment_feature_dim: int = 128,
        gnn_output_dim: int = 256,
        num_coarse_layers: int = 2,
        num_fine_layers: int = 2,
        heatmap_height: int = 56,
        heatmap_width: int = 56,
        **kwargs
    ):
        super().__init__()

        self.max_axles = max_axles
        self.heatmap_height = heatmap_height
        self.heatmap_width = heatmap_width
        self.freeze_backbone = freeze_backbone

        # Backbone with multi-scale features
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=[0, 1, 3]  # Stage 1, 2, 4
        )

        # Freeze backbone if requested (only train task-specific heads)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print(f"✓ Backbone frozen - {sum(p.numel() for p in self.backbone.parameters())/1e6:.1f}M params (not trainable)")

        # Get feature dimensions
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            feats = self.backbone(dummy)
            self.stage1_dim = feats[0].shape[1]
            self.stage2_dim = feats[1].shape[1]
            self.stage4_dim = feats[2].shape[1]

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Conv2d(self.stage1_dim, 128, 1),
            nn.BatchNorm2d(128),
            nn.GELU()
        )

        # Gaussian axle head
        self.gaussian_head = GaussianAxleHead(
            in_channels=128,
            hidden_dim=256,
            max_axles=max_axles,
            min_sigma=0.01,
            max_sigma=0.15
        )

        # Feature projection no longer needed - Transformer handles this internally
        # (Kept commented for reference)
        # self.feature_projection = nn.Sequential(
        #     nn.Linear(self.stage2_dim + 2, axle_feature_dim),
        #     nn.LayerNorm(axle_feature_dim),
        #     nn.GELU()
        # )

        # Gaussian Axle Transformer (replaces GNN)
        # Key innovation: σ becomes positional encoding
        # Self-attention learns axle relationships automatically
        # REDUCED capacity to prevent NaN and overfitting
        self.transformer = GaussianAxleTransformer(
            cnn_dim=128,  # fused features dimension
            hidden_dim=gnn_output_dim,  # Now 128 (reduced from 256)
            num_heads=4,  # Reduced from 8 (fewer parameters)
            num_layers=num_coarse_layers,  # Now 1 (reduced from 2)
            max_axles=max_axles,
            dropout=0.2  # Increased from 0.1 (stronger regularization)
        )

        # Global context
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classification heads
        fusion_dim = gnn_output_dim + self.stage4_dim

        self.fhwa_classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_fhwa_classes)
        )

        self.primary_classifier = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_primary_classes)
        )

        self.trailer_classifier = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_trailer_classes)
        )

        self.axle_count_predictor = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, max_axles + 1)  # Ordinal regression: 0 to max_axles (inclusive)
        )

    def forward(
        self,
        x: torch.Tensor,
        return_heatmap: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [B, 3, H, W] input images
            return_heatmap: Whether to reconstruct heatmap (for visualization)

        Returns:
            Dictionary with predictions and Gaussian parameters
        """
        B = x.size(0)

        # Extract multi-scale features
        stage1, stage2, stage4 = self.backbone(x)
        # stage1: [B, 128, 56, 56]
        # stage2: [B, 256, 28, 28]
        # stage4: [B, 1024, 7, 7]

        # Fuse for Gaussian prediction
        fused = self.fusion(stage1)  # [B, 128, 56, 56]

        # Predict Gaussian parameters
        gaussian_params = self.gaussian_head(fused)
        mu = gaussian_params['mu']  # [B, K, 2]
        sigma = gaussian_params['sigma']  # [B, K, 2]
        confidence = gaussian_params['confidence']  # [B, K]
        intermediate_features = gaussian_params['intermediate_features']  # For reconstruction

        # Optionally reconstruct heatmap
        heatmap = None
        if return_heatmap:
            heatmap = reconstruct_heatmap(
                mu, sigma, confidence,
                self.heatmap_height, self.heatmap_width
            )  # [B, 1, H, W]

        # Gaussian Axle Transformer
        # Transformer handles:
        # 1. Sampling CNN features at μ locations
        # 2. Encoding σ as positional information
        # 3. Self-attention to learn axle relationships
        # No manual edge construction needed!
        graph_features = self.transformer(
            fused_features=fused,  # [B, 128, 56, 56]
            mu=mu,                 # [B, K, 2] - sampling coordinates
            sigma=sigma,           # [B, K, 2] - spacing geometry (positional encoding)
            confidence=confidence  # [B, K] - presence scores
        )  # [B, gnn_output_dim]

        # Global context
        global_context = self.global_pool(stage4).flatten(1)  # [B, stage4_dim]

        # Fusion
        combined = torch.cat([graph_features, global_context], dim=1)  # [B, fusion_dim]

        # Classification
        fhwa_logits = self.fhwa_classifier(combined)
        primary_logits = self.primary_classifier(combined)
        trailer_logits = self.trailer_classifier(combined)
        axle_count_logits = self.axle_count_predictor(combined)

        # Compute num_detected as sum of confidences (expected number of axles)
        num_detected = confidence.sum(dim=1)  # [B]

        return {
            'fhwa_logits': fhwa_logits,
            'primary_logits': primary_logits,
            'trailer_logits': trailer_logits,
            'axle_count_logits': axle_count_logits,
            'axle_logits': axle_count_logits,  # Alias for compatibility with standard loss
            'num_detected': num_detected,  # For detection loss compatibility
            'mu': mu,
            'sigma': sigma,
            'confidence': confidence,
            'heatmap': heatmap,
            'graph_features': graph_features,  # Transformer features (was gnn_features)
            'gnn_features': graph_features,     # Alias for backward compatibility
            # For feature reconstruction loss
            'gaussian_features': intermediate_features,  # [B, hidden_dim * 49]
            'target_features': fused  # [B, 128, 56, 56] - features to reconstruct
        }

    def get_predictions(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Convert model outputs to predictions for evaluation."""
        return {
            'fhwa_pred': outputs['fhwa_logits'].argmax(dim=1),
            'primary_pred': outputs['primary_logits'].argmax(dim=1),
            'axle_pred': predict_from_ordinal_logits(outputs['axle_count_logits'], method='expectation'),
            'trailer_pred': outputs['trailer_logits'].argmax(dim=1),
        }


if __name__ == '__main__':
    # Test
    model = AxleGraphModelV25(
        backbone_name='convnextv2_tiny',
        pretrained=False
    )

    x = torch.randn(2, 3, 224, 224)
    outputs = model(x)

    print("Output shapes:")
    for k, v in outputs.items():
        if v is not None:
            print(f"  {k}: {v.shape}")

    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
