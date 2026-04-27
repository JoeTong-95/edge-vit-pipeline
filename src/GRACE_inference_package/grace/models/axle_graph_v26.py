"""
Axle Graph Model V26 - Hierarchical Prediction + Rotated Gaussians

Key innovations over V25:
1. **Hierarchical prediction**: Axle/Trailer → FHWA → Primary
   - FHWA classifier uses predicted axle/trailer counts as additional features
   - Rationale: FHWA classes are defined by axle count + trailer config

2. **Rotated Gaussians**: Full 2D Gaussians with orientation
   - 5 params per Gaussian: μ (position), σ₁, σ₂ (scales), θ (rotation)
   - Can orient ellipses to match diagonal vehicle structures
   - Better fit to rotated/perspective views

Architecture:
    Image → Backbone → Fused features → Rotated Gaussian Head
                                              ↓
                                    Transformer (axle relationships)
                                    [positional encoding: 6D with rotation]
                                              ↓
                                    Graph + Global features
                                              ↓
                          ┌───────────────────┴───────────────────┐
                          ↓                                       ↓
                   Axle Count Head                         Trailer Count Head
                          ↓                                       ↓
                          └───────────────────┬───────────────────┘
                                              ↓
                          Concatenate [features + axle_pred + trailer_pred]
                                              ↓
                                         FHWA Head
                                              ↓
                                        Primary Head

Performance expectation:
- Better FHWA accuracy (hierarchical + rotated Gaussians)
- Tighter axle localization (rotated ellipses)
- More interpretable geometry
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from .axle_graph_v25 import AxleGraphModelV25
from .gaussian_head_rotated import RotatedGaussianHead, reconstruct_rotated_heatmap
from grace.losses.ordinal import predict_from_ordinal_logits


class AxleGraphModelV26(AxleGraphModelV25):
    """
    V26: Hierarchical prediction with axle/trailer → FHWA dependency.

    Inherits from V25 but modifies classification head architecture.
    """

    def __init__(
        self,
        backbone_name: str = 'convnextv2_tiny',
        pretrained: bool = True,
        freeze_backbone: bool = False,
        num_fhwa_classes: int = 13,
        num_primary_classes: int = 5,
        num_trailer_classes: int = 4,
        max_axles: int = 12,
        max_segments: int = 5,
        num_gaussian_slots: int = None,  # NEW: Separate Gaussian slots from max_axles
        axle_feature_dim: int = 128,
        segment_feature_dim: int = 128,
        gnn_output_dim: int = 256,
        num_coarse_layers: int = 2,
        num_fine_layers: int = 2,
        heatmap_height: int = 56,
        heatmap_width: int = 56,
        use_soft_predictions: bool = False,  # NEW: Use soft predictions vs hard
        **kwargs
    ):
        # Initialize parent V25 model
        super().__init__(
            backbone_name=backbone_name,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            num_fhwa_classes=num_fhwa_classes,
            num_primary_classes=num_primary_classes,
            num_trailer_classes=num_trailer_classes,
            max_axles=max_axles,
            max_segments=max_segments,
            axle_feature_dim=axle_feature_dim,
            segment_feature_dim=segment_feature_dim,
            gnn_output_dim=gnn_output_dim,
            num_coarse_layers=num_coarse_layers,
            num_fine_layers=num_fine_layers,
            heatmap_height=heatmap_height,
            heatmap_width=heatmap_width,
            **kwargs
        )

        self.use_soft_predictions = use_soft_predictions

        # Use num_gaussian_slots if provided, otherwise use max_axles
        self.num_gaussian_slots = num_gaussian_slots if num_gaussian_slots is not None else max_axles

        # Override Gaussian head to use rotated version
        self.gaussian_head = RotatedGaussianHead(
            in_channels=128,
            hidden_dim=256,
            max_axles=self.num_gaussian_slots,  # Use num_gaussian_slots for Gaussians
            min_sigma=0.01,
            max_sigma=0.15
        )

        # Override transformer to use num_gaussian_slots
        from .gaussian_transformer import GaussianAxleTransformer
        self.transformer = GaussianAxleTransformer(
            cnn_dim=128,
            hidden_dim=gnn_output_dim,
            num_heads=4,
            num_layers=num_coarse_layers,
            max_axles=self.num_gaussian_slots,  # Use num_gaussian_slots, not max_axles
            dropout=0.2
        )

        # Override classification heads to implement hierarchical structure
        # fusion_dim is already set in parent: gnn_output_dim + stage4_dim
        fusion_dim = gnn_output_dim + self.stage4_dim

        # ── Level 1: Axle and Trailer Prediction (unchanged) ──
        self.axle_count_predictor = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, max_axles + 1)  # Ordinal regression: 0 to max_axles
        )

        self.trailer_classifier = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_trailer_classes)
        )

        # ── Level 2: FHWA uses axle + trailer predictions ──
        # Input: [fusion_dim + axle_embedding + trailer_embedding]
        # We embed the predictions instead of using raw values
        self.axle_embedding = nn.Linear(1, 16)  # Continuous axle count → embedding
        self.trailer_embedding = nn.Embedding(num_trailer_classes, 16)  # Discrete trailer class → embedding

        fhwa_input_dim = fusion_dim + 16 + 16  # fusion + axle_emb + trailer_emb

        self.fhwa_classifier = nn.Sequential(
            nn.Linear(fhwa_input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_fhwa_classes)
        )

        # ── Level 3: Primary uses FHWA predictions (optional) ──
        # Could also use axle/trailer like FHWA, or just original features
        # For now, keep it simple: use original features only
        self.primary_classifier = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_primary_classes)
        )

        print(f"✓ V26 Hierarchical Model initialized")
        print(f"  - Axle/Trailer → FHWA (with embeddings)")
        print(f"  - Soft predictions: {use_soft_predictions}")

    def forward(
        self,
        x: torch.Tensor,
        return_heatmap: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with hierarchical prediction.

        Args:
            x: [B, 3, H, W] input images
            return_heatmap: Whether to reconstruct heatmap

        Returns:
            Dictionary with predictions and Gaussian parameters
        """
        B = x.size(0)

        # ── Step 1: Backbone + Gaussian Head (same as V25) ──
        stage1, stage2, stage4 = self.backbone(x)
        fused = self.fusion(stage1)

        gaussian_params = self.gaussian_head(fused)
        mu = gaussian_params['mu']
        sigma = gaussian_params['sigma']
        theta = gaussian_params['theta']  # NEW: rotation angle
        confidence = gaussian_params['confidence']
        intermediate_features = gaussian_params['intermediate_features']

        heatmap = None
        if return_heatmap:
            heatmap = reconstruct_rotated_heatmap(
                mu, sigma, theta, confidence,
                self.heatmap_height, self.heatmap_width
            )

        # ── Step 2: Transformer + Global Context ──
        # Transformer now receives rotation info for richer positional encoding
        graph_features = self.transformer(
            fused_features=fused,
            mu=mu,
            sigma=sigma,
            confidence=confidence,
            theta=theta  # NEW: pass rotation for 6D positional encoding
        )

        global_context = self.global_pool(stage4).flatten(1)
        combined = torch.cat([graph_features, global_context], dim=1)  # [B, fusion_dim]

        # ── Step 3: Level 1 - Predict Axle and Trailer FIRST ──
        axle_count_logits = self.axle_count_predictor(combined)  # [B, max_axles+1]
        trailer_logits = self.trailer_classifier(combined)       # [B, num_trailer_classes]

        # Get predictions for hierarchical input
        # For training: use soft predictions (with gradients)
        # For inference: can use hard predictions
        if self.use_soft_predictions or self.training:
            # Soft prediction: expected value from softmax distribution
            axle_probs = F.softmax(axle_count_logits, dim=1)  # [B, max_axles+1]
            axle_values = torch.arange(
                0, self.max_axles + 1,
                dtype=axle_probs.dtype,
                device=axle_probs.device
            ).unsqueeze(0)  # [1, max_axles+1]
            axle_pred_soft = (axle_probs * axle_values).sum(dim=1, keepdim=True)  # [B, 1]

            # For trailer, use soft embedding
            trailer_probs = F.softmax(trailer_logits, dim=1)  # [B, num_trailer_classes]
            # Weighted sum of trailer embeddings
            trailer_emb_all = self.trailer_embedding.weight  # [num_trailer_classes, 16]
            trailer_emb = torch.matmul(trailer_probs, trailer_emb_all)  # [B, 16]

            axle_emb = self.axle_embedding(axle_pred_soft)  # [B, 16]

        else:
            # Hard prediction: argmax (no gradients flow back to axle/trailer heads)
            axle_pred_hard = predict_from_ordinal_logits(
                axle_count_logits,
                method='expectation'
            ).unsqueeze(1)  # [B, 1]
            trailer_pred_hard = trailer_logits.argmax(dim=1)  # [B]

            # Detach to prevent gradient flow
            axle_emb = self.axle_embedding(axle_pred_hard.detach())  # [B, 16]
            trailer_emb = self.trailer_embedding(trailer_pred_hard.detach())  # [B, 16]

        # ── Step 4: Level 2 - FHWA uses axle/trailer predictions ──
        fhwa_input = torch.cat([combined, axle_emb, trailer_emb], dim=1)  # [B, fusion_dim+32]
        fhwa_logits = self.fhwa_classifier(fhwa_input)

        # ── Step 5: Level 3 - Primary classification ──
        # For now, use original features (could also use FHWA predictions)
        primary_logits = self.primary_classifier(combined)

        # ── Step 6: Return all outputs ──
        num_detected = confidence.sum(dim=1)

        return {
            'fhwa_logits': fhwa_logits,
            'primary_logits': primary_logits,
            'trailer_logits': trailer_logits,
            'axle_count_logits': axle_count_logits,
            'axle_logits': axle_count_logits,  # Alias
            'num_detected': num_detected,
            'mu': mu,
            'sigma': sigma,
            'theta': theta,  # NEW: rotation angles
            'confidence': confidence,
            'heatmap': heatmap,
            'graph_features': graph_features,
            'gnn_features': graph_features,  # Alias
            'gaussian_features': intermediate_features,
            'target_features': fused,
            # V26-specific outputs
            'axle_embedding': axle_emb,
            'trailer_embedding': trailer_emb,
        }


if __name__ == '__main__':
    # Test V26
    print("Testing V26 Hierarchical Model...")

    model = AxleGraphModelV26(
        backbone_name='convnextv2_tiny',
        pretrained=False,
        use_soft_predictions=True
    )

    x = torch.randn(2, 3, 224, 224)
    outputs = model(x)

    print("\nOutput shapes:")
    for k, v in outputs.items():
        if v is not None and torch.is_tensor(v):
            print(f"  {k}: {v.shape}")

    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # Test gradient flow
    loss = outputs['fhwa_logits'].sum()
    loss.backward()
    print("\n✓ Backward pass successful (gradients flow through hierarchy)")
