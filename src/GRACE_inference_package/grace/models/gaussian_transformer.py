"""
Gaussian Axle Transformer - Transformer encoder over Gaussian axle tokens.

Each Gaussian axle = one token
Token = CNN features sampled at μ + positional encoding from (μ, σ)

Key innovation: σ (spread) becomes part of positional encoding,
allowing attention to learn spatial grouping patterns.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class GaussianAxleTransformer(nn.Module):
    """
    Transformer encoder over Gaussian axle tokens.
    
    Replaces GNN with self-attention mechanism that:
    - Learns axle relationships automatically (no manual edges)
    - Uses σ as positional encoding (spacing geometry)
    - Handles variable number of axles naturally
    - Provides interpretable attention maps
    
    Args:
        cnn_dim: Dimension of CNN features (from fused backbone)
        hidden_dim: Transformer hidden dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer encoder layers
        max_axles: Maximum number of axles
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        cnn_dim: int = 128,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 2,
        max_axles: int = 12,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Project CNN features to transformer dimension
        self.content_proj = nn.Sequential(
            nn.Linear(cnn_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # Encode geometry (μ, σ, θ) into positional encoding
        # Key insight: σ provides spacing, θ provides orientation
        # Support both 4D (no rotation) and 6D (with rotation)
        self.pos_encoding_4d = nn.Sequential(
            nn.Linear(4, hidden_dim),  # [μ_y, μ_x, σ_y, σ_x]
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.pos_encoding_6d = nn.Sequential(
            nn.Linear(6, hidden_dim),  # [μ_y, μ_x, σ_y, σ_x, sin(θ), cos(θ)]
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # CLS token - aggregates global axle configuration
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        # Confidence embedding - encodes axle presence strength
        self.confidence_embedding = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-norm for stability
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Output projection
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(
        self,
        fused_features: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        confidence: torch.Tensor,
        theta: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass - convert Gaussian axles to tokens, apply attention.

        Args:
            fused_features: [B, C, H, W] - CNN features from backbone
            mu: [B, K, 2] - Gaussian centers (normalized 0-1)
            sigma: [B, K, 2] - Gaussian spreads
            confidence: [B, K] - Presence scores
            theta: [B, K] - Rotation angles (optional, for V26)

        Returns:
            graph_features: [B, hidden_dim] - Aggregated axle representation
        """
        B, K, _ = mu.shape

        # ── Step 1: Reparameterization trick with sampling ──
        # During training: sample from N(μ, σ²) for robustness
        # During inference: use μ directly (deterministic)
        if self.training:
            # Reparameterization: z = μ + σ * ε, where ε ~ N(0, 1)
            epsilon = torch.randn_like(mu)  # [B, K, 2]
            sampled_positions = mu + sigma * epsilon  # Sample from Gaussian
            # Clamp to valid range [0, 1]
            sampled_positions = torch.clamp(sampled_positions, 0.0, 1.0)
        else:
            # Inference: use mean (no sampling)
            sampled_positions = mu

        # Convert to grid coordinates for sampling
        grid = sampled_positions * 2 - 1  # [B, K, 2] → convert to [-1, 1]
        grid = grid.unsqueeze(1)  # [B, 1, K, 2] for grid_sample
        
        sampled_features = F.grid_sample(
            fused_features,
            grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )  # [B, C, 1, K]
        
        sampled_features = sampled_features.squeeze(2).permute(0, 2, 1)  # [B, K, C]
        
        # ── Step 2: Project CNN content to transformer dimension ──
        content_tokens = self.content_proj(sampled_features)  # [B, K, hidden_dim]
        
        # ── Step 3: Create positional encoding from geometry ──
        # Key insight: σ encodes spacing, θ encodes orientation
        # Nearby axles with overlapping σ and similar θ → likely same structure
        # → Higher attention scores → Transformer learns grouping
        if theta is not None:
            # V26: 6D geometry with rotation
            geometry = torch.cat([
                mu,                       # [B, K, 2] - position
                sigma,                    # [B, K, 2] - spread
                theta.sin().unsqueeze(-1), # [B, K, 1] - sin(θ)
                theta.cos().unsqueeze(-1)  # [B, K, 1] - cos(θ)
            ], dim=-1)  # [B, K, 6]
            pos_enc = self.pos_encoding_6d(geometry)  # [B, K, hidden_dim]
        else:
            # V25: 4D geometry (axis-aligned)
            geometry = torch.cat([
                mu,      # [B, K, 2] - position
                sigma    # [B, K, 2] - spread
            ], dim=-1)  # [B, K, 4]
            pos_enc = self.pos_encoding_4d(geometry)  # [B, K, hidden_dim]
        
        # ── Step 4: Encode confidence as learned embedding ──
        conf_enc = self.confidence_embedding(
            confidence.unsqueeze(-1)  # [B, K, 1]
        )  # [B, K, hidden_dim]
        
        # ── Step 5: Combine into tokens ──
        # Token = content + positional encoding + confidence encoding
        tokens = content_tokens + pos_enc + conf_enc  # [B, K, hidden_dim]
        
        # ── Step 6: Prepend CLS token ──
        cls = self.cls_token.expand(B, -1, -1)  # [B, 1, hidden_dim]
        tokens = torch.cat([cls, tokens], dim=1)  # [B, K+1, hidden_dim]
        
        # ── Step 7: Apply Transformer self-attention ──
        # Each axle attends to all other axles
        # CLS token aggregates global axle configuration
        # No manual edge construction needed!
        out = self.transformer(tokens)  # [B, K+1, hidden_dim]
        
        # ── Step 8: Extract CLS output as global representation ──
        cls_out = out[:, 0, :]  # [B, hidden_dim]
        
        # ── Step 9: Output projection ──
        graph_features = self.output_norm(cls_out)
        graph_features = self.output_proj(graph_features)  # [B, hidden_dim]
        
        return graph_features


class GaussianAxleTransformerV2(nn.Module):
    """
    Alternative design: Pool over all axle tokens instead of CLS.
    
    Uses attention pooling to aggregate axle tokens.
    More flexible than CLS for varying number of valid axles.
    """
    
    def __init__(
        self,
        cnn_dim: int = 128,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 2,
        max_axles: int = 12,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Same as V1
        self.content_proj = nn.Sequential(
            nn.Linear(cnn_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        self.pos_encoding = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Attention pooling instead of CLS token
        self.attention_pool = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )
        
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(
        self,
        fused_features: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        confidence: torch.Tensor
    ) -> torch.Tensor:
        B, K, _ = mu.shape
        
        # Same sampling and encoding as V1
        grid = mu * 2 - 1
        grid = grid.unsqueeze(1)
        
        sampled_features = F.grid_sample(
            fused_features,
            grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )
        sampled_features = sampled_features.squeeze(2).permute(0, 2, 1)
        
        content_tokens = self.content_proj(sampled_features)
        
        geometry = torch.cat([mu, sigma], dim=-1)
        pos_enc = self.pos_encoding(geometry)
        
        # Weight by confidence (multiplicative instead of additive)
        conf_weight = confidence.unsqueeze(-1)  # [B, K, 1]
        tokens = (content_tokens + pos_enc) * conf_weight  # [B, K, hidden_dim]
        
        # Transformer without CLS token
        out = self.transformer(tokens)  # [B, K, hidden_dim]
        
        # Attention pooling - learn which axles to focus on
        attention_weights = self.attention_pool(out)  # [B, K, 1]
        pooled = (out * attention_weights).sum(dim=1)  # [B, hidden_dim]
        
        # Output projection
        graph_features = self.output_norm(pooled)
        graph_features = self.output_proj(graph_features)
        
        return graph_features
