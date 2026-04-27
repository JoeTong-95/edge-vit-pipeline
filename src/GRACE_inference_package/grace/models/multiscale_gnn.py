"""Multi-Scale Graph Neural Network for Hierarchical Vehicle Reasoning.

Adds coarse-level (segment) and fine-level (axle) reasoning to the standard GNN.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class LearnableSegmentation(nn.Module):
    """Learn to group axles into vehicle segments (cab, trailer1, trailer2, etc.)."""

    def __init__(self, feature_dim: int = 64, max_segments: int = 4, num_heads: int = 4):
        super().__init__()
        self.segment_queries = nn.Parameter(torch.randn(max_segments, feature_dim))
        self.attention = nn.MultiheadAttention(feature_dim, num_heads=num_heads, batch_first=True)

    def forward(self, axle_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            axle_features: [B, N, F] axle features

        Returns:
            segment_assignments: [B, N, S] soft assignment probabilities
        """
        B, N, F = axle_features.shape
        S = self.segment_queries.shape[0]

        # Expand segment queries for batch
        queries = self.segment_queries.unsqueeze(0).expand(B, -1, -1)  # [B, S, F]

        # Cross-attention: which axles belong to which segments?
        attn_output, attn_weights = self.attention(
            queries, axle_features, axle_features
        )  # attn_weights: [B, S, N]

        # Transpose to [B, N, S] for easier aggregation
        return attn_weights.transpose(1, 2)


class SimpleGraphConv(nn.Module):
    """Simple graph convolution layer (simplified from PyG for easier use)."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B*N, F] node features
            edge_index: [2, E] edge connectivity

        Returns:
            out: [B*N, F'] updated features
        """
        # Simple message passing: aggregate from neighbors
        num_nodes = x.shape[0]

        # Transform features
        out = self.linear(x)

        # Aggregate messages from neighbors
        if edge_index.shape[1] > 0:
            src, dst = edge_index
            messages = torch.zeros_like(out)
            messages.index_add_(0, dst, out[src])

            # Normalize by degree
            degree = torch.bincount(dst, minlength=num_nodes).float().clamp(min=1)
            out = out + messages / degree.unsqueeze(1)

        out = self.bn(out)
        return F.relu(out)


class MultiScaleGraphNetwork(nn.Module):
    """Multi-Scale Graph Network with hierarchical reasoning.

    Architecture:
    1. Segment-level (coarse): Reason about vehicle segments
    2. Axle-level (fine): Refine individual axle features
    """

    def __init__(
        self,
        axle_feature_dim: int = 64,
        segment_feature_dim: int = 128,
        output_dim: int = 128,
        num_coarse_layers: int = 2,
        num_fine_layers: int = 3,
        max_segments: int = 4,
    ):
        super().__init__()
        self.axle_feature_dim = axle_feature_dim
        self.segment_feature_dim = segment_feature_dim
        self.output_dim = output_dim
        self.max_segments = max_segments

        # Segmentation module
        self.segmentation = LearnableSegmentation(axle_feature_dim, max_segments)

        # Projections
        self.axle_to_segment = nn.Linear(axle_feature_dim, segment_feature_dim)
        self.segment_to_axle = nn.Linear(segment_feature_dim, axle_feature_dim)

        # Coarse GNN (segment-level)
        self.coarse_gnn_layers = nn.ModuleList([
            SimpleGraphConv(segment_feature_dim, segment_feature_dim)
            for _ in range(num_coarse_layers)
        ])

        # Fine GNN (axle-level)
        self.fine_gnn_layers = nn.ModuleList([
            SimpleGraphConv(axle_feature_dim, axle_feature_dim)
            for _ in range(num_fine_layers)
        ])

        # Output projection
        self.output_proj = nn.Linear(axle_feature_dim, output_dim)

    def forward(
        self,
        axle_features: torch.Tensor,
        axle_positions,  # List[torch.Tensor] or torch.Tensor
        presence_scores,  # List[torch.Tensor] or torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            axle_features: [B, N, F] detected axle features
            axle_positions: List of [N, 2] tensors or [B, N, 2] tensor
            presence_scores: List of [N] tensors or [B, N] tensor

        Returns:
            graph_features: [B, output_dim] graph representation
        """
        B, N, feat_dim = axle_features.shape

        # Convert lists to tensors if needed (for compatibility with V5/V6/V7)
        if isinstance(axle_positions, list):
            axle_positions = torch.stack([p for p in axle_positions])  # [B, N, 2]
        if isinstance(presence_scores, list):
            presence_scores = torch.stack([p for p in presence_scores])  # [B, N]

        # ===== COARSE LEVEL: Segment Reasoning =====

        # 1. Learn segment assignments
        segment_assignments = self.segmentation(axle_features)  # [B, N, S]
        segment_assignments = F.softmax(segment_assignments, dim=-1)

        # 2. Aggregate axle features into segments
        axle_projected = self.axle_to_segment(axle_features)  # [B, N, segment_dim]
        segment_features = torch.bmm(
            segment_assignments.transpose(1, 2), axle_projected
        )  # [B, S, segment_dim]

        # 3. Build fully-connected segment graph
        segment_edge_index = self._build_fully_connected_edges(self.max_segments)
        segment_edge_index = segment_edge_index.to(axle_features.device)

        # 4. Coarse GNN
        segment_feats_flat = segment_features.reshape(B * self.max_segments, -1)

        for layer in self.coarse_gnn_layers:
            # Repeat edges for each batch
            batch_edge_index = segment_edge_index.clone()
            for b in range(1, B):
                batch_offset = b * self.max_segments
                batch_edge_index = torch.cat([
                    batch_edge_index,
                    segment_edge_index + batch_offset
                ], dim=1)

            segment_feats_flat = layer(segment_feats_flat, batch_edge_index)

        segment_features = segment_feats_flat.reshape(B, self.max_segments, -1)

        # ===== FINE LEVEL: Axle Reasoning =====

        # 5. Broadcast segment context to axles
        segment_context = torch.bmm(segment_assignments, segment_features)  # [B, N, segment_dim]
        segment_context = self.segment_to_axle(segment_context)  # [B, N, axle_dim]

        # 6. Enhance axle features
        enhanced_axles = axle_features + segment_context

        # 7. Build k-NN axle graph
        axle_edge_index = self._build_knn_edges(axle_positions, presence_scores, k=4)
        axle_edge_index = axle_edge_index.to(axle_features.device)

        # 8. Fine GNN
        axle_feats_flat = enhanced_axles.reshape(B * N, -1)

        for layer in self.fine_gnn_layers:
            axle_feats_flat = layer(axle_feats_flat, axle_edge_index)

        refined_axles = axle_feats_flat.reshape(B, N, -1)

        # 9. Global pooling (weighted by presence)
        presence_weights = presence_scores.unsqueeze(-1)  # [B, N, 1]
        graph_features = (refined_axles * presence_weights).sum(dim=1)  # [B, F]
        graph_features = graph_features / (presence_weights.sum(dim=1).clamp(min=1e-8))

        # 10. Output projection
        output = self.output_proj(graph_features)

        return output

    def _build_fully_connected_edges(self, num_nodes: int) -> torch.Tensor:
        """Build fully-connected graph edges."""
        edges = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    edges.append([i, j])

        if len(edges) == 0:
            return torch.zeros((2, 0), dtype=torch.long)

        return torch.tensor(edges, dtype=torch.long).t()

    def _build_knn_edges(
        self,
        positions: torch.Tensor,
        presence_scores: torch.Tensor,
        k: int = 4
    ) -> torch.Tensor:
        """Build k-nearest neighbor edges for axles."""
        B, N = positions.shape[:2]
        all_edges = []

        for b in range(B):
            # Get valid axles
            valid_mask = presence_scores[b] > 0.5
            valid_indices = valid_mask.nonzero(as_tuple=False).squeeze(-1)

            if len(valid_indices) < 2:
                continue

            valid_pos = positions[b, valid_indices]

            # Compute pairwise distances
            dist = torch.cdist(valid_pos, valid_pos)

            # Connect each axle to k nearest neighbors
            k_actual = min(k + 1, len(valid_indices))
            _, nearest = dist.topk(k_actual, dim=1, largest=False)

            for i, idx_i in enumerate(valid_indices):
                for j in nearest[i, 1:]:  # Skip self (nearest[i, 0])
                    idx_j = valid_indices[j]
                    all_edges.append([
                        idx_i.item() + b * N,
                        idx_j.item() + b * N
                    ])

        if len(all_edges) == 0:
            return torch.zeros((2, 0), dtype=torch.long)

        return torch.tensor(all_edges, dtype=torch.long).t()
