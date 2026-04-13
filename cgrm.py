# coding=utf-8
"""Causality-driven Graph Reasoning Module.

This file implements the CGRM described in the manuscript:
masked projection from pixel features to causal/non-causal graph nodes,
graph convolution over the concatenated nodes, transpose projection back to
the pixel grid, and residual feature fusion.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


TensorOrDebug = Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]


class CausalityDrivenGraphReasoningModule(nn.Module):
    """CGRM for lesion/healthy topological reasoning.

    Args:
        in_channels: Number of input feature channels.
        num_nodes: Number of graph nodes for each region branch. The graph has
            ``2 * num_nodes`` nodes after concatenating causal and non-causal
            vertices.
        graph_channels: Hidden channel width inside the graph convolution.
            Defaults to ``in_channels``.
        out_channels: Number of output channels. Defaults to ``in_channels``.
        dropout: Dropout probability applied in the graph branch.
        residual: Add the input feature map to the fused output when channel
            counts match.
        mask_threshold: Threshold used when masks are derived from binary
            coarse logits.
        eps: Numerical stability constant.

    Forward inputs:
        x: Encoded feature map with shape ``[B, C, H, W]``.
        causal_mask: Optional lesion mask with shape ``[B, 1, Hm, Wm]`` or
            ``[B, Hm, Wm]``.
        non_causal_mask: Optional healthy/non-lesion mask with the same shape.
        coarse_logits: Optional coarse segmentation output. If masks are not
            supplied, CGRM derives them from this tensor. For multi-class
            logits or label maps, class 0 is treated as non-causal and classes
            > 0 as causal.

    Returns:
        Refined feature map with shape ``[B, out_channels, H, W]``. When
        ``return_debug=True``, also returns the projection matrices and graph
        nodes used by the module.
    """

    def __init__(
        self,
        in_channels: int,
        num_nodes: int = 8,
        graph_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        dropout: float = 0.0,
        residual: bool = True,
        mask_threshold: float = 0.5,
        eps: float = 1e-6,
    ) -> None:
        super(CausalityDrivenGraphReasoningModule, self).__init__()
        if in_channels <= 0:
            raise ValueError("in_channels must be positive.")
        if num_nodes <= 0:
            raise ValueError("num_nodes must be positive.")

        graph_channels = graph_channels or in_channels
        out_channels = out_channels or in_channels

        self.in_channels = in_channels
        self.num_nodes = num_nodes
        self.graph_channels = graph_channels
        self.out_channels = out_channels
        self.residual = residual and (in_channels == out_channels)
        self.mask_threshold = mask_threshold
        self.eps = eps

        self.causal_projection = nn.Conv2d(in_channels, num_nodes, kernel_size=1)
        self.non_causal_projection = nn.Conv2d(in_channels, num_nodes, kernel_size=1)
        self.causal_context = nn.Conv2d(in_channels, num_nodes, kernel_size=1)
        self.non_causal_context = nn.Conv2d(in_channels, num_nodes, kernel_size=1)

        total_nodes = num_nodes * 2
        self.adjacency = nn.Parameter(torch.empty(total_nodes, total_nodes))
        self.register_buffer("identity", torch.eye(total_nodes), persistent=False)

        self.graph_norm = nn.LayerNorm(in_channels)
        self.graph_fc1 = nn.Linear(in_channels, graph_channels, bias=False)
        self.graph_fc2 = nn.Linear(graph_channels, in_channels, bias=False)
        self.graph_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.causal_reproject = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.non_causal_reproject = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_normal_(self.causal_projection.weight, mode="fan_out", nonlinearity="relu")
        nn.init.kaiming_normal_(self.non_causal_projection.weight, mode="fan_out", nonlinearity="relu")
        nn.init.kaiming_normal_(self.causal_context.weight, mode="fan_out", nonlinearity="relu")
        nn.init.kaiming_normal_(self.non_causal_context.weight, mode="fan_out", nonlinearity="relu")
        nn.init.zeros_(self.causal_projection.bias)
        nn.init.zeros_(self.non_causal_projection.bias)
        nn.init.zeros_(self.causal_context.bias)
        nn.init.zeros_(self.non_causal_context.bias)
        nn.init.uniform_(self.adjacency, -0.02, 0.02)

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: Optional[torch.Tensor] = None,
        non_causal_mask: Optional[torch.Tensor] = None,
        coarse_logits: Optional[torch.Tensor] = None,
        return_debug: bool = False,
    ) -> TensorOrDebug:
        if x.dim() != 4:
            raise ValueError("x must have shape [B, C, H, W].")
        if x.size(1) != self.in_channels:
            raise ValueError(
                "Expected x to have {} channels, got {}.".format(self.in_channels, x.size(1))
            )

        batch_size, channels, height, width = x.shape
        spatial_size = (height, width)

        if causal_mask is None or non_causal_mask is None:
            if coarse_logits is None:
                raise ValueError("Pass causal/non_causal masks or coarse_logits.")
            causal_mask, non_causal_mask = self._masks_from_coarse_logits(coarse_logits, spatial_size)
        else:
            causal_mask = self._prepare_mask(causal_mask, spatial_size, x.dtype, x.device)
            non_causal_mask = self._prepare_mask(non_causal_mask, spatial_size, x.dtype, x.device)
        causal_mask = causal_mask.to(device=x.device, dtype=x.dtype)
        non_causal_mask = non_causal_mask.to(device=x.device, dtype=x.dtype)

        projection_c = self._projection_matrix(
            x=x,
            mask=causal_mask,
            projection=self.causal_projection,
            context_projection=self.causal_context,
        )
        projection_n = self._projection_matrix(
            x=x,
            mask=non_causal_mask,
            projection=self.non_causal_projection,
            context_projection=self.non_causal_context,
        )

        x_flat = x.flatten(2).transpose(1, 2)  # [B, HW, C]
        graph_c = torch.bmm(projection_c, x_flat)  # [B, K, C]
        graph_n = torch.bmm(projection_n, x_flat)  # [B, K, C]
        graph = torch.cat([graph_c, graph_n], dim=1)  # [B, 2K, C]

        graph_hat = self._graph_convolution(graph)
        graph_hat_c, graph_hat_n = torch.split(graph_hat, self.num_nodes, dim=1)

        feature_c = torch.bmm(projection_c.transpose(1, 2), graph_hat_c)
        feature_n = torch.bmm(projection_n.transpose(1, 2), graph_hat_n)
        feature_c = feature_c.transpose(1, 2).contiguous().view(batch_size, channels, height, width)
        feature_n = feature_n.transpose(1, 2).contiguous().view(batch_size, channels, height, width)

        feature_c = self.causal_reproject(feature_c)
        feature_n = self.non_causal_reproject(feature_n)
        out = self.fuse(torch.cat([feature_c, feature_n], dim=1))
        if self.residual:
            out = out + x

        if not return_debug:
            return out

        debug = {
            "causal_mask": causal_mask.detach(),
            "non_causal_mask": non_causal_mask.detach(),
            "projection_c": projection_c.detach(),
            "projection_n": projection_n.detach(),
            "adjacency": self._normalized_adjacency().detach(),
            "graph_c": graph_c.detach(),
            "graph_n": graph_n.detach(),
            "graph_hat": graph_hat.detach(),
            "feature_c": feature_c.detach(),
            "feature_n": feature_n.detach(),
        }
        return out, debug

    def _projection_matrix(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        projection: nn.Conv2d,
        context_projection: nn.Conv2d,
    ) -> torch.Tensor:
        batch_size, _, height, width = x.shape
        spatial_count = height * width

        safe_mask = self._fallback_if_empty(mask)
        masked_mean = (x * safe_mask).sum(dim=(2, 3), keepdim=True)
        masked_mean = masked_mean / safe_mask.sum(dim=(2, 3), keepdim=True).clamp_min(self.eps)

        logits = projection(x) + context_projection(masked_mean)
        logits = logits + torch.log(safe_mask.clamp_min(self.eps))
        logits = logits.view(batch_size, self.num_nodes, spatial_count)
        return F.softmax(logits, dim=-1)

    def _graph_convolution(self, graph: torch.Tensor) -> torch.Tensor:
        residual = graph
        graph = self.graph_norm(graph)
        adjacency = self._normalized_adjacency()

        graph = torch.einsum("ij,bjc->bic", adjacency, graph)
        graph = self.graph_fc1(graph)
        graph = F.relu(graph, inplace=True)
        graph = self.graph_dropout(graph)
        graph = self.graph_fc2(graph)
        return graph + residual

    def _normalized_adjacency(self) -> torch.Tensor:
        adjacency = self.adjacency + self.identity.to(self.adjacency.device, self.adjacency.dtype)
        return F.softmax(adjacency, dim=-1)

    def _masks_from_coarse_logits(
        self,
        coarse_logits: torch.Tensor,
        spatial_size: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if coarse_logits.dim() == 3:
            labels = coarse_logits.unsqueeze(1)
            non_causal_mask = (labels == 0).to(dtype=torch.float32)
            causal_mask = (labels > 0).to(dtype=torch.float32)
            causal_mask = self._prepare_mask(causal_mask, spatial_size, torch.float32, coarse_logits.device)
            non_causal_mask = self._prepare_mask(non_causal_mask, spatial_size, torch.float32, coarse_logits.device)
            return causal_mask, non_causal_mask
        if coarse_logits.dim() != 4:
            raise ValueError("coarse_logits must have shape [B, C, H, W] or [B, H, W].")

        if not torch.is_floating_point(coarse_logits):
            labels = coarse_logits
            non_causal_mask = (labels == 0).to(dtype=torch.float32)
            causal_mask = (labels > 0).to(dtype=torch.float32)
            causal_mask = self._prepare_mask(causal_mask, spatial_size, torch.float32, coarse_logits.device)
            non_causal_mask = self._prepare_mask(non_causal_mask, spatial_size, torch.float32, coarse_logits.device)
            return causal_mask, non_causal_mask

        if coarse_logits.size(1) == 1:
            prob = torch.sigmoid(coarse_logits)
            causal_mask = prob
            non_causal_mask = 1.0 - causal_mask
        else:
            prob = F.softmax(coarse_logits, dim=1)
            non_causal_mask = prob[:, 0:1]
            causal_mask = prob[:, 1:].sum(dim=1, keepdim=True)

        causal_mask = self._prepare_mask(
            causal_mask, spatial_size, coarse_logits.dtype, coarse_logits.device
        )
        non_causal_mask = self._prepare_mask(
            non_causal_mask, spatial_size, coarse_logits.dtype, coarse_logits.device
        )
        return causal_mask, non_causal_mask

    @staticmethod
    def _prepare_mask(
        mask: torch.Tensor,
        spatial_size: Tuple[int, int],
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        if mask.dim() != 4:
            raise ValueError("mask must have shape [B, 1, H, W] or [B, H, W].")
        if mask.size(1) != 1:
            mask = mask.max(dim=1, keepdim=True).values

        mask = mask.to(device=device, dtype=dtype)
        if mask.shape[-2:] != spatial_size:
            mask = F.interpolate(mask, size=spatial_size, mode="nearest")
        return mask.clamp(0.0, 1.0)

    def _fallback_if_empty(self, mask: torch.Tensor) -> torch.Tensor:
        flat_sum = mask.sum(dim=(2, 3), keepdim=True)
        full_mask = torch.ones_like(mask)
        return torch.where(flat_sum > self.eps, mask, full_mask)


CGRM = CausalityDrivenGraphReasoningModule


if __name__ == "__main__":
    module = CGRM(in_channels=64, num_nodes=6)
    feature = torch.randn(2, 64, 32, 32)
    logits = torch.randn(2, 2, 64, 64)
    refined, info = module(feature, coarse_logits=logits, return_debug=True)
    print("refined:", tuple(refined.shape))
    print("projection_c:", tuple(info["projection_c"].shape))
