"""VGGT-style alternating-attention multi-view fusion for WiLoR.

Operates on the frozen ViT trunk's full token sequences (one per view-crop of the SAME
physical hand): layers alternate GLOBAL self-attention (tokens of all V views concatenated,
so evidence flows between views) with FRAME self-attention (per-view, same shape the
pretrained ViT blocks were trained on). Attention is length-agnostic, so any V works —
callers bucket hands by view count instead of padding.

Two pretraining-preservation tricks:
  * each fusion layer is a deep copy of one of the pretrained ViT's last blocks
    (same dim/heads/MLP — warm start);
  * every layer is gated by a zero-init per-channel LayerScale, so at initialization the
    whole module is exactly the identity and the model reproduces single-view WiLoR.
"""
import copy
from typing import List, Optional

import torch
import torch.nn as nn


class CameraExtrinsicsEmbed(nn.Module):
    """Embed each view's camera pose into a per-view token-bias vector ``(G, V, dim)``.

    The pose is encoded RELATIVE to the group's first view (index 0), which makes the signal
    invariant to the per-session/per-dataset world origin & scale (within a (frame, hand_id)
    group all V views share a common world frame). Rotation uses the 6D continuity rep (first
    two columns of R, already bounded — no normalization); translation is divided by a fixed
    ``t_scale`` (metres) so it sits in a comparable range. A LayerNorm normalizes the *learned*
    features after the first Linear, not the raw geometry.

    The final Linear is ZERO-INITIALIZED so the embedding is all-zeros at init; added to the
    token stream this is an exact no-op, preserving MultiViewFusion's identity-at-init property.
    """

    def __init__(self, dim: int, t_scale: float = 0.5, hidden: int = 256):
        super().__init__()
        self.t_scale = t_scale
        self.mlp = nn.Sequential(
            nn.Linear(9, hidden),       # 6D rotation + 3D translation
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, extrinsics: torch.Tensor) -> torch.Tensor:
        # extrinsics: (G, V, 3, 4) = [R_world_to_cam | t_world_to_cam]
        R = extrinsics[..., :3]   # (G, V, 3, 3)
        t = extrinsics[..., 3]    # (G, V, 3)

        # pose relative to view 0: E_v ∘ E_0^{-1}. View 0 -> (I, 0); V=1 -> (I, 0) too.
        R0 = R[:, 0:1]            # (G, 1, 3, 3)
        t0 = t[:, 0:1]            # (G, 1, 3)
        R_rel = R @ R0.transpose(-1, -2)                          # (G, V, 3, 3)
        t_rel = t - (R_rel @ t0.unsqueeze(-1)).squeeze(-1)        # (G, V, 3)

        # 6D rotation rep: first two columns of R_rel, flattened.
        rot6d = R_rel[..., :2].reshape(*R_rel.shape[:-2], 6)      # (G, V, 6)
        t_feat = t_rel / self.t_scale                            # (G, V, 3)
        x = torch.cat([rot6d, t_feat], dim=-1)                   # (G, V, 9)
        return self.mlp(x)                                       # (G, V, dim)


class _GatedBlock(nn.Module):
    """Residual-gated transformer block: ``x + gamma * (block(x) - x)``.

    ``gamma`` is a zero-init per-channel LayerScale, making the layer an exact identity at
    init (block weights — pretrained copies — only blend in as gamma is learned).
    """

    def __init__(self, block: nn.Module, dim: int):
        super().__init__()
        self.block = block
        self.gamma = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.gamma * (self.block(x) - x)


class MultiViewFusion(nn.Module):
    """Alternating global/frame attention over per-view token sequences.

    forward input/output: ``(G, V, N, C)`` — G hand-groups, V views each, N tokens, C dim.
    Even layers attend globally over all ``V*N`` tokens of a group; odd layers attend within
    each view's ``N`` tokens (matching the user-facing spec: global first, then per-frame).
    """

    def __init__(
        self,
        blocks: List[nn.Module],
        dim: int,
        fuse_camera_extrinsics: bool = False,
        t_scale: float = 0.5,
        embed_hidden: int = 256,
    ):
        super().__init__()
        self.layers = nn.ModuleList(_GatedBlock(b, dim) for b in blocks)
        self.cam_embed = (
            CameraExtrinsicsEmbed(dim, t_scale=t_scale, hidden=embed_hidden)
            if fuse_camera_extrinsics else None
        )

    @classmethod
    def from_backbone(
        cls,
        backbone: nn.Module,
        num_layers: int = 8,
        fuse_camera_extrinsics: bool = False,
        t_scale: float = 0.5,
        embed_hidden: int = 256,
    ) -> "MultiViewFusion":
        """Build fusion layers as deep copies of the backbone's last ``num_layers`` blocks.

        The copies drop stochastic depth (the tail ViT-H blocks carry drop_path up to ~0.55,
        which would make a *frozen-trunk* adapter needlessly noisy).
        """
        blocks = []
        for src in backbone.blocks[-num_layers:]:
            blk = copy.deepcopy(src)
            blk.drop_path = nn.Identity()
            blocks.append(blk)
        return cls(
            blocks,
            dim=backbone.embed_dim,
            fuse_camera_extrinsics=fuse_camera_extrinsics,
            t_scale=t_scale,
            embed_hidden=embed_hidden,
        )

    def forward(self, tokens: torch.Tensor, extrinsics: Optional[torch.Tensor] = None) -> torch.Tensor:
        G, V, N, C = tokens.shape
        x = tokens
        if self.cam_embed is not None:
            # Per-view camera-pose bias, broadcast over the N tokens and added once at entry on
            # the residual stream (outside the gated blocks, so it rides through all layers and
            # lets the global-attention layers tag tokens by camera). The .to(dtype) cast keeps
            # the add safe under 16-mixed autocast; cam_embed is zero at init (zero-init final
            # Linear), so this stays an exact no-op until trained -> identity-at-init preserved.
            x = x + self.cam_embed(extrinsics)[:, :, None, :].to(tokens.dtype)
        for i, layer in enumerate(self.layers):
            if i % 2 == 0:  # global: all views of the group in one sequence
                x = layer(x.reshape(G, V * N, C)).reshape(G, V, N, C)
            else:           # frame: each view independently
                x = layer(x.reshape(G * V, N, C)).reshape(G, V, N, C)
        return x
