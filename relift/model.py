"""The corrector: a factorised space-time transformer over point tracks.

Tokens are (track, C-frame chunk) pairs.  Every block applies attention along time (within a track), attention across
tracks (within a chunk) and an MLP, each as a pre-normalised residual update.  The head is zero-initialised, so an
untrained model returns a zero correction, i.e. its initialised trajectory.

Parameter names match the released checkpoint; ``Corrector(dim=384, depth=8, heads=6, in_feats=14, chunk=4)`` is the
evaluated model (19,145,100 parameters).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

IN_FEATS = 14  # see features.features_clean for the channel layout
CHUNK = 4


def sinusoid(x: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """Sinusoidal embedding of x (...,) -> (..., dim): [sin(x w_k), cos(x w_k)], w_k = max_period^(-k/(dim/2))."""
    half = dim // 2
    f = torch.exp(
        -math.log(max_period) * torch.arange(half, device=x.device, dtype=torch.float32) / half
    )
    a = x[..., None].float() * f
    return torch.cat([a.sin(), a.cos()], -1)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (N, L, D)
        N, L, D = x.shape
        q, k, v = self.qkv(x).view(N, L, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        return self.out(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(N, L, D))


class Block(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.n1, self.n2, self.n3 = nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.time, self.space = Attention(dim, heads), Attention(dim, heads)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, P, L, D)
        B, P, L, D = x.shape
        x = x + self.time(self.n1(x).reshape(B * P, L, D)).view(B, P, L, D)
        y = self.n2(x).transpose(1, 2).reshape(B * L, P, D)
        x = x + self.space(y).view(B, L, P, D).transpose(1, 2)
        return x + self.mlp(self.n3(x))


class Corrector(nn.Module):
    def __init__(
        self,
        dim: int = 384,
        depth: int = 8,
        heads: int = 6,
        in_feats: int = IN_FEATS,
        chunk: int = CHUNK,
    ) -> None:
        super().__init__()
        self.dim, self.in_feats, self.chunk = dim, in_feats, chunk
        self.ckpt = False  # per-block activation checkpointing (training only)
        self.embed = nn.Linear(chunk * in_feats, dim)
        self.point = nn.Sequential(nn.Linear(3 * 32, dim), nn.GELU(), nn.Linear(dim, dim))
        self.blocks = nn.ModuleList(Block(dim, heads) for _ in range(depth))
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, chunk * 3)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, feats: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
        """feats (B, T, P, in_feats), anchor (B, P, 3): the reference-frame position of every track.
        Returns the scaled correction (B, T, P, 3).  T must be a multiple of the chunk size."""
        B, T, P, C = feats.shape
        K = self.chunk
        assert T % K == 0 and C == self.in_feats
        L = T // K
        x = feats.permute(0, 2, 1, 3).reshape(B, P, L, K * C)
        x = self.embed(x)
        x = x + sinusoid(torch.arange(L, device=x.device), self.dim, 1000.0)[None, None].to(x.dtype)
        x = x + self.point(sinusoid(anchor * 8.0, 32).flatten(-2).to(x.dtype))[:, :, None, :]
        for blk in self.blocks:
            x = (
                checkpoint(blk, x, use_reentrant=False)
                if self.ckpt and torch.is_grad_enabled()
                else blk(x)
            )
        v = self.head(self.norm(x)).view(B, P, L, K, 3).reshape(B, P, T, 3)
        return v.permute(0, 2, 1, 3)
