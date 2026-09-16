"""Transformer blocks with explicit attention so acceleration methods can read attention maps and reuse K/V.

Nothing here uses fused attention: the attention matrix is materialised on purpose (T <= 261 tokens),
which is what FastV-style pruning needs and what VLA-Cache-style K/V reuse hooks into.
"""
from __future__ import annotations
import math
import torch, torch.nn as nn, torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, d, heads, d_kv=None):
        super().__init__()
        self.h, self.dh = heads, d // heads
        self.q = nn.Linear(d, d); self.k = nn.Linear(d_kv or d, d); self.v = nn.Linear(d_kv or d, d); self.o = nn.Linear(d, d)

    def kv(self, ctx):
        B, T, _ = ctx.shape
        k = self.k(ctx).view(B, T, self.h, self.dh).transpose(1, 2); v = self.v(ctx).view(B, T, self.h, self.dh).transpose(1, 2)
        return k, v

    def forward(self, x, ctx=None, key_mask=None, kv=None, return_attn=False):
        """x: (B,Tq,d). ctx: (B,Tk,d_kv) or None for self-attention. key_mask: (B,Tk) True = attend.
        kv: optional precomputed (k, v) to reuse (caching). Returns out, attn (B,H,Tq,Tk) or None, (k, v)."""
        B, Tq, _ = x.shape
        q = self.q(x).view(B, Tq, self.h, self.dh).transpose(1, 2)
        k, v = kv if kv is not None else self.kv(x if ctx is None else ctx)
        att = (q @ k.transpose(-1, -2)) / math.sqrt(self.dh)
        if key_mask is not None: att = att.masked_fill(~key_mask[:, None, None, :], float("-inf"))
        att = att.softmax(-1)
        out = (att @ v).transpose(1, 2).reshape(B, Tq, self.h * self.dh)
        return self.o(out), (att if return_attn else None), (k, v)


class MLP(nn.Module):
    def __init__(self, d, mult=4):
        super().__init__(); self.fc1 = nn.Linear(d, mult * d); self.fc2 = nn.Linear(mult * d, d)
    def forward(self, x): return self.fc2(F.gelu(self.fc1(x)))


class EncoderBlock(nn.Module):
    """Pre-norm self-attention block used by the prefix transformer."""
    def __init__(self, d, heads):
        super().__init__(); self.ln1 = nn.LayerNorm(d); self.attn = Attention(d, heads); self.ln2 = nn.LayerNorm(d); self.mlp = MLP(d)
    def forward(self, x, key_mask=None, kv=None, return_attn=False):
        a, att, kv_out = self.attn(self.ln1(x), key_mask=key_mask, kv=kv, return_attn=return_attn)
        x = x + a; x = x + self.mlp(self.ln2(x))
        return x, att, kv_out


class DecoderBlock(nn.Module):
    """Pre-norm block for the action expert: self-attention over the chunk + cross-attention to the prefix."""
    def __init__(self, d, heads, d_ctx):
        super().__init__()
        self.ln1 = nn.LayerNorm(d); self.self_attn = Attention(d, heads)
        self.ln2 = nn.LayerNorm(d); self.cross = Attention(d, heads, d_kv=d_ctx)
        self.ln3 = nn.LayerNorm(d); self.mlp = MLP(d)
    def forward(self, x, ctx, ctx_mask=None, ctx_kv=None):
        a, _, _ = self.self_attn(self.ln1(x)); x = x + a
        c, _, kv = self.cross(self.ln2(x), ctx=ctx, key_mask=ctx_mask, kv=ctx_kv); x = x + c
        x = x + self.mlp(self.ln3(x))
        return x, kv


def sinusoidal(t, dim):
    """t: (B,) in [0,1] -> (B,dim)."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    ang = t[:, None].float() * 1000.0 * freqs[None]
    return torch.cat([ang.sin(), ang.cos()], -1)
