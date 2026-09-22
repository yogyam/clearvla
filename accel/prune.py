"""Prune (FastV-style): after prefix layer 2, rank visual tokens by the attention they receive from the text and
proprio queries; keep the top-b fraction for layers 3-6. Both views are ranked jointly (one budget over 392 tokens)."""
from __future__ import annotations
import torch


def prune_scores(att: torch.Tensor, n_vis: int, key_mask: torch.Tensor) -> torch.Tensor:
    """att: (B,H,T,T) post-softmax attention of the prune layer; key_mask: (B,T) valid tokens (text padding False).
    Returns (B,n_vis): mean attention each visual token receives from the valid non-visual queries."""
    q_valid = key_mask[:, n_vis:].float()                                    # (B,Tq)
    a = att[:, :, n_vis:, :n_vis].mean(1)                                    # (B,Tq,n_vis) mean over heads
    return (a * q_valid[..., None]).sum(1) / q_valid.sum(1, keepdim=True)


def topk_mask(scores: torch.Tensor, k: int, protect: torch.Tensor | None = None) -> torch.Tensor:
    """Boolean (B,n_vis) keep mask with exactly k tokens per row (protected tokens first, then best scores).
    If more than k tokens are protected they are all kept (the extra compute is logged by the caller)."""
    B, n = scores.shape
    if protect is None: protect = torch.zeros(B, n, dtype=torch.bool, device=scores.device)
    s = scores.clone(); s[protect] = float("inf")
    kk = max(k, int(protect.sum(1).max()))
    idx = s.topk(kk, dim=1).indices
    keep = torch.zeros(B, n, dtype=torch.bool, device=scores.device); keep.scatter_(1, idx, True)
    return keep
