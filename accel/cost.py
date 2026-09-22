"""Analytic FLOPs per observation for Model A's accelerated region (prefix + expert), from the token counts each
method actually uses. SigLIP (frozen, two views) is constant across methods and reported separately."""
from __future__ import annotations

D, H, L_PREFIX, PRUNE_AFTER = 512, 8, 6, 2
D_EXP, L_EXP, CHUNK, N_STEPS = 384, 4, 16, 10
SIGLIP_GFLOPS_PER_VIEW = 2 * 86e6 * 196 / 1e9 * 1.0        # ~ViT-B/16 @224: 2*params*tokens ≈ 33.7 GFLOPs


def layer_flops(n_q: int, n_k: int, n_kv_new: int, d: int = D) -> float:
    """One pre-norm encoder layer: q,o projections for n_q tokens; k,v projections for n_kv_new tokens;
    QK^T and AV over n_q x n_k; MLP (4x) for n_q tokens. FLOPs = 2 * MACs."""
    proj = 2 * (2 * n_q * d * d + 2 * n_kv_new * d * d)
    attn = 2 * (2 * n_q * n_k * d)
    mlp = 2 * (2 * n_q * d * 4 * d)
    return proj + attn + mlp


def prefix_flops(method: str, n_vis: int, n_txt_valid: int, k: int) -> float:
    """k = visual tokens with full compute after layer 2 (kept for prune, recomputed for cache)."""
    T = n_vis + n_txt_valid + 1; f = 0.0
    for i in range(L_PREFIX):
        if i <= PRUNE_AFTER or method in ("full", "quant"): f += layer_flops(T, T, T)
        elif method.endswith("prune"): Tk = k + n_txt_valid + 1; f += layer_flops(Tk, Tk, Tk)
        elif method.endswith("cache"): R = k + n_txt_valid + 1; f += layer_flops(R, T, R)
        else: raise ValueError(method)
    return f


def expert_flops(n_mem: int) -> float:
    """4 decoder layers x 10 Euler steps over a 16-token chunk; cross-attention K/V of the memory computed once."""
    d = D_EXP; per_step = 0.0
    for _ in range(L_EXP):
        per_step += layer_flops(CHUNK, CHUNK, CHUNK, d)                      # self-attention block
        per_step += 2 * (2 * CHUNK * d * d) + 2 * (2 * CHUNK * n_mem * d)     # cross q,o + QK/AV
    kv_once = L_EXP * 2 * (2 * n_mem * D * d)
    return per_step * N_STEPS + kv_once


def cost(method: str, n_vis: int, n_txt_valid: int, k: int) -> dict:
    n_mem = (k if method.endswith("prune") else n_vis) + n_txt_valid + 1
    pf, ef = prefix_flops(method, n_vis, n_txt_valid, k), expert_flops(n_mem)
    full = prefix_flops("full", n_vis, n_txt_valid, n_vis) + expert_flops(n_vis + n_txt_valid + 1)
    return dict(prefix_gflops=pf / 1e9, expert_gflops=ef / 1e9, total_gflops=(pf + ef) / 1e9, rel_to_full=(pf + ef) / full,
                siglip_gflops=SIGLIP_GFLOPS_PER_VIEW * 2)
