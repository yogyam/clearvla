"""Cache (VLA-Cache-style): per observation, compare each 16x16 pixel patch with the same patch in the previous
observation (cosine similarity of raw pixels). The b most-changed visual tokens (jointly over both views) are
recomputed through prefix layers 3-6; the rest reuse the K/V and layer outputs cached from the previous observation.
Text and proprio tokens are always recomputed. The first observation of an episode is computed in full.
The pixel-change criterion is the assumption under test (glass changes the pixels less than it changes the state)."""
from __future__ import annotations
import numpy as np, torch


def patch_vectors(img_uint8: np.ndarray, patch: int = 16) -> np.ndarray:
    """(224,224,3) uint8 -> (196, 768) float32 patch vectors in raster order (matches SigLIP's token order)."""
    h, w, c = img_uint8.shape; g = h // patch
    return img_uint8.reshape(g, patch, g, patch, c).transpose(0, 2, 1, 3, 4).reshape(g * g, patch * patch * c).astype(np.float32)


def patch_similarity(prev: np.ndarray, cur: np.ndarray) -> np.ndarray:
    """Cosine similarity per patch between two frames of one view -> (196,). 1 = unchanged."""
    a, b = patch_vectors(prev), patch_vectors(cur)
    a = a - a.mean(1, keepdims=True); b = b - b.mean(1, keepdims=True)      # remove per-patch brightness
    num = (a * b).sum(1); den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-6
    return num / den


def changed_mask(prev_imgs, cur_imgs, k: int, protect: np.ndarray | None = None) -> np.ndarray:
    """Boolean (n_vis,) recompute mask: the k least-similar patches over all views (plus protected ones)."""
    sim = np.concatenate([patch_similarity(p, c) for p, c in zip(prev_imgs, cur_imgs)])
    if protect is not None: sim = sim.copy(); sim[protect] = -np.inf
    kk = max(k, int(protect.sum()) if protect is not None else 0)
    idx = np.argsort(sim)[:kk]
    m = np.zeros(sim.shape[0], dtype=bool); m[idx] = True
    return m


class KVCache:
    """Per-layer store used by model.vla.Prefix.forward. `r_idx` (LongTensor of token indices) is set by the accel
    object before each observation; None means compute everything and (re)fill the cache."""
    def __init__(self): self.layers = {}; self.r_idx = None
    def reset(self): self.layers = {}; self.r_idx = None
    def ready(self, i): return self.r_idx is not None and i in self.layers
    def store(self, i, x_in, kv, x_out): self.layers[i] = (x_in, (kv[0], kv[1]), x_out)
    def apply(self, blk, i, x, key_mask):
        x_in_c, kv_c, out_c = self.layers[i]
        x_out, kv, att = blk.forward_partial(x, self.r_idx, kv_c, out_c, key_mask=key_mask)
        self.layers[i] = (x, kv, x_out)
        return x_out, att
