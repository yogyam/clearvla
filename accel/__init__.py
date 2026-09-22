"""Acceleration configs for Model A (Week 4). `make_accel(name)` -> Accel object consumed by eval.policy.ModelAPolicy.

Names: full | prune50 prune25 prune12 | cache50 cache25 cache12 | quant4 | protect_prune25 protect_cache25
Budget = fraction of the visual tokens (both views, 392 in v4) that get full prefix compute after layer 2.
Protect = the same as prune25 / cache25 but tokens overlapping the GT critical mask (>= 25 % of the patch) are always
kept / recomputed; the GT mask comes from the simulator, so Protect is an upper bound (H5)."""
from __future__ import annotations
import re
import numpy as np, torch
from accel.prune import prune_scores, topk_mask
from accel.cache import changed_mask, KVCache
from accel.quant import fake_quant_model
from accel.cost import cost

BUDGETS = {"50": 0.5, "25": 0.25, "12": 0.125, "100": 1.0}


class Accel:
    def __init__(self, name: str):
        self.name = name
        m = re.fullmatch(r"(protect_)?(prune|cache)(\d+)", name)
        if name == "full": self.method, self.budget, self.protect = "full", 1.0, False
        elif name == "quant4": self.method, self.budget, self.protect = "quant", 1.0, False
        elif m: self.protect, self.method, self.budget = bool(m.group(1)), m.group(2), BUDGETS[m.group(3)]
        else: raise ValueError(f"unknown accel config {name}")
        self.needs_gt = self.protect
        self.cache = KVCache() if self.method == "cache" else None
        self.last_keep = None; self.last_cost = None; self.gt_crit = None; self.quant_stats = None
        self._prev_imgs = None

    def prepare(self, model):
        if self.method == "quant": self.quant_stats = fake_quant_model(model)
        return self

    def reset(self):
        self._prev_imgs = None; self.last_keep = None
        if self.cache is not None: self.cache.reset()

    def act_kwargs(self, n_vis: int, imgs: list, txt_mask: torch.Tensor, dev) -> dict:
        """Called once per observation before model.act. `imgs`: list of uint8 views in token order."""
        k = int(round(self.budget * n_vis)); n_txt_valid = int(txt_mask.sum())
        protect = None
        if self.protect:
            assert self.gt_crit is not None, "protect configs need policy.accel.gt_crit set by the evaluator"
            protect = np.asarray(self.gt_crit, bool)
        if self.method in ("full", "quant"):
            self.last_keep = np.ones(n_vis, bool); self.last_cost = cost(self.method, n_vis, n_txt_valid, n_vis); return {}
        if self.method == "prune":
            prot_t = None if protect is None else torch.from_numpy(protect)[None].to(dev)
            def keep_fn(att, key_mask):
                keep = topk_mask(prune_scores(att.float(), n_vis, key_mask), k, prot_t)
                self.last_keep = keep[0].cpu().numpy(); return keep
            self.last_cost = cost("prune", n_vis, n_txt_valid, k)     # updated after the forward if protect enlarged it
            return dict(keep_fn=keep_fn)
        # cache
        if self._prev_imgs is None:
            self.cache.r_idx = None; self.last_keep = np.ones(n_vis, bool); self.last_cost = cost("full", n_vis, n_txt_valid, n_vis)
        else:
            m = changed_mask(self._prev_imgs, imgs, k, protect); self.last_keep = m
            T = n_vis + txt_mask.shape[1] + 1
            r = np.concatenate([np.nonzero(m)[0], np.arange(n_vis, T)])
            self.cache.r_idx = torch.from_numpy(r).long().to(dev); self.last_cost = cost("cache", n_vis, n_txt_valid, int(m.sum()))
        self._prev_imgs = [np.asarray(im).copy() for im in imgs]
        return dict(cache=self.cache)

    def finish(self, n_vis: int, txt_mask: torch.Tensor):
        """After the forward: with protect the kept count can exceed the budget; recompute the cost from the actual keep."""
        if self.method == "prune" and self.last_keep is not None:
            self.last_cost = cost("prune", n_vis, int(txt_mask.sum()), int(self.last_keep.sum()))


def make_accel(name: str) -> Accel | None:
    return None if name == "full" else Accel(name)
