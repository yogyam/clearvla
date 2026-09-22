"""Quant: fake INT4 weights, symmetric, per-group of 64 input channels, on every nn.Linear of the prefix and the
action expert. Activations stay fp16/fp32. Frozen SigLIP is not quantised (outside the accelerated region)."""
from __future__ import annotations
import torch, torch.nn as nn


def fake_quant_tensor(w: torch.Tensor, bits: int = 4, group: int = 64) -> torch.Tensor:
    """w: (out, in). Groups along the input dim; scale = max|w| / qmax per group; round-to-nearest."""
    qmax = 2 ** (bits - 1) - 1
    out, inn = w.shape; pad = (-inn) % group
    x = torch.nn.functional.pad(w, (0, pad)).view(out, -1, group)
    scale = x.abs().amax(-1, keepdim=True).clamp_min(1e-8) / qmax
    q = (x / scale).round().clamp(-qmax - 1, qmax) * scale
    return q.view(out, -1)[:, :inn]


@torch.no_grad()
def fake_quant_model(model: nn.Module, bits: int = 4, group: int = 64) -> dict:
    """In-place. Returns stats: number of linears, mean relative weight error."""
    n = 0; errs = []
    for mod in list(model.prefix.modules()) + list(model.expert.modules()):
        if isinstance(mod, nn.Linear):
            w = mod.weight.data; q = fake_quant_tensor(w.float(), bits, group).to(w.dtype)
            errs.append(((q.float() - w.float()).norm() / w.float().norm()).item()); mod.weight.data.copy_(q); n += 1
    return dict(n_linear=n, mean_rel_err=float(sum(errs) / max(len(errs), 1)), bits=bits, group=group)
