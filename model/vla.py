"""ClearVLA-small (Model A).

prefix  : [196 visual | 64 text | 1 proprio] tokens -> 6-layer d=512 transformer (run once per observation)
expert  : 16-step action chunk, 4-layer d=384, cross-attends to the prefix output (run 10x for Euler steps)
training: flow matching (velocity regression); inference: 10 Euler steps from noise.

Acceleration hooks (Week 4):
  - forward_prefix(..., keep_visual=None, prune_after=2): boolean (B,196) mask applied after layer `prune_after`
    drops visual tokens from layers prune_after+1.. (FastV-style). Attention maps of every layer are returned.
  - forward_prefix(..., keep_fn=...): decide the keep set from layer-2 attention inside the forward (accel/prune.py).
  - forward_prefix(..., cache=KVCache): layers 3.. recompute only the tokens in cache.r_idx, reusing cached K/V and
    outputs for the rest (accel/cache.py, VLA-Cache-style).
"""
from __future__ import annotations
import torch, torch.nn as nn
from model.blocks import EncoderBlock, DecoderBlock, sinusoidal

N_VIS, N_TXT, D_ENC = 196, 64, 768
ACT_DIM, CHUNK = 10, 16
PROPRIO_DIM = 19


class Prefix(nn.Module):
    def __init__(self, d=512, heads=8, layers=6, n_vis=N_VIS):
        super().__init__(); self.n_vis = n_vis
        self.vis_proj = nn.Linear(D_ENC, d); self.txt_proj = nn.Linear(D_ENC, d)
        self.prop_mlp = nn.Sequential(nn.Linear(PROPRIO_DIM, d), nn.GELU(), nn.Linear(d, d))
        self.pos_vis = nn.Parameter(torch.randn(1, n_vis, d) * 0.02); self.pos_txt = nn.Parameter(torch.randn(1, N_TXT, d) * 0.02)
        self.pos_prop = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.type_emb = nn.Parameter(torch.randn(3, d) * 0.02)
        self.blocks = nn.ModuleList([EncoderBlock(d, heads) for _ in range(layers)])
        self.ln_f = nn.LayerNorm(d)

    def embed(self, vis, txt, prop):
        v = self.vis_proj(vis) + self.pos_vis + self.type_emb[0]
        t = self.txt_proj(txt) + self.pos_txt + self.type_emb[1]
        p = self.prop_mlp(prop)[:, None] + self.pos_prop + self.type_emb[2]
        return torch.cat([v, t, p], 1)

    def forward(self, vis, txt, txt_mask, prop, keep_visual=None, prune_after=2, return_attn=False, keep_fn=None, cache=None):
        """Returns memory (B,T',d), key mask (B,T'), token index (B,T') into the original slots, attn maps.
        keep_fn(att, key_mask) -> (B,n_vis) bool is called with layer `prune_after`'s attention to decide the keep set
        (Prune / Protect). `cache` is an accel.cache.KVCache: layers after `prune_after` recompute only cache.r_idx."""
        B = vis.shape[0]
        x = self.embed(vis, txt, prop)
        key_mask = torch.cat([torch.ones(B, self.n_vis, dtype=torch.bool, device=x.device), txt_mask.bool(),
                              torch.ones(B, 1, dtype=torch.bool, device=x.device)], 1)
        index = torch.arange(x.shape[1], device=x.device)[None].expand(B, -1)
        attns = []
        for i, blk in enumerate(self.blocks):
            need_att = return_attn or (i == prune_after and (keep_visual is not None or keep_fn is not None))
            if cache is not None and i > prune_after and cache.ready(i):
                x, att = cache.apply(blk, i, x, key_mask)
            else:
                x_in = x
                x, att, kv = blk(x, key_mask=key_mask, return_attn=need_att)
                if cache is not None and i > prune_after: cache.store(i, x_in, kv, x)
            if return_attn: attns.append(att)
            if i == prune_after and keep_visual is None and keep_fn is not None: keep_visual = keep_fn(att, key_mask)
            if keep_visual is not None and i == prune_after:
                keep = torch.cat([keep_visual.bool(), torch.ones(B, x.shape[1] - self.n_vis, dtype=torch.bool, device=x.device)], 1)
                # all rows keep the same count (budget), so gather to a dense tensor
                n = int(keep[0].sum()); idx = keep.nonzero()[:, 1].view(B, n)
                x = torch.gather(x, 1, idx[..., None].expand(-1, -1, x.shape[-1]))
                key_mask = torch.gather(key_mask, 1, idx); index = torch.gather(index, 1, idx)
        return self.ln_f(x), key_mask, index, attns


class ActionExpert(nn.Module):
    def __init__(self, d=384, heads=8, layers=4, d_ctx=512):
        super().__init__()
        self.act_in = nn.Linear(ACT_DIM, d); self.pos = nn.Parameter(torch.randn(1, CHUNK, d) * 0.02)
        self.t_mlp = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d)); self.d = d
        self.blocks = nn.ModuleList([DecoderBlock(d, heads, d_ctx) for _ in range(layers)])
        self.ln_f = nn.LayerNorm(d); self.out = nn.Linear(d, ACT_DIM)

    def forward(self, x_t, t, memory, mem_mask, ctx_kv=None):
        h = self.act_in(x_t) + self.pos + self.t_mlp(sinusoidal(t, self.d))[:, None]
        kvs = []
        for i, blk in enumerate(self.blocks):
            h, kv = blk(h, memory, mem_mask, ctx_kv=None if ctx_kv is None else ctx_kv[i]); kvs.append(kv)
        return self.out(self.ln_f(h)), kvs


class ClearVLA(nn.Module):
    def __init__(self, d_prefix=512, d_expert=384, prefix_layers=6, expert_layers=4, heads=8, n_vis=N_VIS):
        super().__init__()
        self.prefix = Prefix(d_prefix, heads, prefix_layers, n_vis=n_vis); self.expert = ActionExpert(d_expert, heads, expert_layers, d_prefix)

    def n_params(self): return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def loss(self, batch):
        vis, txt, txt_mask, prop, act, act_mask = (batch[k] for k in ("vis", "txt", "txt_mask", "prop", "act", "act_mask"))
        mem, mask, _, _ = self.prefix(vis, txt, txt_mask, prop)
        B = act.shape[0]; t = torch.rand(B, device=act.device)
        noise = torch.randn_like(act); x_t = (1 - t)[:, None, None] * noise + t[:, None, None] * act
        v, _ = self.expert(x_t, t, mem, mask)
        err = ((v.float() - (act - noise)) ** 2).mean(-1)                 # (B,16)
        return (err * act_mask).sum() / act_mask.sum()

    @torch.no_grad()
    def act(self, vis, txt, txt_mask, prop, n_steps=10, keep_visual=None, return_attn=False, keep_fn=None, cache=None):
        """Returns a normalised (B,16,10) action chunk plus prefix diagnostics."""
        mem, mask, index, attns = self.prefix(vis, txt, txt_mask, prop, keep_visual=keep_visual, return_attn=return_attn, keep_fn=keep_fn, cache=cache)
        B = vis.shape[0]; x = torch.randn(B, CHUNK, ACT_DIM, device=vis.device)
        ctx_kv = None
        for i in range(n_steps):
            t = torch.full((B,), i / n_steps, device=vis.device)
            v, kvs = self.expert(x, t, mem, mask, ctx_kv=ctx_kv)
            ctx_kv = kvs                                   # prefix K/V for cross-attention computed once, reused 9x
            x = x + v / n_steps
        return x, dict(index=index, attns=attns, mask=mask)
