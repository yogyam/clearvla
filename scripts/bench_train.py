"""F5: MPS training-speed check for a Model-A-sized dummy VLA.

Prefix transformer (6 layers, d=512, 8 heads) over 196 visual + 16 text + 1 proprio tokens,
action expert (4 layers, d=384) cross-attending to the prefix, flow-matching MSE loss.
Inputs are random fp16 SigLIP-shaped features (the real pipeline reads them from a memmap).
Pass condition: >= 5 it/s at batch 64.
"""
import argparse, time, torch, torch.nn as nn

class PrefixTransformer(nn.Module):
    def __init__(self, d=512, heads=8, layers=6):
        super().__init__()
        self.vis_proj = nn.Linear(768, d)
        self.txt_proj = nn.Linear(768, d)
        self.prop_mlp = nn.Sequential(nn.Linear(14, d), nn.GELU(), nn.Linear(d, d))
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
    def forward(self, vis, txt, prop):
        x = torch.cat([self.vis_proj(vis), self.txt_proj(txt), self.prop_mlp(prop)[:, None]], 1)
        return self.enc(x)

class ActionExpert(nn.Module):
    def __init__(self, d=384, d_prefix=512, heads=8, layers=4, chunk=16, act_dim=10):
        super().__init__()
        self.act_in = nn.Linear(act_dim, d)
        self.t_mlp = nn.Sequential(nn.Linear(1, d), nn.SiLU(), nn.Linear(d, d))
        self.pos = nn.Parameter(torch.randn(1, chunk, d) * 0.02)
        self.kv_proj = nn.Linear(d_prefix, d)
        layer = nn.TransformerDecoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.dec = nn.TransformerDecoder(layer, layers)
        self.out = nn.Linear(d, act_dim)
    def forward(self, x_t, t, prefix):
        h = self.act_in(x_t) + self.pos + self.t_mlp(t[:, None])[:, None]
        return self.out(self.dec(h, self.kv_proj(prefix)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=64); ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--device", default="mps"); ap.add_argument("--amp", default="none", choices=["none","fp16","bf16"])
    a = ap.parse_args()
    dev = torch.device(a.device)
    prefix, expert = PrefixTransformer().to(dev), ActionExpert().to(dev)
    n_p, n_e = (sum(p.numel() for p in m.parameters()) for m in (prefix, expert))
    print(f"params: prefix {n_p/1e6:.2f}M  expert {n_e/1e6:.2f}M  total {(n_p+n_e)/1e6:.2f}M")
    opt = torch.optim.AdamW(list(prefix.parameters()) + list(expert.parameters()), lr=1e-4, weight_decay=0.01)
    B = a.batch
    vis = torch.randn(B, 196, 768, dtype=torch.float16, device=dev)
    txt = torch.randn(B, 16, 768, dtype=torch.float16, device=dev)
    prop = torch.randn(B, 14, device=dev); actions = torch.randn(B, 16, 10, device=dev)
    def step():
        noise = torch.randn_like(actions); t = torch.rand(B, device=dev)
        x_t = (1 - t)[:, None, None] * noise + t[:, None, None] * actions
        amp = a.amp != "none"
        with torch.autocast(device_type=dev.type, dtype=torch.float16 if a.amp=="fp16" else torch.bfloat16, enabled=amp):
            pred = expert(x_t, t, prefix(vis.float(), txt.float(), prop))
            loss = ((pred.float() - (actions - noise)) ** 2).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        return loss
    for _ in range(5): step()
    if dev.type == "mps": torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(a.iters): loss = step()
    if dev.type == "mps": torch.mps.synchronize()
    dt = time.perf_counter() - t0
    ips = a.iters / dt
    mem = torch.mps.current_allocated_memory() / 2**30 if dev.type == "mps" else float("nan")
    print(f"amp={a.amp} batch {B}: {ips:.2f} it/s  ({dt/a.iters*1000:.0f} ms/it)  loss {loss.item():.3f}  mps alloc {mem:.2f} GB")
    print("F5 PASS" if ips >= 5 else "F5 FAIL (need >= 5 it/s)")
if __name__ == "__main__": main()
