"""Latency of Model A's accelerated region on an NVIDIA L4 (Modal): eager fp16 vs torch.compile (CUDA graphs) vs
TensorRT (torch_tensorrt, fp16, fixed shapes), per budget. Also SigLIP (two views) eager vs compiled.

  modal volume put clearvla-data checkpoints/v4/best.pt /models/v4/best.pt
  modal run --detach jobs/latency_trt.py
Writes /vol/latency/l4.json and prints it. Exits when done (never idles).
"""
import modal, json, os, time

APP = "clearvla-latency"; VOL = modal.Volume.from_name("clearvla-data")
image = (modal.Image.from_registry("nvcr.io/nvidia/pytorch:25.06-py3")
         .pip_install("transformers==4.46.3")
         .add_local_python_source("model"))
app = modal.App(APP, image=image)


@app.function(gpu="L4", timeout=3600, volumes={"/vol": VOL})
def run(n: int = 300):
    import torch, torch.nn as nn, numpy as np
    from model.vla import ClearVLA
    dev = torch.device("cuda"); torch.backends.cuda.matmul.allow_tf32 = True
    state = torch.load("/vol/models/v4/best.pt", map_location="cpu")
    n_vis = state["model"]["prefix.pos_vis"].shape[1]
    model = ClearVLA(n_vis=n_vis).to(dev).eval(); model.load_state_dict(state["model"]); model.half()
    vis = torch.randn(1, n_vis, 768, device=dev, dtype=torch.half); txt = torch.randn(1, 64, 768, device=dev, dtype=torch.half)
    txt_mask = torch.zeros(1, 64, dtype=torch.bool, device=dev); txt_mask[0, :12] = True; prop = torch.randn(1, 19, device=dev, dtype=torch.half)
    budgets = {"full": 1.0, "prune50": 0.5, "prune25": 0.25, "prune12": 0.125}
    info = dict(gpu=torch.cuda.get_device_name(0), torch=torch.__version__, n=n, runs=[])

    def timeit(fn, n):
        for _ in range(20): fn()
        torch.cuda.synchronize(); ts = []
        for _ in range(n):
            t0 = time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
        return float(np.median(ts) * 1000)

    class Head(nn.Module):                     # embed + layers 0..2 on all tokens
        def __init__(s, p): super().__init__(); s.p = p
        def forward(s, vis, txt, prop, key_mask):
            x = s.p.embed(vis, txt, prop)
            for blk in s.p.blocks[:3]: x, _, _ = blk(x, key_mask=key_mask)
            return x
    class Tail(nn.Module):                     # layers 3..5 + ln_f on the kept tokens
        def __init__(s, p): super().__init__(); s.p = p
        def forward(s, x, key_mask):
            for blk in s.p.blocks[3:]: x, _, _ = blk(x, key_mask=key_mask)
            return s.p.ln_f(x)
    class Step(nn.Module):                     # one Euler step of the expert (K/V of memory recomputed; no cache)
        def __init__(s, e): super().__init__(); s.e = e
        def forward(s, x, t, mem, mask): v, _ = s.e(x, t, mem, mask); return x + v / 10
    head, tail, step = Head(model.prefix).eval(), Tail(model.prefix).eval(), Step(model.expert).eval()
    T = n_vis + 65; key_mask = torch.cat([torch.ones(1, n_vis, dtype=torch.bool, device=dev), txt_mask, torch.ones(1, 1, dtype=torch.bool, device=dev)], 1)

    def make_fns(h, tl, st, budget):
        k = int(round(budget * n_vis)); Tk = k + 65
        idx = torch.cat([torch.arange(k, device=dev), torch.arange(n_vis, T, device=dev)])
        km = key_mask[:, idx].contiguous(); x0 = torch.randn(1, 16, 10, device=dev, dtype=torch.half); t = torch.full((1,), 0.5, device=dev, dtype=torch.half)
        def full_obs():
            with torch.no_grad():
                x = h(vis, txt, prop, key_mask); xk = x[:, idx]; mem = tl(xk, km); a = x0
                for i in range(10): a = st(a, t, mem, km)
            return a
        return full_obs, Tk

    with torch.no_grad():
        for mode in ("eager", "compile", "tensorrt"):
            for name, b in budgets.items():
                k = int(round(b * n_vis)); Tk = k + 65
                try:
                    if mode == "eager": h, tl, st = head, tail, step
                    elif mode == "compile":
                        h = torch.compile(head, mode="reduce-overhead"); tl = torch.compile(tail, mode="reduce-overhead"); st = torch.compile(step, mode="reduce-overhead")
                    else:
                        import torch_tensorrt as trt
                        ex_x = torch.randn(1, Tk, 512, device=dev, dtype=torch.half); ex_km = key_mask[:, :Tk].contiguous(); ex_mem = torch.randn(1, Tk, 512, device=dev, dtype=torch.half)
                        h = trt.compile(head, ir="dynamo", inputs=[vis, txt, prop, key_mask], enabled_precisions={torch.half}, min_block_size=1)
                        tl = trt.compile(tail, ir="dynamo", inputs=[ex_x, ex_km], enabled_precisions={torch.half}, min_block_size=1)
                        st = trt.compile(step, ir="dynamo", inputs=[torch.randn(1, 16, 10, device=dev, dtype=torch.half), torch.full((1,), 0.5, device=dev, dtype=torch.half), ex_mem, ex_km], enabled_precisions={torch.half}, min_block_size=1)
                    fn, _ = make_fns(h, tl, st, b); ms = timeit(fn, n)
                    info["runs"].append(dict(mode=mode, config=name, tokens_after_l2=Tk, ms_per_obs=ms)); print(mode, name, f"{ms:.2f} ms", flush=True)
                except Exception as e:
                    info["runs"].append(dict(mode=mode, config=name, error=repr(e)[:300])); print(mode, name, "ERROR", repr(e)[:200], flush=True)
        # SigLIP, two views
        try:
            from transformers import AutoModel
            sig = AutoModel.from_pretrained("google/siglip-base-patch16-224").vision_model.to(dev).half().eval(); px = torch.randn(2, 3, 224, 224, device=dev, dtype=torch.half)
            ms = timeit(lambda: sig(pixel_values=px).last_hidden_state, n); info["siglip_two_views_eager_ms"] = ms; print("siglip eager", ms)
            sigc = torch.compile(sig, mode="reduce-overhead"); ms = timeit(lambda: sigc(pixel_values=px).last_hidden_state, n); info["siglip_two_views_compile_ms"] = ms; print("siglip compile", ms)
        except Exception as e: info["siglip_error"] = repr(e)[:300]; print("siglip ERROR", repr(e)[:200])
    os.makedirs("/vol/latency", exist_ok=True); json.dump(info, open("/vol/latency/l4.json", "w"), indent=1); VOL.commit()
    return info


@app.local_entrypoint()
def main(n: int = 300):
    info = run.remote(n); print(json.dumps(info, indent=1))
