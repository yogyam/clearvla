# Week 0 feasibility results

Machine: Apple M5 Pro, 24 GB RAM, 669 GB free (2026-09-15).

| # | Check | Pass condition | Measured | Pass? | Notes |
|---|-------|----------------|----------|-------|-------|
| F1 | Gap still open | no paper tests VLA acceleration vs transparency | 14 papers reviewed; none vary material under pruning/caching/quant. Closest: SEVO (transparent + SmolVLA, no accel), VLA-IAP (pruning drops low-texture edges, no material). | **PASS** | see `paper/related_work_notes.md`. Name collision: github.com/logwood/ClearVLA |
| F2 | AutoBio on macOS | runs or fails on something fixable | ships x86-64 Linux ELF `.so` (libmjlab, meshplane); README: Linux only; references external `../assetlab` GLBs not in repo; no top-level LICENSE | **FAIL → fallback** | Build own MuJoCo scenes (Menagerie Franka). Reuse AutoBio only as reference for liquid/glass Blender setup (`render_blender.py`, `render_liquid.py`). Do not ship its assets without a license. |
| F3 | Blender render cost | ≤ 0.5 s/frame @224², glass | in-process `bpy` 5.0.1, Blender 5.2 LTS. Cycles/Metal 32 samples + denoise: glass 0.35 s, opaque 0.35 s. EEVEE 16 samples: glass 0.09 s, opaque 0.07 s. `scripts/bench_render.py`, `results/week0/render/bench_render.json` | **PASS** | Cycles for reported results (true refraction); EEVEE viable as a faster tier. 135k offline frames ≈ 13 h Cycles single process. |
| F4 | Glass looks hard | refractive + SigLIP feature diff on object | Visible refraction bands in Cycles glass. SigLIP patch cosine distance glass vs opaque: on-tube 0.73 vs off-tube 0.25 (Cycles, ratio 2.9×); EEVEE 0.65 vs 0.20 (3.3×). Tube = 8 of 196 patches at this camera. `scripts/check_glass_features.py` | **PASS** | Object is small (8 patches): keep camera framing in mind when setting 25% budget (49 tokens). |
| F5 | MPS training speed | ≥ 5 it/s, batch 64, 196 tokens, ~25M params | 29.8M params (prefix 19.97M + expert 9.83M). fp32: 2.12 it/s; fp16/bf16 autocast: 4.17 it/s (240 ms/it); batch 32 fp32: 4.21 it/s. MPS alloc 0.47 GB. torch 2.14, `scripts/bench_train.py` | **MARGINAL** (4.17 < 5) | 50k steps ≈ 3.3 h with fp16 autocast — fits overnight, so local training stays the plan. Fallback if needed: Modal L4 for training (~$3/run). |
| F6 | Episode wall time | ≤ 20 s closed-loop w/ ~19 renders | 150 steps @10 Hz, 19 obs, random policy, Franka + tube + rack, Cycles/Metal 32 spp: **7.8 s total** (Blender 357 ms/obs, MuJoCo mask 15 ms/obs, dummy Model-A inference 34 ms/obs, sim 0.05 s). Bridge setup 0.3 s once. `scripts/bench_episode.py`, `render/bridge.py` | **PASS** | 5,400 episodes ≈ 12 h single process. Blender is ~87% of episode time; EEVEE tier would cut episodes to ~2.5 s. |

## Summary (2026-09-15)

5 of 6 checks pass; F5 is marginal (4.2 it/s vs 5 target, still ≈3.3 h per 50k-step run); F2 fails as expected and the fallback (own MuJoCo scenes with Menagerie Franka + in-process Blender) is already built and benchmarked by F3/F4/F6.

**Decision: proceed to Week 1 with own scenes.**

Open design issue found during F4: in the *opaque* condition a matte-white container hides the liquid, so the pour task's fill level is invisible for opaque. Needs a resolution before Week 1 task design (see PLAN.md §0).
