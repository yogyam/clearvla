# Week 0 feasibility results

Machine: Apple M5 Pro, 24 GB RAM, 669 GB free (2026-09-15).

| # | Check | Pass condition | Measured | Pass? | Notes |
|---|-------|----------------|----------|-------|-------|
| F1 | Gap still open | no paper tests VLA acceleration vs transparency | — | — | |
| F2 | AutoBio on macOS | runs or fails on something fixable | — | — | |
| F3 | Blender render cost | ≤ 0.5 s/frame @224², glass | — | — | |
| F4 | Glass looks hard | refractive + SigLIP feature diff on object | — | — | |
| F5 | MPS training speed | ≥ 5 it/s, batch 64, 196 tokens, ~25M params | — | — | |
| F6 | Episode wall time | ≤ 20 s closed-loop w/ ~19 renders | — | — | |
