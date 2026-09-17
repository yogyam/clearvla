# Validity gate — config `full`

| Task | Material | Success | 95% CI | n | Mean steps | Failures |
|---|---|---|---|---|---|---|
| grasp | opaque | **39%** | [29, 49] | 100 | 227 | xy×44, tube×17 |
| grasp | glass | **33%** | [24, 42] | 100 | 239 | xy×37, tube×30 |
| pour | opaque | **8%** | [3, 14] | 100 | 290 | spilled×50, underfilled×39, overfilled×3 |
| pour | glass | **7%** | [2, 12] | 96 | 291 | underfilled×43, spilled×43, overfilled×3 |
| insert | opaque | **33%** | [0, 100] | 3 | 224 | xy×2 |
| insert | glass | **nan%** | [nan, nan] | 0 | nan | — |

| Task | opaque − glass (pts) | 95% CI | paired n |
|---|---|---|---|
| grasp | +6 | [-8, +20] | 100 |
| pour | +1 | [-6, +8] | 96 |
| insert | +nan | [+nan, +nan] | 0 |

Per-observation timing: encoder 10 ms, policy 77 ms; 33.1 observations/episode; 19.3 s/episode wall.

**Gate (≥60% every cell, |gap| ≤ 15 pts): FAIL**
