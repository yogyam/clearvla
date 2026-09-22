# Validity gate — config `full`

| Task | Material | Success | 95% CI | n | Mean steps | Failures |
|---|---|---|---|---|---|---|
| grasp | opaque | **53%** | [43, 63] | 100 | 202 | xy×32, tube×13, tilted×2 |
| grasp | glass | **48%** | [38, 58] | 100 | 211 | xy×34, tube×18 |
| pour | opaque | **7%** | [3, 12] | 100 | 292 | underfilled×59, spilled×33, overfilled×1 |
| pour | glass | **10%** | [5, 16] | 100 | 289 | spilled×50, underfilled×39, overfilled×1 |
| insert | opaque | **52%** | [42, 62] | 100 | 183 | xy×46, dropped×2 |
| insert | glass | **50%** | [40, 60] | 100 | 187 | xy×48, dropped×2 |

| Task | opaque − glass (pts) | 95% CI | paired n |
|---|---|---|---|
| grasp | +5 | [-9, +19] | 100 |
| pour | -3 | [-10, +4] | 100 |
| insert | +2 | [-12, +16] | 100 |

Per-observation timing: encoder 0 ms, policy 250 ms; 28.9 observations/episode; 28.4 s/episode wall.

**Gate (≥60% every cell, |gap| ≤ 15 pts): FAIL**

