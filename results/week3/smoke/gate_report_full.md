# Validity gate — config `full`

| Task | Material | Success | 95% CI | n | Mean steps | Failures |
|---|---|---|---|---|---|---|
| grasp | opaque | **35%** | [15, 55] | 20 | 233 | xy×9, tube×4 |
| grasp | glass | **30%** | [10, 50] | 20 | 247 | xy×11, tube×3 |
| pour | opaque | **0%** | [0, 0] | 20 | 300 | spilled×13, underfilled×6, overfilled×1 |
| pour | glass | **0%** | [0, 0] | 20 | 300 | underfilled×13, spilled×7 |
| insert | opaque | **45%** | [25, 65] | 20 | 197 | xy×6, dropped×5 |
| insert | glass | **60%** | [40, 80] | 20 | 163 | xy×6, dropped×2 |

| Task | opaque − glass (pts) | 95% CI | paired n |
|---|---|---|---|
| grasp | +5 | [-25, +35] | 20 |
| pour | +0 | [+0, +0] | 20 |
| insert | -15 | [-45, +15] | 20 |

Per-observation timing: encoder 5 ms, policy 38 ms; 30.5 observations/episode; 12.3 s/episode wall.

**Gate (≥60% every cell, |gap| ≤ 15 pts): FAIL**
