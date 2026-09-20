# Validity gate — config `full`

| Task | Material | Success | 95% CI | n | Mean steps | Failures |
|---|---|---|---|---|---|---|
| grasp | opaque | **77%** | [69, 85] | 100 | 157 | tube×15, xy×8 |
| grasp | glass | **80%** | [72, 87] | 100 | 150 | tube×13, xy×7 |
| pour | opaque | **43%** | [33, 53] | 100 | 250 | spilled×48, underfilled×5, overfilled×4 |
| pour | glass | **37%** | [28, 47] | 100 | 254 | spilled×53, underfilled×6, overfilled×4 |
| insert | opaque | **98%** | [95, 100] | 100 | 77 | dropped×1, xy×1 |
| insert | glass | **99%** | [97, 100] | 100 | 74 | dropped×1 |

| Task | opaque − glass (pts) | 95% CI | paired n |
|---|---|---|---|
| grasp | -3 | [-14, +8] | 100 |
| pour | +6 | [-8, +20] | 100 |
| insert | -1 | [-4, +2] | 100 |

Per-observation timing: encoder 25 ms, policy 40 ms; 20.4 observations/episode; 17.5 s/episode wall.

**Gate (≥60% every cell, |gap| ≤ 15 pts): FAIL**
