# Validity gate — config `full`

| Task | Material | Success | 95% CI | n | Mean steps | Failures |
|---|---|---|---|---|---|---|
| grasp | opaque | **80%** | [72, 88] | 100 | 150 | xy×13, tube×7 |
| grasp | glass | **82%** | [74, 89] | 100 | 146 | xy×10, tube×8 |
| pour | opaque | **10%** | [5, 16] | 100 | 288 | spilled×56, underfilled×32, overfilled×2 |
| pour | glass | **22%** | [14, 30] | 100 | 272 | spilled×51, underfilled×18, overfilled×9 |
| insert | opaque | **98%** | [95, 100] | 100 | 73 | dropped×1, xy×1 |
| insert | glass | **96%** | [92, 99] | 100 | 78 | xy×3, dropped×1 |

| Task | opaque − glass (pts) | 95% CI | paired n |
|---|---|---|---|
| grasp | -2 | [-12, +8] | 100 |
| pour | -12 | [-22, -2] | 100 |
| insert | +2 | [-3, +7] | 100 |

Per-observation timing: encoder 26 ms, policy 38 ms; 21.4 observations/episode; 16.8 s/episode wall.

**Gate (≥60% every cell, |gap| ≤ 15 pts): FAIL**
