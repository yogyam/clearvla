# Scripted expert verification (100 seeds per cell, 21 s wall, 6 workers)

| Task | Material | Success | Mean steps | Failures |
|---|---|---|---|---|
| grasp | opaque | **100%** (100/100) | 104 | — |
| grasp | glass | **100%** (100/100) | 104 | — |
| pour | opaque | **99%** (99/100) | 236 | spilled×1 |
| pour | glass | **99%** (99/100) | 236 | spilled×1 |
| insert | opaque | **100%** (100/100) | 68 | — |
| insert | glass | **100%** (100/100) | 68 | — |

Paired-seed mismatches across materials: 0 []

**Gate (≥95% every cell): PASS**

Failed episodes:
- pour/opaque/s11: spilled
- pour/glass/s11: spilled
