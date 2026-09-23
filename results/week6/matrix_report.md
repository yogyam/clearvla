# Closed-loop matrix — `results/week6`

Paired seeds; success = task check held 2 s; CIs = bootstrap over seeds (10k). Retained success = succ(config) / succ(full) per task × material; diff-in-diff = retained(opaque) − retained(glass), paired over seeds.

## Success per cell (%, [95 % CI], n)

| Config | grasp O | grasp G | pour O | pour G | insert O | insert G |
|---|---|---|---|---|---|---|
| full | 84 [74, 94] n=50 | 82 [70, 92] n=50 | 42 [28, 56] n=50 | 28 [16, 40] n=50 | 92 [84, 98] n=50 | 100 [100, 100] n=50 |
| cache50 | 0 [0, 0] n=50 | 2 [0, 6] n=50 | 0 [0, 0] n=50 | 0 [0, 0] n=50 | 48 [34, 62] n=50 | 66 [54, 78] n=50 |
| prune12 | 56 [42, 70] n=50 | 82 [70, 92] n=50 | 40 [26, 54] n=50 | 50 [36, 64] n=50 | 92 [84, 98] n=50 | 96 [90, 100] n=50 |
| prune50 | 84 [74, 94] n=50 | 78 [66, 88] n=50 | 28 [16, 40] n=50 | 20 [10, 32] n=50 | 98 [94, 100] n=50 | 100 [100, 100] n=50 |

## Retained success vs Full and material diff-in-diff (pts)

| Config | Task | Retained opaque | Retained glass | DiD (opaque − glass) | 95 % CI |
|---|---|---|---|---|---|
| cache50 | grasp | 0 % | 2 % | -2 | [-8, +0] |
| cache50 | pour | 0 % | 0 % | +0 | [+0, +0] |
| cache50 | insert | 52 % | 66 % | -14 | [-33, +5] |
| prune12 | grasp | 67 % | 100 % | -33 | [-63, -5] |
| prune12 | pour | 95 % | 179 % | -83 | [-213, -1] |
| prune12 | insert | 100 % | 96 % | +4 | [-7, +17] |
| prune50 | grasp | 100 % | 95 % | +5 | [-19, +30] |
| prune50 | pour | 67 % | 71 % | -5 | [-74, +49] |
| prune50 | insert | 107 % | 100 % | +7 | [-2, +17] |

## Absolute success change vs Full (pts) and material diff-in-diff, paired over seeds

| Config | Task | Δ opaque | Δ glass | DiD (Δopaque − Δglass) | 95 % CI |
|---|---|---|---|---|---|
| cache50 | grasp | -84 | -80 | -4 | [-18, +10] |
| cache50 | pour | -42 | -28 | -14 | [-32, +4] |
| cache50 | insert | -44 | -34 | -10 | [-28, +10] |
| prune12 | grasp | -28 | +0 | -28 | [-52, -4] |
| prune12 | pour | -2 | +22 | -24 | [-48, +0] |
| prune12 | insert | +0 | -4 | +4 | [-6, +16] |
| prune50 | grasp | +0 | -4 | +4 | [-16, +24] |
| prune50 | pour | -14 | -8 | -6 | [-28, +16] |
| prune50 | insert | +6 | +0 | +6 | [-2, +16] |

## Failure modes, GT-patch recall, compute

| Config | Material | Failures (top 3) | Mean GT recall | GFLOPs | Policy ms/obs |
|---|---|---|---|---|---|
| full | opaque | spilled×24, tube_knocked_over×6, xy_off_#mm×3 | — | — | 37 |
| full | glass | spilled×30, xy_off_#mm×5, tube_knocked_over×4 | — | — | 36 |
| cache50 | opaque | spilled×50, tube_knocked_over×44, xy_off_#mm×16 | 72 % | 20.3 | 38 |
| cache50 | glass | spilled×47, tube_knocked_over×45, xy_off_#mm×13 | 68 % | 20.3 | 37 |
| prune12 | opaque | spilled×27, tube_knocked_over×13, xy_off_#mm×11 | 15 % | 15.3 | 40 |
| prune12 | glass | spilled×20, tube_knocked_over×6, xy_off_#mm×4 | 8 % | 15.3 | 39 |
| prune50 | opaque | spilled×31, tube_knocked_over×5, xy_off_#mm×4 | 57 % | 19.0 | 41 |
| prune50 | glass | spilled×33, xy_off_#mm×7, tube_knocked_over×4 | 56 % | 19.0 | 40 |

## Pre-registered verdicts

- **H2 for cache50** (≥ 15 pts lower retained success on glass, CI excludes 0, on ≥ 2 of 3 tasks): tasks meeting it = none → not supported
- **H2 for prune12** (≥ 15 pts lower retained success on glass, CI excludes 0, on ≥ 2 of 3 tasks): tasks meeting it = none → not supported
- **H2 for prune50** (≥ 15 pts lower retained success on glass, CI excludes 0, on ≥ 2 of 3 tasks): tasks meeting it = none → not supported
- **H3 (cache50 hurts which task most, mean retained-success loss):** grasp 99 pts, pour 100 pts, insert 41 pts → worst = pour (pre-registered: Cache → pour, Prune → grasp)
- **H3 (prune12 hurts which task most, mean retained-success loss):** grasp 17 pts, pour -37 pts, insert 2 pts → worst = grasp (pre-registered: Cache → pour, Prune → grasp)
- **H3 (prune50 hurts which task most, mean retained-success loss):** grasp 2 pts, pour 31 pts, insert -3 pts → worst = pour (pre-registered: Cache → pour, Prune → grasp)
