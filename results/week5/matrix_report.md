# Closed-loop matrix — `results/week5`

Paired seeds; success = task check held 2 s; CIs = bootstrap over seeds (10k). Retained success = succ(config) / succ(full) per task × material; diff-in-diff = retained(opaque) − retained(glass), paired over seeds.

## Success per cell (%, [95 % CI], n)

| Config | grasp O | grasp G | pour O | pour G | insert O | insert G |
|---|---|---|---|---|---|---|
| full | 84 [74, 94] n=50 | 82 [70, 92] n=50 | 42 [28, 56] n=50 | 28 [16, 40] n=50 | 92 [84, 98] n=50 | 100 [100, 100] n=50 |
| cache25 | 0 [0, 0] n=50 | 0 [0, 0] n=50 | 0 [0, 0] n=50 | 0 [0, 0] n=50 | 22 [12, 34] n=50 | 24 [12, 36] n=50 |
| cachef25 | 0 [0, 0] n=50 | 0 [0, 0] n=50 | 0 [0, 0] n=50 | 0 [0, 0] n=50 | 12 [4, 22] n=50 | 20 [10, 32] n=50 |
| protect_prune25 | 80 [68, 90] n=50 | 76 [64, 88] n=50 | 46 [32, 60] n=50 | 26 [14, 38] n=50 | 90 [82, 98] n=50 | 96 [90, 100] n=50 |
| prune25 | 76 [64, 88] n=50 | 78 [66, 90] n=50 | 34 [22, 48] n=50 | 32 [20, 44] n=50 | 94 [86, 100] n=50 | 100 [100, 100] n=50 |
| quant4 | 76 [64, 88] n=50 | 74 [62, 86] n=50 | 38 [24, 52] n=50 | 22 [12, 34] n=50 | 94 [86, 100] n=50 | 100 [100, 100] n=50 |

## Retained success vs Full and material diff-in-diff (pts)

| Config | Task | Retained opaque | Retained glass | DiD (opaque − glass) | 95 % CI |
|---|---|---|---|---|---|
| cache25 | grasp | 0 % | 0 % | +0 | [+0, +0] |
| cache25 | pour | 0 % | 0 % | +0 | [+0, +0] |
| cache25 | insert | 24 % | 24 % | -0 | [-16, +16] |
| cachef25 | grasp | 0 % | 0 % | +0 | [+0, +0] |
| cachef25 | pour | 0 % | 0 % | +0 | [+0, +0] |
| cachef25 | insert | 13 % | 20 % | -7 | [-21, +7] |
| protect_prune25 | grasp | 95 % | 93 % | +3 | [-17, +24] |
| protect_prune25 | pour | 110 % | 93 % | +17 | [-85, +98] |
| protect_prune25 | insert | 98 % | 96 % | +2 | [-11, +17] |
| prune25 | grasp | 90 % | 95 % | -5 | [-30, +21] |
| prune25 | pour | 81 % | 114 % | -33 | [-133, +26] |
| prune25 | insert | 102 % | 100 % | +2 | [-7, +12] |
| quant4 | grasp | 90 % | 90 % | +0 | [-21, +21] |
| quant4 | pour | 90 % | 79 % | +12 | [-75, +90] |
| quant4 | insert | 102 % | 100 % | +2 | [-8, +14] |

## Absolute success change vs Full (pts) and material diff-in-diff, paired over seeds

| Config | Task | Δ opaque | Δ glass | DiD (Δopaque − Δglass) | 95 % CI |
|---|---|---|---|---|---|
| cache25 | grasp | -84 | -82 | -2 | [-16, +10] |
| cache25 | pour | -42 | -28 | -14 | [-32, +4] |
| cache25 | insert | -70 | -76 | +6 | [-10, +24] |
| cachef25 | grasp | -84 | -82 | -2 | [-14, +10] |
| cachef25 | pour | -42 | -28 | -14 | [-32, +4] |
| cachef25 | insert | -80 | -80 | +0 | [-16, +16] |
| protect_prune25 | grasp | -4 | -6 | +2 | [-16, +20] |
| protect_prune25 | pour | +4 | -2 | +6 | [-20, +34] |
| protect_prune25 | insert | -2 | -4 | +2 | [-10, +16] |
| prune25 | grasp | -8 | -4 | -4 | [-26, +18] |
| prune25 | pour | -8 | +4 | -12 | [-34, +8] |
| prune25 | insert | +2 | +0 | +2 | [-6, +10] |
| quant4 | grasp | -8 | -8 | +0 | [-18, +18] |
| quant4 | pour | -4 | -6 | +2 | [-26, +28] |
| quant4 | insert | +2 | +0 | +2 | [-8, +12] |

## Failure modes, GT-patch recall, compute

| Config | Material | Failures (top 3) | Mean GT recall | GFLOPs | Policy ms/obs |
|---|---|---|---|---|---|
| full | opaque | spilled×24, tube_knocked_over×6, xy_off_#mm×3 | — | — | 37 |
| full | glass | spilled×30, xy_off_#mm×5, tube_knocked_over×4 | — | — | 36 |
| cache25 | opaque | spilled×49, tube_knocked_over×39, xy_off_#mm×25 | 62 % | 18.2 | 38 |
| cache25 | glass | spilled×47, tube_knocked_over×42, xy_off_#mm×26 | 60 % | 18.2 | 38 |
| cachef25 | opaque | spilled×47, tube_knocked_over×38, xy_off_#mm×32 | 37 % | 18.2 | 38 |
| cachef25 | glass | spilled×43, tube_knocked_over×32, xy_off_#mm×29 | 45 % | 18.2 | 36 |
| protect_prune25 | opaque | spilled×26, xy_off_#mm×9, tube_knocked_over×5 | 100 % | 16.5 | 40 |
| protect_prune25 | glass | spilled×31, xy_off_#mm×7, tube_knocked_over×5 | 100 % | 16.5 | 39 |
| prune25 | opaque | spilled×30, xy_off_#mm×8, tube_knocked_over×3 | 30 % | 16.5 | 41 |
| prune25 | glass | spilled×25, tube_knocked_over×6, underfilled_#_of_#×6 | 23 % | 16.5 | 40 |
| quant4 | opaque | spilled×21, xy_off_#mm×8, underfilled_#_of_#×8 | 100 % | 24.4 | 37 |
| quant4 | glass | spilled×25, underfilled_#_of_#×12, xy_off_#mm×7 | 100 % | 24.4 | 36 |

## Pre-registered verdicts

- **H2 for cache25** (≥ 15 pts lower retained success on glass, CI excludes 0, on ≥ 2 of 3 tasks): tasks meeting it = none → not supported
- **H2 for cachef25** (≥ 15 pts lower retained success on glass, CI excludes 0, on ≥ 2 of 3 tasks): tasks meeting it = none → not supported
- **H2 for protect_prune25** (≥ 15 pts lower retained success on glass, CI excludes 0, on ≥ 2 of 3 tasks): tasks meeting it = none → not supported
- **H2 for prune25** (≥ 15 pts lower retained success on glass, CI excludes 0, on ≥ 2 of 3 tasks): tasks meeting it = none → not supported
- **H4 (Quant material-neutral, |DiD| < 5 pts):** grasp: +0, pour: +12, insert: +2 → not supported (still interesting)
- **H3 (cache25 hurts which task most, mean retained-success loss):** grasp 100 pts, pour 100 pts, insert 76 pts → worst = grasp (pre-registered: Cache → pour, Prune → grasp)
- **H3 (cachef25 hurts which task most, mean retained-success loss):** grasp 100 pts, pour 100 pts, insert 83 pts → worst = grasp (pre-registered: Cache → pour, Prune → grasp)
- **H3 (prune25 hurts which task most, mean retained-success loss):** grasp 7 pts, pour 2 pts, insert -1 pts → worst = grasp (pre-registered: Cache → pour, Prune → grasp)
