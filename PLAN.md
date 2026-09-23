# ClearVLA — pre-registered design

Fixed 2026-09-15, before any model was trained. This file is the pre-registration the paper refers to; the study's outcome is in `docs/report.md` and the paper in `paper/`.

## Question
Do VLA acceleration methods (visual-token pruning, cross-observation token caching, INT4 weight quantization) lose disproportionately more task success on transparent laboratory objects than on opaque ones?

## Hypotheses (frozen)

| ID | Claim | Metric | Supported if | Falsified if |
|----|-------|--------|--------------|--------------|
| H1 | Importance scores under-rank transparent regions | Critical-patch recall @25% budget (a patch counts if ≥25% of its pixels overlap the GT object mask) | recall(opaque) − recall(glass) ≥ 10 pts, bootstrap 95% CI excludes 0 | gap < 10 pts or CI includes 0 |
| H2 (main) | Speed-ups cost more success on glass | Retained success = succ(b)/succ(full); diff-in-diff across materials | @25% budget, ≥15 pts lower retained success on glass, CI excludes 0, on ≥2 of 3 tasks | statistically indistinguishable |
| H3 | Each method fails where its assumption fails | method × task interaction | Cache hurts Pour most; Prune hurts Grasp most | both degrade the same way |
| H4 | Quantization is material-neutral | glass − opaque diff in retained success | < 5 pts | ≥ 5 pts |
| H5 | Protecting instruction-referenced patches recovers the loss | recovery fraction, added compute | ≥50% of the transparency-specific drop recovered, ≤10% extra compute | — |

**Decision gate.** H1 is cheap and offline and runs first. H1 supported → full closed-loop matrix (8 configs × 3 tasks × 2 materials × 100 episodes). H1 falsified → reduced matrix (25% budget only, 50 paired episodes per cell); if success is also unaffected by material → negative-result write-up.

## Design
- **Scenes:** MuJoCo (Menagerie Franka Panda) scenes built per seed; Blender Cycles rendering with true refraction for glass; the same seed gives identical geometry, expert actions and masks under `opaque` and `glass`, so every comparison is paired. Randomisation per seed: object ±5 cm, receptacle ±3 cm, light ±20%, camera ±1 cm / ±2°.
- **Tasks:** grasp-and-rack, pour-to-line (scripted liquid; spill = pouring outside the receiver), insert (starts in-hand). Horizon 300 control steps at 10 Hz; success must hold 2 s. Material-neutral instructions, 8 paraphrases per task.
- **Data:** scripted experts (≥95% success required on both materials); paired demonstrations; every 4th step rendered; recovery (DART-style) demonstrations allowed if the validity gate fails.
- **Instrument:** a small from-scratch VLA (frozen SigLIP encoders → prefix transformer → flow-matching action expert) with attention maps and K/V exposed. **Validity gate:** ≥60% closed-loop success on every task for both materials, |glass − opaque| ≤ 15 pts, 100 unseen paired seeds per cell, before any acceleration experiment.
- **Methods (identical weights):** Prune (FastV-style, attention-ranked after layer 2; budgets 50 / 25 / 12.5%), Cache (VLA-Cache-style, pixel-change criterion; same budgets), Quant (fake INT4, per-group 64, prefix + expert), Protect (Prune/Cache with GT-mask patches always kept; upper bound). Budget = fraction of visual tokens with full prefix compute after layer 2; FLOPs and latency logged.
- **Evaluation rules:** paired seeds across every config and material; synchronous evaluation (simulator pauses during inference); 95% bootstrap CIs over paired seeds for every difference; per-episode logs with keep sets.
- **Latency:** measured on an Apple M5 Pro (MPS) and an NVIDIA L4 (eager, compiled, TensorRT); FLOPs reduction reported next to measured latency.

## Schedule at a glance

| Week | Work | Exit criterion |
|------|------|----------------|
| 0 | environment, feasibility checks | all pass or fallback chosen |
| 1 | tasks, materials, success checks, experts | expert ≥95% both materials |
| 2 | demonstrations, renders, masks, feature cache | dataset loads; spot check |
| 3 | instrument + training | validity gate |
| 4 | prune / cache / quant / protect + H1 | H1 decision gate |
| 5–6 | closed-loop matrix (and budget curves) | all cells logged |
| 7 | mechanism analysis, latency | figures drafted |
| 8 | write-up + release | report, public repo, dataset |

## Pre-stated limitations
Simulated glass is easier than real glass; the liquid is scripted; H5 uses simulator ground-truth masks (upper bound); evaluation is synchronous; a sticky-gripper weld removes grasp-slip as a failure mode.
