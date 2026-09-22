# H1 — critical-patch recall of the keep sets (offline, expert rollouts, seeds 1000–1049)

Keep set = visual tokens with full prefix compute after layer 2 (both views, 392 tokens). Critical patch = ≥ 25 % of its pixels on the instruction-referenced object (simulator GT). Recall = |keep ∩ crit| / |crit|. Episode means, then mean over episodes; gap = opaque − glass with a bootstrap over paired seeds (10k resamples). Cache excludes the first observation of each episode (always full).


## Recall, view = all

| Method | Task | Opaque | Glass | Gap (pts) | 95 % CI | n seeds |
|---|---|---|---|---|---|---|
| prune50 | grasp | 57.8 | 57.3 | +0.6 | [-1.1, +2.3] | 50 |
| prune50 | pour | 64.0 | 69.7 | -5.7 | [-7.6, -3.8] | 50 |
| prune50 | insert | 45.5 | 47.3 | -1.8 | [-3.1, -0.6] | 50 |
| prune50 | pooled | 55.8 | 58.1 | -2.3 | [-3.4, -1.3] | 150 |
| prune25 | grasp | 33.2 | 25.2 | +8.0 | [+6.2, +9.8] | 50 |
| prune25 | pour | 26.6 | 28.8 | -2.2 | [-4.6, +0.2] | 50 |
| prune25 | insert | 13.7 | 10.4 | +3.3 | [+2.5, +4.1] | 50 |
| prune25 | pooled | 24.5 | 21.5 | +3.0 | [+1.8, +4.2] | 150 |
| prune12 | grasp | 20.2 | 10.5 | +9.7 | [+8.6, +10.7] | 50 |
| prune12 | pour | 11.5 | 9.6 | +2.0 | [+0.3, +3.6] | 50 |
| prune12 | insert | 7.5 | 4.6 | +2.9 | [+2.4, +3.4] | 50 |
| prune12 | pooled | 13.1 | 8.2 | +4.8 | [+3.9, +5.7] | 150 |
| cache50 | grasp | 59.3 | 65.5 | -6.1 | [-7.0, -5.3] | 50 |
| cache50 | pour | 99.3 | 100.0 | -0.7 | [-1.0, -0.5] | 50 |
| cache50 | insert | 62.4 | 63.9 | -1.6 | [-2.1, -1.0] | 50 |
| cache50 | pooled | 73.6 | 76.5 | -2.8 | [-3.3, -2.3] | 150 |
| cache25 | grasp | 51.6 | 58.5 | -6.8 | [-7.4, -6.3] | 50 |
| cache25 | pour | 93.2 | 97.2 | -4.0 | [-4.7, -3.3] | 50 |
| cache25 | insert | 55.4 | 56.4 | -1.0 | [-1.6, -0.4] | 50 |
| cache25 | pooled | 66.8 | 70.7 | -3.9 | [-4.5, -3.4] | 150 |
| cache12 | grasp | 38.3 | 40.6 | -2.3 | [-2.9, -1.7] | 50 |
| cache12 | pour | 69.9 | 88.0 | -18.1 | [-20.0, -16.4] | 50 |
| cache12 | insert | 44.5 | 42.7 | +1.8 | [+1.2, +2.4] | 50 |
| cache12 | pooled | 50.9 | 57.1 | -6.2 | [-7.8, -4.7] | 150 |

## Recall, view = front

| Method | Task | Opaque | Glass | Gap (pts) | 95 % CI | n seeds |
|---|---|---|---|---|---|---|
| prune50 | grasp | 34.9 | 43.5 | -8.7 | [-11.3, -6.0] | 50 |
| prune50 | insert | 34.8 | 35.3 | -0.4 | [-2.2, +1.3] | 50 |
| prune50 | pooled | 34.8 | 39.4 | -4.5 | [-6.4, -2.7] | 100 |
| prune25 | grasp | 22.1 | 18.6 | +3.5 | [+1.1, +6.0] | 50 |
| prune25 | insert | 9.9 | 7.8 | +2.1 | [+1.0, +3.1] | 50 |
| prune25 | pooled | 16.0 | 13.2 | +2.8 | [+1.5, +4.2] | 100 |
| prune12 | grasp | 14.6 | 8.1 | +6.4 | [+4.8, +8.0] | 50 |
| prune12 | insert | 4.3 | 5.1 | -0.9 | [-1.7, -0.1] | 50 |
| prune12 | pooled | 9.4 | 6.6 | +2.8 | [+1.7, +4.0] | 100 |
| cache50 | grasp | 68.1 | 62.8 | +5.3 | [+4.2, +6.4] | 50 |
| cache50 | insert | 54.5 | 51.3 | +3.2 | [+2.4, +4.0] | 50 |
| cache50 | pooled | 61.3 | 57.0 | +4.2 | [+3.5, +5.0] | 100 |
| cache25 | grasp | 57.4 | 56.1 | +1.3 | [+0.3, +2.4] | 50 |
| cache25 | insert | 46.3 | 44.1 | +2.1 | [+1.2, +3.1] | 50 |
| cache25 | pooled | 51.8 | 50.1 | +1.7 | [+1.0, +2.4] | 100 |
| cache12 | grasp | 43.2 | 43.1 | +0.1 | [-1.2, +1.3] | 50 |
| cache12 | insert | 38.1 | 32.6 | +5.5 | [+4.6, +6.5] | 50 |
| cache12 | pooled | 40.7 | 37.8 | +2.8 | [+1.9, +3.8] | 100 |

## Recall, view = wrist

| Method | Task | Opaque | Glass | Gap (pts) | 95 % CI | n seeds |
|---|---|---|---|---|---|---|
| prune50 | grasp | 70.6 | 60.9 | +9.7 | [+7.7, +11.8] | 50 |
| prune50 | pour | 64.0 | 69.7 | -5.7 | [-7.6, -3.8] | 50 |
| prune50 | insert | 48.6 | 51.2 | -2.6 | [-4.3, -0.9] | 50 |
| prune50 | pooled | 61.1 | 60.6 | +0.5 | [-1.1, +2.0] | 150 |
| prune25 | grasp | 34.3 | 28.5 | +5.8 | [+3.5, +8.1] | 50 |
| prune25 | pour | 26.6 | 28.8 | -2.2 | [-4.6, +0.1] | 50 |
| prune25 | insert | 14.9 | 11.3 | +3.6 | [+2.7, +4.5] | 50 |
| prune25 | pooled | 25.3 | 22.9 | +2.4 | [+1.2, +3.7] | 150 |
| prune12 | grasp | 19.5 | 12.1 | +7.3 | [+5.5, +9.2] | 50 |
| prune12 | pour | 11.5 | 9.6 | +2.0 | [+0.3, +3.7] | 50 |
| prune12 | insert | 9.3 | 4.1 | +5.2 | [+4.5, +5.8] | 50 |
| prune12 | pooled | 13.4 | 8.6 | +4.8 | [+3.9, +5.8] | 150 |
| cache50 | grasp | 70.1 | 81.5 | -11.4 | [-12.7, -10.0] | 50 |
| cache50 | pour | 99.3 | 100.0 | -0.7 | [-1.0, -0.5] | 50 |
| cache50 | insert | 66.0 | 71.1 | -5.2 | [-5.8, -4.5] | 50 |
| cache50 | pooled | 78.4 | 84.2 | -5.8 | [-6.6, -4.9] | 150 |
| cache25 | grasp | 64.0 | 74.5 | -10.5 | [-11.1, -9.8] | 50 |
| cache25 | pour | 93.2 | 97.2 | -4.0 | [-4.7, -3.3] | 50 |
| cache25 | insert | 61.1 | 64.6 | -3.6 | [-4.2, -2.9] | 50 |
| cache25 | pooled | 72.8 | 78.8 | -6.0 | [-6.7, -5.4] | 150 |
| cache12 | grasp | 50.6 | 54.2 | -3.6 | [-4.3, -2.9] | 50 |
| cache12 | pour | 69.9 | 88.0 | -18.1 | [-20.0, -16.3] | 50 |
| cache12 | insert | 49.4 | 49.8 | -0.4 | [-1.4, +0.6] | 50 |
| cache12 | pooled | 56.6 | 64.0 | -7.4 | [-8.8, -5.9] | 150 |

## Mechanism

| Task | Material | Attention mass on critical tokens (layer 2) | Critical patches front / wrist | Pixel similarity to previous obs, front / wrist |
|---|---|---|---|---|
| grasp | opaque | 4.7 % | 3.3 / 5.5 | 0.809 / 0.898 |
| grasp | glass | 2.1 % | 3.3 / 5.5 | 0.811 / 0.897 |
| pour | opaque | 0.7 % | 0.0 / 2.0 | 0.838 / 0.828 |
| pour | glass | 0.6 % | 0.0 / 2.0 | 0.837 / 0.824 |
| insert | opaque | 4.6 % | 5.5 / 9.5 | 0.883 / 0.931 |
| insert | glass | 3.7 % | 5.5 / 9.5 | 0.886 / 0.927 |

## H1 verdict (pre-registered: recall(opaque) − recall(glass) ≥ 10 pts at the 25 % budget, CI excludes 0)

- **prune25** pooled gap +3.0 pts [+1.8, +4.2] → not supported (needs ≥ +10 with CI excluding 0)
- **cache25** pooled gap -3.9 pts [-4.5, -3.4] → not supported (needs ≥ +10 with CI excluding 0)
