#!/bin/zsh
# 10-seed closed-loop smoke of every accel config (grasp, opaque): hooks run end-to-end, latency + FLOPs logged.
cd /Users/yogyamehrotra/Desktop/ClearVLA; source ~/miniconda3/etc/profile.d/conda.sh && conda activate clearvla
for c in full prune50 prune25 prune12 cache50 cache25 cache12 quant4 protect_prune25 protect_cache25; do
  python -m eval.run_matrix --ckpt checkpoints/v4 --config $c --material opaque --tasks grasp --seeds 1000-1009 --out results/week4/smoke --strips 1 > results/week4/smoke/$c.log 2>&1
  echo "$c: $(tail -1 results/week4/smoke/$c.log)"
done
echo "smoke done"
