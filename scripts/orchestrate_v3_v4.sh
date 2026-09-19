#!/bin/zsh
# Runs after v3 training: gate v3 || DART renders (2 at a time) -> DART caches -> v4 training. Idempotent-ish (resume flags).
set -u
cd /Users/yogyamehrotra/Desktop/ClearVLA
source ~/miniconda3/etc/profile.d/conda.sh && conda activate clearvla
log() { echo "$(date '+%H:%M:%S') $*" >> results/week3/orchestrate.log; }
waitfor() { until grep -qE "$1" "$2" 2>/dev/null; do sleep 60; done; }
log "waiting for v3 training to finish"
waitfor "^done|non-finite|Traceback" results/week3/train_v3.log
log "v3 training finished: $(grep -E '^done|non-finite|Traceback' results/week3/train_v3.log | head -1)"
# 1) gate v3 (2 procs) + DART front renders (2 procs)
rm -f results/week3/v3_gate_*.log
for m in opaque glass; do nohup python eval/run_matrix.py --ckpt checkpoints/v3 --config full --material $m --seeds 1000-1099 --out results/week3/v3 > results/week3/v3_gate_$m.log 2>&1 & done
for m in opaque glass; do nohup python data/render_demos.py --raw datasets/raw_dart --material $m --camera front --resume > results/week3/render_dart_front_$m.log 2>&1 & done
log "launched gate v3 + DART front renders"
waitfor "all rendered" results/week3/render_dart_front_opaque.log; waitfor "all rendered" results/week3/render_dart_front_glass.log
log "DART front renders done; launching wrist renders"
for m in opaque glass; do nohup python data/render_demos.py --raw datasets/raw_dart --material $m --camera wrist --resume > results/week3/render_dart_wrist_$m.log 2>&1 & done
waitfor "^done" results/week3/v3_gate_opaque.log; waitfor "^done" results/week3/v3_gate_glass.log
python analysis/gate.py --out results/week3/v3 --config full > results/week3/v3/gate_stdout.txt 2>&1
log "gate v3 done: $(grep -E 'Gate' results/week3/v3/gate_report_full.md)"
waitfor "all rendered" results/week3/render_dart_wrist_opaque.log; waitfor "all rendered" results/week3/render_dart_wrist_glass.log
log "DART wrist renders done; caching"
python data/cache_features.py --raw datasets/raw_dart --feat datasets/features_dart --source rgb_blender > results/week3/cache_dart_front.log 2>&1
python data/cache_features.py --raw datasets/raw_dart --feat datasets/features_dart --source rgb_blender_wrist --materials opaque,glass > results/week3/cache_dart_wrist.log 2>&1
cp datasets/features/text_tokens.fp16.npy datasets/features/text_index.parquet datasets/features_dart/ 2>/dev/null
log "DART caches done: $(tail -1 results/week3/cache_dart_front.log) | $(tail -1 results/week3/cache_dart_wrist.log)"
rm -rf checkpoints/v4
nohup python train.py --config configs/model_a_v4.yaml --run v4 > results/week3/train_v4.log 2>&1 &
log "v4 training launched"
