#!/bin/zsh
# After the four DART renders (already running) finish: build DART caches, then launch v4 training.
set -u
cd /Users/yogyamehrotra/Desktop/ClearVLA
source ~/miniconda3/etc/profile.d/conda.sh && conda activate clearvla
log() { echo "$(date '+%H:%M:%S') $*" >> results/week3/orchestrate.log; }
waitfor() { until grep -qE "$1" "$2" 2>/dev/null; do sleep 60; done; }
log "orchestrate_v4: waiting for 4 DART renders"
for f in front_opaque front_glass wrist_opaque wrist_glass; do waitfor "all rendered" results/week3/render_dart_$f.log; done
log "DART renders done; caching"
python data/cache_features.py --raw datasets/raw_dart --feat datasets/features_dart --source rgb_blender > results/week3/cache_dart_front.log 2>&1
python data/cache_features.py --raw datasets/raw_dart --feat datasets/features_dart --source rgb_blender_wrist --materials opaque,glass > results/week3/cache_dart_wrist.log 2>&1
cp datasets/features/text_tokens.fp16.npy datasets/features/text_index.parquet datasets/features_dart/ 2>/dev/null
log "DART caches done: $(tail -1 results/week3/cache_dart_front.log) | $(tail -1 results/week3/cache_dart_wrist.log)"
rm -rf checkpoints/v4
nohup python train.py --config configs/model_a_v4.yaml --run v4 > results/week3/train_v4.log 2>&1 &
log "v4 training launched"
