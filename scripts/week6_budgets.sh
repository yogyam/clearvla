#!/bin/zsh
# Week 6 budget curve: 3 more configs x 3 tasks x 2 materials x 50 paired seeds. Two workers (one per material).
# Resumable: the evaluator skips (task, seed) pairs already in the log.
cd /Users/yogyamehrotra/Desktop/ClearVLA; source ~/miniconda3/etc/profile.d/conda.sh && conda activate clearvla
OUT=results/week6; mkdir -p $OUT
log() { echo "$(date '+%H:%M:%S') $*" >> $OUT/orchestrate.log; }
CONFIGS=(prune50 prune12 cache50)
worker() {
  m=$1
  for c in $CONFIGS; do
    python -m eval.run_matrix --ckpt checkpoints/v4 --config $c --material $m --seeds 1000-1049 --out $OUT --strips 3 > $OUT/run_${c}_${m}.log 2>&1
    log "$c $m: $(tail -1 $OUT/run_${c}_${m}.log | cut -c1-60)"
  done
}
log "matrix launched: ${CONFIGS[*]}"
worker opaque & worker glass & wait
log "episodes done"
python analysis/matrix.py --out $OUT > $OUT/matrix_report.md 2>> $OUT/orchestrate.log
log "matrix done"
