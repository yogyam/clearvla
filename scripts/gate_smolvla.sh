#!/bin/zsh
# Wait for the Modal fine-tune to exit, download the final checkpoint, run the paired 600-episode gate on the Mac.
set -u
cd /Users/yogyamehrotra/Desktop/ClearVLA
source ~/miniconda3/etc/profile.d/conda.sh
RUN=${1:-smolvla_v1}; STEP=${2:-020000}; OUT=results/smolvla/v1   # `last` is a symlink the volume CLI cannot fetch; name the step dir
log() { echo "$(date '+%H:%M:%S') $*" >> $OUT/orchestrate.log; }
conda activate clearvla
if [ -f checkpoints/smolvla_v1/final/pretrained_model/model.safetensors ]; then log "checkpoint already local; skipping wait/download"; SKIP=1; else SKIP=0; fi
log "waiting for Modal run $RUN to exit"
while [ $SKIP = 0 ]; do
  st=$(modal run jobs/finetune_smolvla.py --status-only --run $RUN 2>&1 | grep -E "^exit" | tail -1)
  [ -n "$st" ] && break; sleep 300
done
if [ $SKIP = 0 ]; then
log "modal: $st"
if ! echo "$st" | grep -q "^exit 0"; then log "training failed; stopping"; exit 1; fi
rm -rf checkpoints/smolvla_v1/final && mkdir -p checkpoints/smolvla_v1/final
modal volume get clearvla-data runs/$RUN/lerobot/checkpoints/$STEP/pretrained_model checkpoints/smolvla_v1/final/ --force >> $OUT/orchestrate.log 2>&1
ls checkpoints/smolvla_v1/final/pretrained_model/model.safetensors || { log "download failed"; exit 1; }
log "checkpoint downloaded: $(du -sh checkpoints/smolvla_v1/final | cut -f1)"
fi
conda activate lerobot311
for m in opaque glass; do
  nohup python -m eval.run_matrix --policy smolvla --ckpt checkpoints/smolvla_v1/final --config full --material $m --seeds 1000-1099 --out $OUT --strips 3 > $OUT/gate_$m.log 2>&1 &
done
log "gate launched (opaque + glass in parallel)"
wait
log "gate episodes done: $(tail -1 $OUT/gate_opaque.log) | $(tail -1 $OUT/gate_glass.log)"
python analysis/gate.py --out $OUT --config full > $OUT/gate_report_full.md 2>> $OUT/orchestrate.log
log "gate report: $OUT/gate_report_full.md"
log "gate done"
