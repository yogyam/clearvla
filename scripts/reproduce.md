# Reproducing every number and figure

Environment: `clearvla` conda env (Python 3.11; `requirements.txt`, exact versions in `requirements-lock.txt`). The Blender bridge needs `bpy==5.0.1` (macOS arm64 wheel); MuJoCo 3.13. A Mac with an M-series GPU was used throughout; CUDA works for training/eval but the renderer path assumes Cycles/Metal or CPU.

Large artifacts are not in git: the dataset is on Hugging Face (`yogyamehrotra/clearvla-sim`, LeRobot v3 format) and the Model A v4 checkpoint at `yogyamehrotra/clearvla-model-a-v4` (put `best.pt` + `norm.json` under `checkpoints/v4/`). Every JSONL episode log used in the paper is committed under `results/`, so the analysis and figure commands below run without any simulation.

| Artifact | Command |
|---|---|
| Scripted-expert validity (Week 1) | `python scripts/verify_experts.py --trials 100` |
| Demos, both materials (Week 2) | `python data/record.py --per-cell 300 --stride 4 --workers 6`; recovery demos `python data/record.py --per-cell 150 --seed-start 400 --exec-noise 0.006 --out datasets/raw_dart` |
| Renders + SigLIP cache | `python data/render_demos.py --material {opaque,glass} --camera {front,wrist}`; `python data/cache_features.py --source rgb_blender{,_wrist}` |
| Model A v4 training (9 h M5 Pro) | `python train.py --config configs/model_a_v4.yaml --run v4` |
| Validity gate (Week 3) | `python -m eval.run_matrix --ckpt checkpoints/v4 --config full --material {opaque,glass} --seeds 1000-1099 --out results/week3/v4`; `python analysis/gate.py --out results/week3/v4` |
| SmolVLA fine-tune + gate (appendix) | `modal run --detach jobs/finetune_smolvla.py`; `scripts/gate_smolvla.sh` (env `lerobot311`) |
| Accel unit checks (Week 4) | `python scripts/accel_checks.py` |
| H1 observation set + recall | `python analysis/h1_collect.py --material {opaque,glass}`; `python analysis/h1_recall.py --material {opaque,glass}`; `python analysis/h1_recall.py --report` |
| Closed-loop matrix (Week 5, 5 h) | `scripts/week5_matrix.sh` → `python analysis/matrix.py --out results/week5` |
| Budget curves (Week 6, 4 h) | `scripts/week6_budgets.sh` → `python analysis/matrix.py --out results/week6` |
| Latency, M5 Pro | `python analysis/latency.py --n 300` |
| Latency, L4 (Modal, ≈ $1) | `modal volume put clearvla-data checkpoints/v4/best.pt /models/v4/best.pt`; `modal run --detach jobs/latency_trt.py`; `modal run --detach jobs/latency_trt.py --modes compile` |
| Figures 1–5 | `python analysis/figs.py --figs 1,2,3,4,5` |

Analysis-only reproduction (no simulation, < 5 min): `python analysis/h1_recall.py --report && python analysis/matrix.py --out results/week5 && python analysis/matrix.py --out results/week6 && python analysis/figs.py --figs 2,3,5`.
