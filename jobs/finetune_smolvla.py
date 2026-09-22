"""Fine-tune SmolVLA (lerobot/smolvla_base) on the ClearVLA LeRobot dataset, on a Modal H100.

  modal run jobs/finetune_smolvla.py --steps 20000            # full run (~4 h H100)
  modal run jobs/finetune_smolvla.py --steps 500 --smoke      # 10-min smoke test
  modal run jobs/finetune_smolvla.py --status                  # print the loss log from the volume

Data lives on the Modal Volume `clearvla-data` under /lerobot/clearvla_all (uploaded with `modal volume put`).
Checkpoints and the loss log are written back to the volume under /runs/<run>/.
The process exits when training ends — never leaves a container idle.
"""
import modal, os, subprocess, time, json, sys

APP = "clearvla-smolvla"
VOL = modal.Volume.from_name("clearvla-data")
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg", "git")
    .pip_install("lerobot[smolvla]==0.4.4", "h5py", "pyarrow")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "0", "TOKENIZERS_PARALLELISM": "false"})
)
app = modal.App(APP, image=image)


@app.function(gpu="H100", timeout=8 * 3600, volumes={"/vol": VOL}, secrets=[modal.Secret.from_name("huggingface")])
def train(run: str, steps: int, batch: int, dataset: str, smoke: bool):
    os.makedirs("/vol/runs", exist_ok=True)
    out = f"/vol/runs/{run}/lerobot"                    # lerobot-train insists on creating this itself
    logp = f"/vol/runs/{run}_train.log"
    ds_root = f"/vol/lerobot/{dataset}"
    assert os.path.exists(ds_root + "/meta/info.json"), f"dataset not on volume: {ds_root}"
    cmd = [
        "lerobot-train",
        f"--dataset.repo_id=clearvla/{dataset}", f"--dataset.root={ds_root}",
        "--policy.path=lerobot/smolvla_base", "--policy.push_to_hub=false",
        f"--output_dir={out}", f"--job_name={run}",
        f"--steps={steps}", f"--batch_size={batch}", "--save_freq=2000", "--log_freq=50",
        "--policy.device=cuda", "--wandb.enable=false", "--num_workers=8",
        # smolvla_base was pretrained with cameras named camera1..3; map ours onto the first two
        '--rename_map={"observation.images.front": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}',
    ]
    log = open(logp, "a"); log.write(" ".join(cmd) + "\n"); log.flush()
    t0 = time.time(); p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    last_commit = t0
    for line in p.stdout:
        log.write(line); sys.stdout.write(line)
        if time.time() - last_commit > 300: log.flush(); VOL.commit(); last_commit = time.time()
    p.wait(); log.write(f"exit {p.returncode} after {(time.time()-t0)/60:.1f} min\n"); log.close(); VOL.commit()
    return p.returncode


@app.function(volumes={"/vol": VOL})
def status(run: str):
    p = f"/vol/runs/{run}_train.log"
    if not os.path.exists(p): return "no log yet"
    lines = open(p).read().splitlines()
    keep = [l for l in lines if ("loss" in l and "step" in l) or l.startswith("exit")]
    return "\n".join(keep[-25:]) + f"\n({len(lines)} lines total)"


@app.local_entrypoint()
def main(steps: int = 20000, batch: int = 64, run: str = "smolvla_v1", dataset: str = "clearvla_all", smoke: bool = False, status_only: bool = False):
    if status_only: print(status.remote(run)); return
    if smoke: run = run + "_smoke"
    rc = train.remote(run, steps, batch, dataset, smoke)
    print("training exit code", rc); print(status.remote(run))
