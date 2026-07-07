# Running the thesis project on RunPod (step by step)

This walks you from zero to a finished run with **saved results and
checkpoints**, written for someone who has never used RunPod.

The whole job needs only ~4–5 GB of VRAM, so any **16 GB GPU** is comfortable
and fast.

---

## 0. The one thing you must understand first: storage

RunPod has three storage areas. Getting this right is what "save my
checkpoints" depends on.

| Area | Mount | Survives pod **Stop**? | Survives pod **Terminate**? |
|---|---|:--:|:--:|
| Container disk | `/` | ❌ wiped | ❌ gone |
| Volume disk | `/workspace` | ✅ yes | ❌ deleted with the pod |
| **Network Volume** | `/workspace` | ✅ yes | ✅ **persists**, reattachable |

**Rules of thumb**
- Always put the project (code, `data/`, `checkpoints/`, `results/`) under
  **`/workspace`**, never in the home dir or `/root`.
- For a short one-off run, the ordinary volume disk is fine — just **download
  your outputs before you Terminate** (step 7).
- If you want checkpoints to live on after the pod is gone (recommended for a
  thesis), create a **Network Volume** and attach it — then `/workspace`
  contents survive termination and you can mount them on a future pod.

---

## 1. Create an account and add credit

1. Sign up at <https://runpod.io> and add a little credit (a few dollars is
   plenty — a 16 GB GPU is roughly $0.20–0.44/hr).

## 2. (Optional but recommended) Create a Network Volume

Do this if you want checkpoints to survive after you delete the pod.

1. Left sidebar → **Storage** → **Network Volumes** → **+ Network Volume**.
2. Pick a **data center**, name it e.g. `disert-vol`, size **20 GB** (enough
   for CIFAR + all checkpoints), Create.
3. Remember the data center — your pod must be in the **same** one.

## 3. Deploy a pod

1. Left sidebar → **Pods** → **+ Deploy** (or **GPU Pods**).
2. **GPU:** choose a 16 GB card — any of: **RTX A4000, RTX 4000 Ada, L4,
   RTX A5000**. All are fine and cheap.
3. **Template:** search and pick a **"RunPod PyTorch 2.x"** template (it comes
   with Python, PyTorch, CUDA, and JupyterLab preinstalled).
4. If you made a Network Volume: expand storage options and **attach it** (it
   mounts at `/workspace`). Otherwise set **Volume Disk** ≈ 20 GB (mounts at
   `/workspace`). Container disk 10–20 GB is fine.
5. **Deploy On-Demand** (not Spot — Spot can be interrupted mid-run).
6. Wait ~1 min for status **Running**.

## 4. Open a terminal on the pod

Two easy ways:

- **JupyterLab (simplest):** on the pod card click **Connect → Jupyter Lab**
  (or the HTTP :8888 link). In Jupyter: **File → New → Terminal**.
- **Web terminal:** **Connect → Start Web Terminal → Connect**.

All commands below run in that terminal.

```bash
cd /workspace          # <-- always work here so outputs persist
```

## 5. Get the project onto the pod

Pick **one** option.

### Option A — via `runpodctl` (no GitHub account needed)

On **your Mac**, install the tool once and send the folder:

```bash
brew install runpod/runpodctl/runpodctl        # or see github.com/runpod/runpodctl
cd ~/Desktop
tar czf disert.tar.gz --exclude='disert/.venv' --exclude='disert/data' disert
runpodctl send disert.tar.gz
```

It prints a one-time code like `8338-galileo-...`. On the **pod**:

```bash
cd /workspace
runpodctl receive <the-code-it-printed>
tar xzf disert.tar.gz && cd disert
```

(We exclude `.venv` and `data` — the pod rebuilds those; no need to upload
gigabytes.)

### Option B — via GitHub (best if you'll iterate / want version control)

On your Mac, once:

```bash
cd ~/Desktop/disert
git init && git add -A && git commit -m "thesis KD project"
# create an empty repo on github.com, then:
git remote add origin https://github.com/<you>/disert.git
git branch -M main && git push -u origin main
```

On the pod:

```bash
cd /workspace
git clone https://github.com/<you>/disert.git && cd disert
```

## 6. Set up and run

From `/workspace/disert` on the pod:

```bash
bash scripts/setup_runpod.sh
```

This checks the GPU is visible, installs the few extra deps, and downloads
CIFAR-10 (from the reliable fast.ai mirror). You should see a line like
`cuda_available True   gpu: NVIDIA RTX A4000 16.0 GB`.

**Start the run inside `tmux`** so it keeps going even if your browser tab
disconnects:

```bash
tmux new -s train        # opens a persistent session

python src/pipeline.py --dataset cifar10 --device cuda --image_size 160 \
  --batch_size 128 --num_workers 8 --epochs 50 --seeds 0,1,2 \
  --methods baseline,classic_kd,feature_kd,attention_kd,confidence_kd,forgetting_kd,instability_kd \
  2>&1 | tee results/run.log
```

- Detach (leave it running): press **Ctrl-b** then **d**.
- Reattach later: `tmux attach -t train`.
- Watch progress without attaching: `tail -f results/run.log`.

**Tip — do a cheap dry run first** to confirm everything works before paying
for the full sweep:

```bash
python src/pipeline.py --smoke --methods baseline,classic_kd,instability_kd
```

### Rough time / cost (16 GB GPU, 160px)

| Scope | Approx wall-clock |
|---|---|
| `--smoke` | a few minutes |
| 7 methods × **1 seed** × 50 epochs | ~2–3 h |
| 7 methods × **3 seeds** × 50 epochs | ~6–10 h |

The teacher is trained once and reused; students are tiny.

## 7. Save your outputs and checkpoints (do this before Terminating!)

Everything lands in `results/`, `plots/`, and `checkpoints/` under
`/workspace/disert`. To pull them to your laptop:

**Bundle them:**

```bash
bash scripts/pack_outputs.sh final          # -> outputs_final.tar.gz (incl. checkpoints)
# or skip the big model files:
bash scripts/pack_outputs.sh final --no-ckpt
```

**Then download** (either way):

- **Jupyter:** in the file browser, right-click `outputs_final.tar.gz` →
  **Download**. (Individual files in `checkpoints/` / `results/` work too.)
- **runpodctl:** on the pod `runpodctl send outputs_final.tar.gz`, then on your
  Mac `runpodctl receive <code>`.

Checkpoint sizes: teacher ≈ 82 MB, each student ≈ 6 MB (21 students ≈ 130 MB),
so the full bundle is ~210 MB.

If you attached a **Network Volume**, the files also remain safely at
`/workspace` and you can reattach the volume to a future pod instead of
re-downloading.

## 8. Stop or terminate to stop billing

- **Stop** the pod (pause): keeps `/workspace`, still bills a small storage fee.
- **Terminate** (delete): stops all billing. `/workspace` is lost **unless** it
  was a Network Volume. → Only Terminate after step 7.

---

## Troubleshooting

- **`cuda_available False`** — you deployed a CPU pod or a non-PyTorch template.
  Redeploy with a GPU + "RunPod PyTorch" template.
- **Out of memory** — lower `--batch_size` (96 or 64), or `--image_size 128`.
- **Weights download fails** (`download.pytorch.org`) — run
  `bash scripts/download_weights.sh`, or add `--no_pretrained`.
- **Browser tab closed and training stopped** — you forgot `tmux`. Restart
  inside `tmux new -s train` and detach with Ctrl-b d.
- **Want outputs on a mounted path directly** — keep the repo under
  `/workspace` (already persistent); no extra config needed.
