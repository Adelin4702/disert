# Knowledge Distillation Thesis — Student-Instability Guided KD

Teacher **EfficientNetV2-S** → student **MobileNetV3-Small**, on **CIFAR-10**
(CIFAR-100 optional). Compares a baseline student against classic, feature, and
attention KD, two reweighting baselines, and the proposed **Student-Instability
Guided KD**.

> CIFAR is 32×32; images are upsampled to `--image_size` (default 160) to reuse
> ImageNet-pretrained backbones. This is a deliberate transfer shortcut —
> accuracies are **not** comparable to native-32px CIFAR papers.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Runs on CUDA, Apple Silicon (MPS), or CPU — `--device auto` picks the best.
Note: AMP applies only on CUDA; MPS runs fp32. Report latency/throughput from
the CUDA machine.

## Run

Full CUDA experiment (all methods, 3 seeds):

```bash
python src/pipeline.py --dataset cifar10 --image_size 160 --batch_size 32 \
  --epochs 50 --device cuda --seeds 0,1,2 \
  --methods baseline,classic_kd,feature_kd,attention_kd,confidence_kd,forgetting_kd,instability_kd
```

MacBook Air M5 development run:

```bash
python src/pipeline.py --dataset cifar10 --image_size 160 --batch_size 8 \
  --epochs 20 --device mps --seeds 0 \
  --methods baseline,classic_kd,instability_kd
```

Smoke test (tiny subset, ~minutes, proves everything wires up):

```bash
python src/pipeline.py --smoke --methods baseline,classic_kd,instability_kd
```

## Data download (and flaky networks)

torchvision fetches CIFAR from `cs.toronto.edu`, which **resets sustained
transfers and does not support resume** on many connections
(`Connection reset by peer`). Use the fast.ai AWS-S3 mirror instead — it
supports HTTP range/resume and is CDN-backed:

```bash
bash scripts/download_cifar.sh cifar10    # -> data/cifar10/{train,test}/<class>/*.png
bash scripts/download_cifar.sh cifar100   # optional
```

This extracts an **ImageFolder** layout that `src/data.py` auto-detects (no
code change, no torchvision download). Direct URLs, if you prefer to fetch
manually (both support `curl -C -` resume):

- CIFAR-10:  `https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz`
- CIFAR-100: `https://s3.amazonaws.com/fast-ai-imageclas/cifar100.tgz`

> Note: this mirror stores CIFAR as PNGs in class folders (alphabetical class
> order), not the original pickle batches. Data is identical; only the integer
> label ordering may differ from canonical CIFAR — irrelevant here since teacher
> and students share the same mapping.

Pretrained ImageNet weights download from `download.pytorch.org` (same CDN as
the PyTorch wheels, usually reliable). If that host also fails, pre-fetch them:

```bash
bash scripts/download_weights.sh          # -> ~/.cache/torch/hub/checkpoints/*.pth
```

Or run with `--no_pretrained` to train both networks from scratch (no weight
download; needs more epochs to converge).

## Running on RunPod / a cloud GPU (16 GB)

The whole job fits in ~4–5 GB, so a 16 GB card is comfortable and fast. Use a
**PyTorch 2.x** pod template (torch + CUDA preinstalled), clone into the
persistent `/workspace` volume, then:

```bash
cd /workspace/disert
bash scripts/setup_runpod.sh          # checks CUDA, installs deps, downloads CIFAR-10

python src/pipeline.py --dataset cifar10 --device cuda --image_size 160 \
  --batch_size 128 --num_workers 8 --epochs 50 --seeds 0,1,2 \
  --methods baseline,classic_kd,feature_kd,attention_kd,confidence_kd,forgetting_kd,instability_kd
```

Batch-size guidance for 16 GB: **128 @ 160px**, **64–96 @ 224px**. AMP (mixed
precision) turns on automatically on CUDA. Keep `data/`, `checkpoints/`,
`results/` on `/workspace` so they persist across pod restarts.

## Methods

| Name | What it does |
|---|---|
| `baseline` | CE only, no teacher |
| `classic_kd` | Hinton KD (CE + KL with T² scaling) |
| `feature_kd` | + MSE on features via a 1×1 channel adapter |
| `attention_kd` | + attention transfer (Zagoruyko & Komodakis) |
| `confidence_kd` | reweighting baseline: static difficulty (1 − maxprob) |
| `forgetting_kd` | reweighting baseline: forgetting events (Toneva et al.) |
| `instability_kd` | **ours**: 0.5·flip + 0.3·forget + 0.2·uncertainty |

The two reweighting baselines exist to show the instability signal beats simple
per-example difficulty weighting — otherwise the contribution reduces to "we
reweight hard examples". `normalize_weights: true` (per-batch mean weight = 1)
decouples the weighting from effective learning rate.

## Outputs

- `checkpoints/` — teacher + one student `.pth` per method × seed
- `results/` — per-run JSON, plus `summary_<dataset>.{json,md}` (main table +
  analysis table)
- `plots/` — accuracy comparison bar chart

## Layout

```
configs/   one YAML per method (loss + weighting config)
src/        data, models, losses, metrics, instability_memory, train_*, pipeline, device, seed
checkpoints/ results/ plots/ data/
```
