#!/usr/bin/env bash
# One-shot setup for a RunPod GPU pod (or any CUDA box).
#
# Recommended pod: a "PyTorch 2.x" template (torch + CUDA preinstalled) with a
# 16 GB+ GPU (RTX 4000 Ada / A4000 / L4 / A5000 all work). Run from the repo
# root on the persistent /workspace volume:
#
#   cd /workspace/disert && bash scripts/setup_runpod.sh
#
set -euo pipefail

echo "== python =="
python -c "import sys; print(sys.version)"

echo "== torch (CUDA) =="
if ! python -c "import torch" 2>/dev/null; then
  echo "torch not found; installing CUDA build (cu124)..."
  pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cu124
fi
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0),
          f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
PY

echo "== project deps =="
pip install --quiet pyyaml matplotlib thop

echo "== data (fast.ai mirror; reliable on any network) =="
bash scripts/download_cifar.sh cifar10

echo
echo "Setup done. Example full run (16 GB GPU):"
echo "  python src/pipeline.py --dataset cifar10 --device cuda --image_size 160 \\"
echo "    --batch_size 128 --num_workers 8 --epochs 50 --seeds 0,1,2 \\"
echo "    --methods baseline,classic_kd,feature_kd,attention_kd,confidence_kd,forgetting_kd,instability_kd"
