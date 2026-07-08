"""Evaluate already-trained checkpoints and (re)generate the summary + plots.

Use this when training finished (or you have the .pth files) but you want to
recompute test metrics / tables / plots without retraining.

    python src/evaluate_checkpoints.py --dataset cifar10 --image_size 160 --seed 0

It reuses the pipeline's own evaluate() and aggregate_and_report(), so the
outputs are identical in format to a full run.

Note: the per-method *analysis* table (prediction flips, forgetting events,
stable-vs-unstable accuracy) is derived from TRAINING dynamics, which a
checkpoint does not contain. If the original results/student_*.json files still
exist, their analysis is reused; otherwise those columns are left blank. All
other metrics are recomputed fresh from the checkpoints.
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import build_loaders
from device import get_device
from metrics import efficiency_report, evaluate
from models import build_student
from pipeline import (CKPT_DIR, PLOTS_DIR, RESULTS_DIR, ROOT, aggregate_and_report,
                      METHOD_FILES)
from train_teacher import load_teacher

_EMPTY_ANALYSIS = {
    "total_prediction_flips": None,
    "total_forgetting_events": None,
    "acc_stable_train": None,
    "acc_unstable_train": None,
    "instability_median": None,
}


def _load_student(ckpt, num_classes, device):
    model = build_student(num_classes, pretrained=False).to(device)
    state = torch.load(ckpt, map_location=device, weights_only=False)
    # strict=False tolerates stray thop buffers in older checkpoints.
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def _existing_analysis(method, dataset, seed):
    """Reuse the analysis block from the original training JSON if present."""
    path = os.path.join(RESULTS_DIR, f"student_{method}_{dataset}_seed{seed}.json")
    if os.path.exists(path):
        try:
            with open(path) as f:
                data = json.load(f)
            if isinstance(data.get("analysis"), dict):
                return data["analysis"]
        except Exception:
            pass
    return dict(_EMPTY_ANALYSIS)


def main():
    p = argparse.ArgumentParser(description="Evaluate checkpoints; rebuild summary + plots")
    p.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100"])
    p.add_argument("--image_size", type=int, default=160,
                   help="MUST match the value used during training")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=50, help="only used for the summary title")
    p.add_argument("--methods", default=",".join(METHOD_FILES))
    p.add_argument("--max_eval_batches", type=int, default=0)
    args = p.parse_args()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    for d in (RESULTS_DIR, PLOTS_DIR):
        os.makedirs(d, exist_ok=True)

    device = get_device(args.device)
    print(f"Using device: {device}")

    loaders = build_loaders(
        dataset=args.dataset, data_dir=os.path.join(ROOT, "data"),
        image_size=args.image_size, batch_size=args.batch_size, device=device,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )

    # ---- Teacher ----
    teacher_metrics = None
    teacher_ckpt = os.path.join(CKPT_DIR, f"teacher_efficientnetv2s_{args.dataset}.pth")
    if os.path.exists(teacher_ckpt):
        print(f"[teacher] evaluating {teacher_ckpt}")
        teacher = load_teacher(teacher_ckpt, loaders.num_classes, device)
        em = evaluate(teacher, loaders.test, device, loaders.num_classes,
                      transform=loaders.eval_tf, max_batches=args.max_eval_batches)
        eff = efficiency_report(teacher, device, args.image_size)
        teacher_metrics = {"role": "teacher", "model": "efficientnet_v2_s", **em, **eff}
        with open(os.path.join(RESULTS_DIR, f"teacher_metrics_{args.dataset}.json"), "w") as f:
            json.dump(teacher_metrics, f, indent=2)
        print(f"[teacher] acc={em['accuracy']:.4f}")
        del teacher
    else:
        print(f"[teacher] checkpoint not found: {teacher_ckpt} (skipping teacher row)")

    # ---- Students ----
    runs = []
    for method in methods:
        ckpt = os.path.join(CKPT_DIR, f"student_{method}_{args.dataset}_seed{args.seed}.pth")
        if not os.path.exists(ckpt):
            print(f"[{method}] checkpoint not found, skipping: {ckpt}")
            continue
        model = _load_student(ckpt, loaders.num_classes, device)
        em = evaluate(model, loaders.test, device, loaders.num_classes,
                      transform=loaders.eval_tf, max_batches=args.max_eval_batches)
        eff = efficiency_report(model, device, args.image_size)
        res = {
            "role": "student", "method": method, "seed": args.seed,
            "train_time_s": None, "train_peak_memory_mb": None,
            **em, **eff,
            "analysis": _existing_analysis(method, args.dataset, args.seed),
        }
        # write/refresh the per-run json (keeps original analysis if it was there)
        with open(os.path.join(RESULTS_DIR, f"student_{method}_{args.dataset}_seed{args.seed}.json"), "w") as f:
            json.dump(res, f, indent=2)
        runs.append(res)
        print(f"[{method}] acc={em['accuracy']:.4f}  f1={em['macro_f1']:.4f}  ece={em['ece']:.4f}")

    if not runs:
        raise SystemExit("No student checkpoints found — nothing to summarize.")

    aggregate_and_report(runs, teacher_metrics, args)
    print(f"\nWrote results/summary_{args.dataset}.md and plots/accuracy_comparison_{args.dataset}.png")


if __name__ == "__main__":
    main()
