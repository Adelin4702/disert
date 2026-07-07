"""One-command pipeline for the knowledge-distillation thesis.

Trains (or loads) the teacher, then runs each requested student method across
one or more seeds, saves every checkpoint and per-run JSON, and finally writes
an aggregated summary (main table + method-analysis table) and comparison plots.

Example (CUDA, all methods, 3 seeds):
    python src/pipeline.py --dataset cifar10 --image_size 160 --batch_size 32 \
        --epochs 50 --device cuda --seeds 0,1,2 \
        --methods baseline,classic_kd,feature_kd,attention_kd,confidence_kd,forgetting_kd,instability_kd

Smoke test (fast, proves the pipeline runs end-to-end):
    python src/pipeline.py --smoke
"""
import argparse
import json
import os
import statistics

import torch
import yaml

# allow running as `python src/pipeline.py` (add src/ to path)
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import build_loaders
from device import get_device
from seed import seed_worker, set_seed
from train_student import MethodConfig, train_student
from train_teacher import load_teacher, train_teacher

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(ROOT, "configs")
CKPT_DIR = os.path.join(ROOT, "checkpoints")
RESULTS_DIR = os.path.join(ROOT, "results")
PLOTS_DIR = os.path.join(ROOT, "plots")

# method name -> config file
METHOD_FILES = {
    "baseline": "student_baseline.yaml",
    "classic_kd": "classic_kd.yaml",
    "feature_kd": "feature_kd.yaml",
    "attention_kd": "attention_kd.yaml",
    "confidence_kd": "confidence_kd.yaml",
    "forgetting_kd": "forgetting_kd.yaml",
    "instability_kd": "instability_kd.yaml",
}


def load_method_config(name: str) -> MethodConfig:
    path = os.path.join(CONFIG_DIR, METHOD_FILES[name])
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    if "lambda" in raw:                       # yaml key -> dataclass field
        raw["lambda_"] = raw.pop("lambda")
    if "score_weights" in raw:
        raw["score_weights"] = tuple(raw["score_weights"])
    valid = MethodConfig.__dataclass_fields__.keys()
    raw = {k: v for k, v in raw.items() if k in valid}
    return MethodConfig(**raw)


def make_loaders(args, device, seed):
    gen = torch.Generator().manual_seed(seed)
    return build_loaders(
        dataset=args.dataset, data_dir=os.path.join(ROOT, "data"),
        image_size=args.image_size, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        worker_init_fn=seed_worker, generator=gen, train_subset=args.train_subset,
    )


def run(args):
    for d in (CKPT_DIR, RESULTS_DIR, PLOTS_DIR):
        os.makedirs(d, exist_ok=True)

    device = get_device(args.device)
    print(f"Using device: {device}")
    seeds = [int(s) for s in str(args.seeds).split(",") if s != ""]

    # ---- Teacher (train once, then freeze) ----
    teacher_ckpt = os.path.join(CKPT_DIR, f"teacher_efficientnetv2s_{args.dataset}.pth")
    teacher_results = os.path.join(RESULTS_DIR, f"teacher_metrics_{args.dataset}.json")
    set_seed(seeds[0])
    loaders = make_loaders(args, device, seeds[0])

    need_teacher = any(load_method_config(m).use_kd for m in args.methods)
    teacher = None
    if need_teacher:
        if os.path.exists(teacher_ckpt) and not args.retrain_teacher:
            print(f"[teacher] loading existing checkpoint {teacher_ckpt}")
            teacher = load_teacher(teacher_ckpt, loaders.num_classes, device)
        else:
            _, tres = train_teacher(
                loaders, device, epochs=args.teacher_epochs, lr=args.teacher_lr,
                weight_decay=args.weight_decay, checkpoint_path=teacher_ckpt,
                results_path=teacher_results, pretrained=args.pretrained,
                image_size=args.image_size, max_train_batches=args.max_train_batches,
                max_eval_batches=args.max_eval_batches,
            )
            teacher = load_teacher(teacher_ckpt, loaders.num_classes, device)

    # ---- Students ----
    all_runs = []
    for method in args.methods:
        cfg = load_method_config(method)
        for seed in seeds:
            set_seed(seed)
            loaders = make_loaders(args, device, seed)
            ckpt = os.path.join(CKPT_DIR, f"student_{method}_{args.dataset}_seed{seed}.pth")
            res_path = os.path.join(RESULTS_DIR, f"student_{method}_{args.dataset}_seed{seed}.json")
            res = train_student(
                cfg, loaders, teacher if cfg.use_kd else None, device,
                epochs=args.epochs, lr=args.student_lr, weight_decay=args.weight_decay,
                checkpoint_path=ckpt, results_path=res_path, pretrained=args.pretrained,
                image_size=args.image_size, seed=seed,
                max_train_batches=args.max_train_batches,
                max_eval_batches=args.max_eval_batches,
            )
            all_runs.append(res)

    teacher_metrics = None
    if os.path.exists(teacher_results):
        with open(teacher_results) as f:
            teacher_metrics = json.load(f)

    aggregate_and_report(all_runs, teacher_metrics, args)


def aggregate_and_report(runs, teacher_metrics, args):
    by_method = {}
    for r in runs:
        by_method.setdefault(r["method"], []).append(r)

    summary = {"dataset": args.dataset, "image_size": args.image_size,
               "epochs": args.epochs, "teacher": teacher_metrics, "methods": {}}

    def ms(vals):
        vals = [v for v in vals if v is not None]
        if not vals:
            return (None, None)
        return (statistics.mean(vals), statistics.stdev(vals) if len(vals) > 1 else 0.0)

    for method, rs in by_method.items():
        acc_m, acc_s = ms([r["accuracy"] for r in rs])
        f1_m, f1_s = ms([r["macro_f1"] for r in rs])
        ece_m, ece_s = ms([r["ece"] for r in rs])
        nll_m, nll_s = ms([r["nll"] for r in rs])
        an = [r["analysis"] for r in rs]
        summary["methods"][method] = {
            "n_seeds": len(rs),
            "accuracy": {"mean": acc_m, "std": acc_s},
            "macro_f1": {"mean": f1_m, "std": f1_s},
            "ece": {"mean": ece_m, "std": ece_s},
            "nll": {"mean": nll_m, "std": nll_s},
            "params": rs[0]["params"],
            "model_size_mb": rs[0]["model_size_mb"],
            "flops_macs": rs[0]["flops_macs"],
            "latency_ms": rs[0]["latency_ms"],
            "train_time_s": ms([r["train_time_s"] for r in rs])[0],
            "train_peak_memory_mb": ms([r["train_peak_memory_mb"] for r in rs])[0],
            "acc_stable_train": ms([a["acc_stable_train"] for a in an])[0],
            "acc_unstable_train": ms([a["acc_unstable_train"] for a in an])[0],
            "total_forgetting_events": ms([a["total_forgetting_events"] for a in an])[0],
            "total_prediction_flips": ms([a["total_prediction_flips"] for a in an])[0],
        }

    with open(os.path.join(RESULTS_DIR, f"summary_{args.dataset}.json"), "w") as f:
        json.dump(summary, f, indent=2)

    write_markdown(summary, args)
    make_plots(summary, args)
    print(f"\nSummary written to results/summary_{args.dataset}.md")


def _fmt(mean, std=None, pct=False, nd=2):
    if mean is None:
        return "-"
    val = mean * 100 if pct else mean
    if std is None:
        return f"{val:.{nd}f}"
    sval = std * 100 if pct else std
    return f"{val:.{nd}f} ± {sval:.{nd}f}"


def write_markdown(summary, args):
    lines = [f"# Results — {args.dataset} @ {args.image_size}px, {args.epochs} epochs\n"]
    t = summary.get("teacher")
    if t:
        lines.append(f"**Teacher (EfficientNetV2-S):** acc={t['accuracy']*100:.2f}%, "
                     f"params={t['params']:,}, size={t['model_size_mb']:.1f} MB, "
                     f"latency={t['latency_ms']:.2f} ms\n")

    lines.append("## Main comparison (student, mean ± std over seeds)\n")
    lines.append("| Method | Acc % | Macro-F1 % | ECE | NLL | Params | Size MB | FLOPs (MACs) | Latency ms | Train s |")
    lines.append("|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
    for m, d in summary["methods"].items():
        flops = f"{d['flops_macs']:.3e}" if d["flops_macs"] else "-"
        lines.append(
            f"| {m} | {_fmt(d['accuracy']['mean'], d['accuracy']['std'], pct=True)} "
            f"| {_fmt(d['macro_f1']['mean'], d['macro_f1']['std'], pct=True)} "
            f"| {_fmt(d['ece']['mean'], d['ece']['std'], nd=4)} "
            f"| {_fmt(d['nll']['mean'], d['nll']['std'], nd=4)} "
            f"| {d['params']:,} | {d['model_size_mb']:.1f} | {flops} "
            f"| {d['latency_ms']:.2f} | {_fmt(d['train_time_s'], nd=1)} |")

    lines.append("\n## Method analysis (train-set diagnostic)\n")
    lines.append("Acc on stable vs unstable train samples (split at each method's own "
                 "instability median), plus total flips / forgetting events.\n")
    lines.append("| Method | Acc stable % | Acc unstable % | Forgetting events | Prediction flips |")
    lines.append("|---|--:|--:|--:|--:|")
    for m, d in summary["methods"].items():
        lines.append(
            f"| {m} | {_fmt(d['acc_stable_train'], pct=True)} "
            f"| {_fmt(d['acc_unstable_train'], pct=True)} "
            f"| {_fmt(d['total_forgetting_events'], nd=0)} "
            f"| {_fmt(d['total_prediction_flips'], nd=0)} |")

    with open(os.path.join(RESULTS_DIR, f"summary_{args.dataset}.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def make_plots(summary, args):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("[plots] matplotlib not available; skipping plots.")
        return

    methods = list(summary["methods"].keys())
    accs = [summary["methods"][m]["accuracy"]["mean"] * 100 for m in methods]
    errs = [(summary["methods"][m]["accuracy"]["std"] or 0) * 100 for m in methods]

    fig, ax = plt.subplots(figsize=(max(6, len(methods) * 1.2), 4.5))
    ax.bar(methods, accs, yerr=errs, capsize=4, color="#4C78A8")
    if summary.get("teacher"):
        ax.axhline(summary["teacher"]["accuracy"] * 100, color="#E45756",
                   linestyle="--", label="teacher")
        ax.legend()
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title(f"Student accuracy by method — {args.dataset}")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, f"accuracy_comparison_{args.dataset}.png")
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[plots] saved {out}")


def build_argparser():
    p = argparse.ArgumentParser(description="Knowledge-distillation thesis pipeline")
    p.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100"])
    p.add_argument("--teacher", default="efficientnet_v2_s")
    p.add_argument("--student", default="mobilenet_v3_small")
    p.add_argument("--image_size", type=int, default=160)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--teacher_epochs", type=int, default=15)
    p.add_argument("--seeds", default="0")
    p.add_argument("--device", default="auto")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--student_lr", type=float, default=1e-3)
    p.add_argument("--teacher_lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--pretrained", action="store_true", default=True)
    p.add_argument("--no_pretrained", dest="pretrained", action="store_false")
    p.add_argument("--retrain_teacher", action="store_true")
    p.add_argument("--methods", default="baseline,classic_kd,feature_kd,attention_kd,confidence_kd,forgetting_kd,instability_kd")
    p.add_argument("--train_subset", type=int, default=0, help="use first N train samples (dev)")
    p.add_argument("--max_train_batches", type=int, default=0, help="cap train batches/epoch (dev)")
    p.add_argument("--max_eval_batches", type=int, default=0, help="cap eval batches (dev)")
    p.add_argument("--smoke", action="store_true", help="tiny end-to-end correctness run")
    return p


def main():
    args = build_argparser().parse_args()
    args.methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    unknown = [m for m in args.methods if m not in METHOD_FILES]
    if unknown:
        raise SystemExit(f"Unknown methods: {unknown}. Choose from {list(METHOD_FILES)}")

    if args.smoke:
        # Minimal config that exercises every code path quickly.
        args.epochs = 2
        args.teacher_epochs = 1
        args.batch_size = 8
        args.image_size = 96
        args.train_subset = 64
        args.max_eval_batches = 4
        args.num_workers = 0
    run(args)


if __name__ == "__main__":
    main()
