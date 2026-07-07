"""Evaluation metrics and efficiency measurements.

Accuracy, macro-F1, ECE, NLL for quality; params, model size, FLOPs, latency,
peak memory for the compression story. FLOPs uses `thop` if installed and is
skipped gracefully otherwise.
"""
import os
import tempfile
import time

import torch
import torch.nn.functional as F

from device import peak_memory_mb, reset_peak_memory


@torch.no_grad()
def evaluate(model, loader, device, num_classes: int, transform=None,
             n_ece_bins: int = 15, max_batches: int = 0):
    model.eval()
    reset_peak_memory(device)

    all_probs, all_labels = [], []
    for b, batch in enumerate(loader):
        if max_batches and b >= max_batches:
            break
        images, labels = batch[0], batch[1]
        images = images.to(device, non_blocking=True)
        if transform is not None:
            images = transform(images)
        logits, _ = model(images)
        probs = F.softmax(logits, dim=1).cpu()
        all_probs.append(probs)
        all_labels.append(labels)

    probs = torch.cat(all_probs)
    labels = torch.cat(all_labels)
    preds = probs.argmax(dim=1)

    acc = (preds == labels).float().mean().item()
    f1 = macro_f1(preds, labels, num_classes)
    ece = expected_calibration_error(probs, labels, n_ece_bins)
    nll = F.nll_loss(torch.log(probs.clamp(min=1e-12)), labels).item()

    return {
        "accuracy": acc,
        "macro_f1": f1,
        "ece": ece,
        "nll": nll,
        "eval_peak_memory_mb": peak_memory_mb(device),
    }


def macro_f1(preds, labels, num_classes: int) -> float:
    f1s = []
    for c in range(num_classes):
        tp = ((preds == c) & (labels == c)).sum().item()
        fp = ((preds == c) & (labels != c)).sum().item()
        fn = ((preds != c) & (labels == c)).sum().item()
        if tp + fp == 0 or tp + fn == 0:
            f1s.append(0.0)
            continue
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        f1s.append(0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))
    return sum(f1s) / len(f1s)


def expected_calibration_error(probs, labels, n_bins: int = 15) -> float:
    conf, preds = probs.max(dim=1)
    correct = (preds == labels).float()
    bins = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = probs.size(0)
    for i in range(n_bins):
        lo, hi = bins[i].item(), bins[i + 1].item()
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        acc_bin = correct[mask].mean().item()
        conf_bin = conf[mask].mean().item()
        ece += (mask.float().mean().item()) * abs(acc_bin - conf_bin)
    return ece


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def model_size_mb(model) -> float:
    """Serialized state_dict size on disk in MB."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as f:
        path = f.name
    try:
        torch.save(model.state_dict(), path)
        size = os.path.getsize(path) / 1e6
    finally:
        os.remove(path)
    return size


@torch.no_grad()
def measure_latency(model, device, image_size: int, batch_size: int = 1,
                    warmup: int = 5, iters: int = 20):
    """Mean forward latency (ms) per batch. Report from CUDA for the thesis."""
    model.eval()
    x = torch.randn(batch_size, 3, image_size, image_size, device=device)
    for _ in range(warmup):
        model(x)
    _sync(device)
    start = time.perf_counter()
    for _ in range(iters):
        model(x)
    _sync(device)
    total = time.perf_counter() - start
    return (total / iters) * 1000.0


def measure_flops(model, device, image_size: int):
    """MACs/FLOPs via thop if available, else None.

    Profiles a deep copy: thop attaches `total_ops`/`total_params` buffers to
    every submodule, which would otherwise leak into the saved checkpoint and
    break a strict load.
    """
    try:
        import copy

        from thop import profile
    except Exception:
        return None
    x = torch.randn(1, 3, image_size, image_size, device=device)
    try:
        m = copy.deepcopy(model)
        m.eval()
        macs, _ = profile(m, inputs=(x,), verbose=False)
        del m
        return float(macs)
    except Exception:
        return None


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        try:
            torch.mps.synchronize()
        except Exception:
            pass


def efficiency_report(model, device, image_size: int):
    return {
        "params": count_params(model),
        "model_size_mb": model_size_mb(model),
        "flops_macs": measure_flops(model, device, image_size),
        "latency_ms": measure_latency(model, device, image_size),
    }
