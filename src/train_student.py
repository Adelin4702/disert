"""Train a MobileNetV3-Small student under one distillation method.

Supported methods (via MethodConfig):
  baseline        - CE only, no teacher
  classic_kd      - CE + KD (Hinton)
  feature_kd      - classic_kd + feature MSE (1x1 adapter)
  attention_kd    - classic_kd + attention transfer
  confidence_kd   - classic_kd, per-sample weight from instantaneous (1 - maxprob)
  forgetting_kd   - classic_kd, per-sample weight from forgetting rate
  instability_kd  - classic_kd, per-sample weight from full instability score  (ours)

Per-sample weighting multiplies the KD term:
    loss_i = (1 - alpha) * CE_i + alpha * w_i * KD_i         (+ feature/attention)
with, after warmup,  w_i = 1 + lambda * score_i.

`normalize_weights` rescales w within each batch so mean(w) == 1. This is the
critical control: without it, "stronger weighting" is confounded with a larger
effective learning rate, and any gain could be attributed to that instead of to
the instability signal.
"""
import json
import time
from contextlib import nullcontext
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast

from device import device_supports_amp, peak_memory_mb, reset_peak_memory
from instability_memory import InstabilityMemory
from losses import attention_loss, ce_per_sample, feature_loss, kd_per_sample
from metrics import efficiency_report, evaluate
from models import FeatureAdapter, build_student


@dataclass
class MethodConfig:
    name: str
    use_kd: bool = False
    alpha: float = 0.5
    temperature: float = 4.0
    feature_kd: bool = False
    feature_beta: float = 100.0
    attention_kd: bool = False
    attention_gamma: float = 1000.0
    weighting: str = "none"            # none | confidence | forgetting | instability
    lambda_: float = 1.0
    warmup_epochs: int = 3
    normalize_weights: bool = True
    score_weights: tuple = (0.5, 0.3, 0.2)


def _sample_weights(cfg, epoch, idx, confidence, memory, device):
    """Per-sample KD weights for this batch (all ones for unweighted methods)."""
    bs = idx.size(0)
    if cfg.weighting == "none" or epoch < cfg.warmup_epochs:
        return torch.ones(bs, device=device)

    if cfg.weighting == "confidence":
        score = (1.0 - confidence).detach()             # instantaneous, [B]
    elif cfg.weighting in ("forgetting", "instability"):
        score = memory.scores(cfg.weighting)[idx.cpu()].to(device)
    else:
        raise ValueError(f"unknown weighting {cfg.weighting!r}")

    w = 1.0 + cfg.lambda_ * score
    if cfg.normalize_weights:
        w = w / w.mean().clamp(min=1e-8)
    return w


def train_student(cfg: MethodConfig, loaders, teacher, device, epochs: int,
                  lr: float, weight_decay: float, checkpoint_path: str,
                  results_path: str, pretrained: bool = True, image_size: int = 160,
                  seed: int = 0, max_train_batches: int = 0,
                  max_eval_batches: int = 0):
    student = build_student(loaders.num_classes, pretrained=pretrained).to(device)

    params = list(student.parameters())
    adapter = None
    if cfg.feature_kd:
        adapter = FeatureAdapter(student.feature_channels, teacher.feature_channels).to(device)
        params += list(adapter.parameters())

    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    use_amp = device_supports_amp(device)
    scaler = GradScaler(enabled=use_amp)
    amp_ctx = (lambda: autocast(device_type=device.type)) if use_amp else nullcontext

    # A tracking memory is kept for every method so the analysis table can be
    # filled for the baselines too (it only *drives* weighting when configured).
    memory = InstabilityMemory(loaders.num_train, score_weights=cfg.score_weights)

    if teacher is not None:
        teacher.eval()

    reset_peak_memory(device)
    t0 = time.perf_counter()
    for epoch in range(epochs):
        student.train()
        running, seen = 0.0, 0
        for b, (images, labels, idx) in enumerate(loaders.train):
            if max_train_batches and b >= max_train_batches:
                break
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            idx = idx.to(device)

            optimizer.zero_grad(set_to_none=True)
            with amp_ctx():
                s_logits, s_feat = student(images)

                if cfg.use_kd:
                    with torch.no_grad():
                        t_logits, t_feat = teacher(images)

                ce_i = ce_per_sample(s_logits, labels)

                if cfg.use_kd:
                    kd_i = kd_per_sample(s_logits, t_logits, cfg.temperature)
                    probs = F.softmax(s_logits.detach(), dim=1)
                    conf, preds = probs.max(dim=1)
                    w = _sample_weights(cfg, epoch, idx, conf, memory, device)
                    loss = ((1.0 - cfg.alpha) * ce_i + cfg.alpha * w * kd_i).mean()

                    if cfg.feature_kd:
                        loss = loss + cfg.feature_beta * feature_loss(adapter(s_feat), t_feat)
                    if cfg.attention_kd:
                        loss = loss + cfg.attention_gamma * attention_loss(s_feat, t_feat)
                else:
                    loss = ce_i.mean()
                    conf, preds = F.softmax(s_logits.detach(), dim=1).max(dim=1)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            memory.update(idx, preds, labels, conf)
            running += loss.item() * images.size(0)
            seen += images.size(0)

        scheduler.step()
        print(f"[{cfg.name}|seed{seed}] epoch {epoch + 1}/{epochs}  loss={running / max(seen, 1):.4f}")

    train_time = time.perf_counter() - t0
    eval_metrics = evaluate(student, loaders.test, device, loaders.num_classes,
                            max_batches=max_eval_batches)
    eff = efficiency_report(student, device, image_size)

    torch.save(student.state_dict(), checkpoint_path)
    results = {
        "role": "student",
        "method": cfg.name,
        "seed": seed,
        "train_time_s": train_time,
        "train_peak_memory_mb": peak_memory_mb(device),
        **eval_metrics,
        **eff,
        "analysis": memory.analysis(),
    }
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[{cfg.name}|seed{seed}] test acc={eval_metrics['accuracy']:.4f}  saved -> {checkpoint_path}")
    return results
