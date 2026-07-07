"""Fine-tune the EfficientNetV2-S teacher on CIFAR and save its checkpoint.

The teacher is trained ONCE and then frozen; every student method distills from
the same checkpoint.
"""
import json
import os
import time
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast

from device import device_supports_amp, peak_memory_mb, reset_peak_memory
from metrics import efficiency_report, evaluate
from models import build_teacher


def train_teacher(loaders, device, epochs: int, lr: float, weight_decay: float,
                  checkpoint_path: str, results_path: str, pretrained: bool = True,
                  image_size: int = 160, max_train_batches: int = 0,
                  max_eval_batches: int = 0):
    model = build_teacher(loaders.num_classes, pretrained=pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    use_amp = device_supports_amp(device)
    scaler = GradScaler(enabled=use_amp)
    amp_ctx = (lambda: autocast(device_type=device.type)) if use_amp else nullcontext

    reset_peak_memory(device)
    t0 = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        running, seen = 0.0, 0
        for b, (images, labels, _) in enumerate(loaders.train):
            if max_train_batches and b >= max_train_batches:
                break
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with amp_ctx():
                logits, _ = model(images)
                loss = F.cross_entropy(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * images.size(0)
            seen += images.size(0)
        scheduler.step()
        print(f"[teacher] epoch {epoch + 1}/{epochs}  loss={running / max(seen, 1):.4f}")

    train_time = time.perf_counter() - t0
    eval_metrics = evaluate(model, loaders.test, device, loaders.num_classes,
                            max_batches=max_eval_batches)
    eff = efficiency_report(model, device, image_size)

    torch.save(model.state_dict(), checkpoint_path)
    results = {
        "role": "teacher",
        "model": "efficientnet_v2_s",
        "train_time_s": train_time,
        "train_peak_memory_mb": peak_memory_mb(device),
        **eval_metrics,
        **eff,
    }
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[teacher] test acc={eval_metrics['accuracy']:.4f}  saved -> {checkpoint_path}")
    return model, results


def load_teacher(checkpoint_path: str, num_classes: int, device) -> nn.Module:
    model = build_teacher(num_classes, pretrained=False).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    # strict=False tolerates stray thop profiling buffers (total_ops/total_params)
    # that older checkpoints may contain; real weights all match by name.
    model.load_state_dict(state, strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model
