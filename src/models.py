"""Teacher (EfficientNetV2-S) and student (MobileNetV3-Small) wrappers.

Both torchvision models share the .features / .avgpool / .classifier layout,
so a single wrapper exposes (logits, feature_map) in one forward pass. The
feature_map is the pre-pool output of `.features`, used by feature- and
attention-based distillation.
"""
import torch
import torch.nn as nn
from torchvision import models


class WrappedNet(nn.Module):
    """Runs a torchvision backbone and returns (logits, last_feature_map)."""

    def __init__(self, base: nn.Module, feature_channels: int):
        super().__init__()
        self.base = base
        self.feature_channels = feature_channels

    def forward(self, x):
        f = self.base.features(x)          # [B, C, H, W]
        pooled = self.base.avgpool(f)
        pooled = torch.flatten(pooled, 1)
        logits = self.base.classifier(pooled)
        return logits, f


def build_teacher(num_classes: int, pretrained: bool = True) -> WrappedNet:
    weights = models.EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
    base = models.efficientnet_v2_s(weights=weights)
    in_features = base.classifier[1].in_features
    base.classifier[1] = nn.Linear(in_features, num_classes)
    return WrappedNet(base, feature_channels=_infer_feature_channels(base, default=1280))


def build_student(num_classes: int, pretrained: bool = True) -> WrappedNet:
    weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    base = models.mobilenet_v3_small(weights=weights)
    in_features = base.classifier[3].in_features
    base.classifier[3] = nn.Linear(in_features, num_classes)
    return WrappedNet(base, feature_channels=_infer_feature_channels(base, default=576))


def _infer_feature_channels(base: nn.Module, default: int) -> int:
    """Number of channels produced by base.features (probe with a dummy input)."""
    was_training = base.training
    base.eval()
    try:
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 64, 64)
            f = base.features(dummy)
            ch = f.shape[1]
    except Exception:
        ch = default
    finally:
        base.train(was_training)
    return ch


class FeatureAdapter(nn.Module):
    """1x1 conv mapping student feature channels -> teacher feature channels."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)

    def forward(self, x):
        return self.conv(x)
