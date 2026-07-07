"""Distillation losses.

Per-sample CE and KD are returned unreduced so the training loop can apply
per-sample instability weights. Feature and attention losses are scalar means.
"""
import torch
import torch.nn.functional as F


def ce_per_sample(student_logits, labels):
    return F.cross_entropy(student_logits, labels, reduction="none")


def kd_per_sample(student_logits, teacher_logits, temperature: float):
    """Classic Hinton KD, per sample, with the T^2 gradient-scaling factor.

    Omitting T^2 is a common bug: it makes gradient magnitude depend on T and
    silently changes results across temperatures.
    """
    t = temperature
    log_p_student = F.log_softmax(student_logits / t, dim=1)
    p_teacher = F.softmax(teacher_logits / t, dim=1)
    # KL(teacher || student) per sample, summed over classes.
    kl = F.kl_div(log_p_student, p_teacher, reduction="none").sum(dim=1)
    return (t * t) * kl


def _attention_map_2d(feature):
    """Spatial attention: mean of squared activations over channels -> [B,1,H,W]."""
    return feature.pow(2).mean(dim=1, keepdim=True)


def attention_loss(student_feat, teacher_feat):
    """Zagoruyko & Komodakis (2017) attention transfer.

    Spatial dims are matched by interpolating the student map to the teacher's
    size; each map is L2-normalized per sample before the MSE.
    """
    a_s = _attention_map_2d(student_feat)
    a_t = _attention_map_2d(teacher_feat)
    if a_s.shape[-2:] != a_t.shape[-2:]:
        a_s = F.interpolate(a_s, size=a_t.shape[-2:], mode="bilinear", align_corners=False)
    a_s = F.normalize(a_s.flatten(1), p=2, dim=1)
    a_t = F.normalize(a_t.flatten(1), p=2, dim=1)
    return (a_s - a_t).pow(2).sum(dim=1).mean()


def feature_loss(student_feat_adapted, teacher_feat):
    """MSE between channel-adapted student feature and teacher feature.

    Student feature (already channel-matched by the 1x1 adapter) is spatially
    interpolated to the teacher's map size.
    """
    fs = student_feat_adapted
    if fs.shape[-2:] != teacher_feat.shape[-2:]:
        fs = F.interpolate(fs, size=teacher_feat.shape[-2:], mode="bilinear", align_corners=False)
    return F.mse_loss(fs, teacher_feat)
