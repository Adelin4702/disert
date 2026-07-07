"""Per-sample training-dynamics memory for the Student-Instability Guided KD.

Cheap: a handful of arrays sized to the training set. Tracks, per sample:
  - prediction flips  (prediction changed vs previous epoch)
  - forgetting events (was correct, became incorrect)  [Toneva et al., 2019]
  - uncertainty       (1 - max softmax probability)

Each component is normalised to [0, 1] so the fixed score weights are
meaningful (raw flip counts and entropy live on different scales).

    instability_score = w_flip * flip_rate
                      + w_forget * forget_rate
                      + w_uncert * uncertainty

The score drives a per-sample weight:  weight = 1 + lambda * score, applied
only after a warmup (early epochs have no history -> weight = 1 for all).

A tracking memory is maintained for EVERY method (not just the weighted one)
so the analysis table can report flips / forgetting / stable-vs-unstable
accuracy for the baselines too.
"""
import torch


class InstabilityMemory:
    def __init__(self, n: int, score_weights=(0.5, 0.3, 0.2)):
        self.n = n
        self.w_flip, self.w_forget, self.w_uncert = score_weights
        self.prev_pred = torch.full((n,), -1, dtype=torch.long)
        self.prev_correct = torch.zeros(n, dtype=torch.bool)
        self.flips = torch.zeros(n)
        self.forgets = torch.zeros(n)
        self.obs = torch.zeros(n)          # times each sample has been seen
        self.uncertainty = torch.zeros(n)  # last (1 - max prob)

    @torch.no_grad()
    def update(self, idx, preds, labels, confidence):
        """Update stats from one training batch (using student's forward preds)."""
        idx = idx.detach().cpu()
        preds = preds.detach().cpu()
        labels = labels.detach().cpu()
        conf = confidence.detach().cpu()

        prev = self.prev_pred[idx]
        seen = prev != -1
        flip = seen & (prev != preds)
        self.flips[idx] += flip.float()

        now_correct = preds == labels
        forget = self.prev_correct[idx] & (~now_correct) & seen
        self.forgets[idx] += forget.float()

        self.prev_pred[idx] = preds
        self.prev_correct[idx] = now_correct
        self.uncertainty[idx] = 1.0 - conf
        self.obs[idx] += 1.0

    def flip_rate(self):
        return self.flips / self.obs.clamp(min=1.0)

    def forget_rate(self):
        return self.forgets / self.obs.clamp(min=1.0)

    def instability(self):
        return (self.w_flip * self.flip_rate()
                + self.w_forget * self.forget_rate()
                + self.w_uncert * self.uncertainty)

    def scores(self, kind: str):
        """Full-dataset score vector for a weighting scheme."""
        if kind == "forgetting":
            return self.forget_rate()
        if kind == "instability":
            return self.instability()
        raise ValueError(f"memory has no score kind {kind!r}")

    def analysis(self):
        """Summary for the method-specific analysis table.

        acc_stable / acc_unstable are computed on the TRAIN set using this
        method's own instability median as the split (a diagnostic, disclosed
        as such in the thesis).
        """
        inst = self.instability()
        median = inst.median().item()
        stable = inst <= median
        unstable = inst > median
        correct = self.prev_correct.float()
        seen = self.obs > 0
        return {
            "total_prediction_flips": float(self.flips.sum().item()),
            "total_forgetting_events": float(self.forgets.sum().item()),
            "instability_median": median,
            "acc_stable_train": _masked_mean(correct, stable & seen),
            "acc_unstable_train": _masked_mean(correct, unstable & seen),
        }


def _masked_mean(values, mask):
    m = mask.float()
    denom = m.sum().item()
    if denom == 0:
        return None
    return float((values * m).sum().item() / denom)
