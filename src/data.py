"""CIFAR-10 / CIFAR-100 data loading — in-memory + GPU-side transforms.

Why this design: CIFAR is 50k tiny 32x32 images. Reading 50k PNG files from a
(container) filesystem every epoch and upscaling 32->160 on the CPU is the real
bottleneck — it dominates wall-clock and forces many dataloader workers, which
then trip fork/OpenMP/shm problems. Instead we:

  1. Decode the whole dataset into a single uint8 tensor [N, 3, 32, 32] ONCE and
     cache it to data/_cache_<dataset>_<split>.pt (reloads are instant).
  2. Serve raw uint8 32x32 tensors (no per-file I/O, no CPU decode) so
     num_workers=0 is optimal — sidestepping all worker/OMP/shm issues.
  3. Do resize -> normalize -> (train) horizontal-flip on the GPU, per batch.

CIFAR is upsampled to `image_size` to reuse ImageNet-pretrained backbones. This
is a deliberate transfer-learning shortcut, disclosed in the thesis: accuracies
are NOT comparable to native-32px CIFAR papers.

The training dataset yields (uint8_image, label, index); the index lets the
instability memory track per-sample training dynamics.

Data source, auto-detected:
  1. ImageFolder layout  data/<dataset>/{train,test}/<class>/*.png
     (fast.ai mirrors: https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz)
  2. torchvision pickle format (downloads from cs.toronto.edu if absent).
"""
import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from torchvision import datasets

# ImageNet statistics (the pretrained backbones expect these).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

_NUM_CLASSES = {"cifar10": 10, "cifar100": 100}
_TV = {"cifar10": datasets.CIFAR10, "cifar100": datasets.CIFAR100}


# --------------------------------------------------------------------------- #
# One-time decode + cache to a single uint8 tensor
# --------------------------------------------------------------------------- #
def _decode_split(dataset, data_dir, train):
    """Return (images_uint8 [N,3,32,32], labels_long [N], classes) for a split."""
    img_root = os.path.join(data_dir, dataset)
    split_dir = os.path.join(img_root, "train" if train else "test")

    if os.path.isdir(split_dir):
        base = datasets.ImageFolder(split_dir)          # (PIL, label), no transform
        classes = base.classes
        n = len(base)
        images = torch.empty(n, 3, 32, 32, dtype=torch.uint8)
        labels = torch.empty(n, dtype=torch.long)
        for i in range(n):
            img, lab = base[i]
            arr = np.array(img)                          # H,W,3 uint8 (writable copy)
            images[i] = torch.from_numpy(arr).permute(2, 0, 1)
            labels[i] = lab
            if (i + 1) % 10000 == 0:
                print(f"  [cache] decoded {i + 1}/{n} {'train' if train else 'test'} images")
        return images, labels, classes

    # torchvision fallback (downloads if needed)
    ds = _TV[dataset](root=data_dir, train=train, download=True)
    data = ds.data                                       # numpy [N,32,32,3] uint8
    images = torch.from_numpy(data).permute(0, 3, 1, 2).contiguous().to(torch.uint8)
    labels = torch.tensor(ds.targets, dtype=torch.long)
    classes = list(getattr(ds, "classes", range(_NUM_CLASSES[dataset])))
    return images, labels, classes


def _load_cached(dataset, data_dir, train):
    split = "train" if train else "test"
    cache = os.path.join(data_dir, f"_cache_{dataset}_{split}.pt")
    if os.path.exists(cache):
        blob = torch.load(cache, weights_only=False)
        return blob["images"], blob["labels"], blob["classes"]
    print(f"[cache] building {split} tensor cache (one-time)...")
    images, labels, classes = _decode_split(dataset, data_dir, train)
    torch.save({"images": images, "labels": labels, "classes": classes}, cache)
    print(f"[cache] wrote {cache}  ({images.shape[0]} images)")
    return images, labels, classes


# --------------------------------------------------------------------------- #
# Dataset + GPU transform
# --------------------------------------------------------------------------- #
class InMemoryCIFAR(Dataset):
    """Serves raw uint8 [3,32,32] images from RAM plus (label, index)."""

    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, i):
        return self.images[i], int(self.labels[i]), i


class GPUBatchTransform:
    """Resize -> normalize (+ train-time horizontal flip), all on the GPU.

    Input:  uint8 batch [B,3,32,32] already on `device`.
    Output: float batch [B,3,image_size,image_size], ImageNet-normalized.
    """

    def __init__(self, image_size, train, device):
        self.image_size = image_size
        self.train = train
        self.mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def __call__(self, x):
        x = x.float().div_(255.0)
        if self.train:
            flip = torch.rand(x.size(0), device=x.device) < 0.5
            if flip.any():
                x[flip] = torch.flip(x[flip], dims=[-1])
        x = F.interpolate(x, size=(self.image_size, self.image_size),
                          mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        return x


@dataclass
class Loaders:
    train: DataLoader
    test: DataLoader
    num_classes: int
    num_train: int
    train_tf: GPUBatchTransform
    eval_tf: GPUBatchTransform


def build_loaders(
    dataset: str,
    data_dir: str,
    image_size: int,
    batch_size: int,
    device,
    num_workers: int = 0,
    pin_memory: bool = False,
    generator=None,
    train_subset: int = 0,
) -> Loaders:
    dataset = dataset.lower()
    if dataset not in _NUM_CLASSES:
        raise ValueError(f"Unknown dataset {dataset!r}; choose from {list(_NUM_CLASSES)}")

    tr_imgs, tr_labels, classes = _load_cached(dataset, data_dir, train=True)
    te_imgs, te_labels, _ = _load_cached(dataset, data_dir, train=False)
    num_classes = len(classes) if classes else _NUM_CLASSES[dataset]

    if train_subset and train_subset < tr_imgs.shape[0]:
        # Evenly-spaced subset spans all classes even when stored in class order.
        n = tr_imgs.shape[0]
        step = max(1, n // train_subset)
        idxs = torch.arange(0, n, step)[:train_subset]
        tr_imgs, tr_labels = tr_imgs[idxs], tr_labels[idxs]

    train_ds = InMemoryCIFAR(tr_imgs, tr_labels)
    test_ds = InMemoryCIFAR(te_imgs, te_labels)

    # num_workers=0 is ideal here: __getitem__ just indexes a RAM tensor, so
    # there is nothing to parallelize and no worker processes to fork.
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
        generator=generator, drop_last=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    train_tf = GPUBatchTransform(image_size, train=True, device=device)
    eval_tf = GPUBatchTransform(image_size, train=False, device=device)
    return Loaders(train_loader, test_loader, num_classes, len(train_ds), train_tf, eval_tf)