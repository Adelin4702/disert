"""CIFAR-10 / CIFAR-100 loaders.

CIFAR is 32x32; we upsample to `image_size` to reuse ImageNet-pretrained
backbones (EfficientNetV2-S / MobileNetV3-Small). This is a deliberate
transfer-learning shortcut and is disclosed in the thesis: reported accuracies
are NOT directly comparable to papers trained at native 32x32.

The training dataset yields (image, label, index). The index lets the
instability memory track per-sample training dynamics.

Two data sources are supported, auto-detected in this order:
  1. ImageFolder layout  data/<dataset>/{train,test}/<class>/*.png
     (e.g. the fast.ai mirrors, reliable on networks that can't reach
     cs.toronto.edu:  https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz
                      https://s3.amazonaws.com/fast-ai-imageclas/cifar100.tgz)
  2. torchvision pickle format (downloads from cs.toronto.edu if absent).
"""
import os
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

# ImageNet statistics (the pretrained backbones expect these).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

_DATASETS = {
    "cifar10": (datasets.CIFAR10, 10),
    "cifar100": (datasets.CIFAR100, 100),
}


class IndexedDataset(Dataset):
    """Wraps a dataset so __getitem__ also returns the sample index."""

    def __init__(self, base: Dataset):
        self.base = base

    def __getitem__(self, i):
        x, y = self.base[i]
        return x, y, i

    def __len__(self):
        return len(self.base)


def build_transforms(image_size: int):
    train_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    test_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, test_tf


@dataclass
class Loaders:
    train: DataLoader
    test: DataLoader
    num_classes: int
    num_train: int


def _make_base_datasets(dataset, data_dir, train_tf, test_tf):
    """Return (train_base, test_base, num_classes) from whichever source exists.

    Prefers an ImageFolder layout (reliable mirrors); falls back to torchvision's
    downloader.
    """
    img_root = os.path.join(data_dir, dataset)  # e.g. data/cifar10
    train_dir = os.path.join(img_root, "train")
    test_dir = os.path.join(img_root, "test")
    if os.path.isdir(train_dir) and os.path.isdir(test_dir):
        train_base = datasets.ImageFolder(train_dir, transform=train_tf)
        test_base = datasets.ImageFolder(test_dir, transform=test_tf)
        return train_base, test_base, len(train_base.classes)

    ds_cls, num_classes = _DATASETS[dataset]
    train_base = ds_cls(root=data_dir, train=True, download=True, transform=train_tf)
    test_base = ds_cls(root=data_dir, train=False, download=True, transform=test_tf)
    return train_base, test_base, num_classes


def build_loaders(
    dataset: str,
    data_dir: str,
    image_size: int,
    batch_size: int,
    num_workers: int = 2,
    pin_memory: bool = False,
    worker_init_fn=None,
    generator=None,
    train_subset: int = 0,
) -> Loaders:
    dataset = dataset.lower()
    if dataset not in _DATASETS:
        raise ValueError(f"Unknown dataset {dataset!r}; choose from {list(_DATASETS)}")
    train_tf, test_tf = build_transforms(image_size)

    train_base, test_base, num_classes = _make_base_datasets(
        dataset, data_dir, train_tf, test_tf)

    if train_subset and train_subset < len(train_base):
        # Deterministic, evenly-spaced subset for quick dev runs. Evenly spaced
        # (not first-N) so it spans all classes even when the source is stored
        # in class order (ImageFolder), keeping smoke metrics representative.
        n = len(train_base)
        step = max(1, n // train_subset)
        idxs = list(range(0, n, step))[:train_subset]
        train_base = Subset(train_base, idxs)

    train_ds = IndexedDataset(train_base)

    # persistent_workers keeps the decode/resize workers alive across epochs
    # (critical for many short epochs); prefetch_factor buffers batches ahead so
    # the GPU isn't starved. Both require num_workers > 0.
    extra = {}
    if num_workers > 0:
        extra = {"persistent_workers": True, "prefetch_factor": 4}

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
        worker_init_fn=worker_init_fn, generator=generator, drop_last=False,
        **extra,
    )
    test_loader = DataLoader(
        test_base, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        **extra,
    )
    return Loaders(train_loader, test_loader, num_classes, len(train_ds))
