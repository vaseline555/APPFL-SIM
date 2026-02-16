from __future__ import annotations

from typing import Any, Dict

import torch

from appfl_sim.datasets import (
    fetch_custom_dataset,
    fetch_external_dataset,
    fetch_flamby,
    fetch_leaf,
    fetch_medmnist_dataset,
    fetch_tff_dataset,
    fetch_torchaudio_dataset,
    fetch_torchtext_dataset,
    fetch_torchvision_dataset,
)
from appfl_sim.datasets.common import extract_targets, simulate_split, split_subset_for_client, to_namespace


LEAF_DATASETS = {"FEMNIST", "SHAKESPEARE", "SENT140", "CELEBA", "REDDIT"}
FLAMBY_DATASET_KEYS = {"HEART", "ISIC2019", "IXITINY"}


def _has_medmnist_dataset(name: str) -> bool:
    try:
        import medmnist

        return str(name).lower() in {k.lower() for k in medmnist.INFO.keys()}
    except Exception:
        return False


def _has_torchtext_dataset(name: str) -> bool:
    try:
        import torchtext

        return hasattr(torchtext.datasets, str(name))
    except Exception:
        return False


def _has_torchaudio_dataset(name: str) -> bool:
    try:
        import torchaudio

        if hasattr(torchaudio.datasets, str(name)):
            return True
        lowered = str(name).lower()
        return any(c.lower() == lowered for c in dir(torchaudio.datasets))
    except Exception:
        return False


def load_dataset(args: Any):
    """Unified dataset loader API.

    Return contract:
      split_map, client_datasets, server_dataset, args

    `dataset_loader` modes:
    - `auto`: infer parser by dataset name/library.
    - `custom`: local path or callable parser (`custom_dataset_path` / `custom_dataset_loader`).
    - `external`: external source parser (`hf` or `timm`).
    - built-ins: `torchvision`, `torchtext`, `torchaudio`, `medmnist`, `flamby`, `leaf`, `tff`.
    """
    args = to_namespace(args)
    mode = str(getattr(args, "dataset_loader", "auto")).strip().lower()
    dataset_name = str(args.dataset)
    dataset_upper = dataset_name.upper()
    dataset_lower = dataset_name.lower()

    if mode in {"custom"}:
        return fetch_custom_dataset(args)
    if mode in {"external", "hf", "timm"}:
        if mode in {"hf", "timm"}:
            args.external_source = mode
        return fetch_external_dataset(args)
    if mode == "torchvision":
        return fetch_torchvision_dataset(args)
    if mode == "torchtext":
        return fetch_torchtext_dataset(args)
    if mode == "torchaudio":
        return fetch_torchaudio_dataset(args)
    if mode == "medmnist":
        return fetch_medmnist_dataset(args)
    if mode == "flamby":
        return fetch_flamby(args)
    if mode == "leaf":
        return fetch_leaf(args)
    if mode == "tff":
        return fetch_tff_dataset(args)

    # auto mode
    if dataset_lower.startswith("hf:") or dataset_lower.startswith("timm:"):
        return fetch_external_dataset(args)
    if dataset_upper in LEAF_DATASETS:
        return fetch_leaf(args)
    if dataset_upper in FLAMBY_DATASET_KEYS:
        return fetch_flamby(args)
    if dataset_lower.startswith("tff:"):
        return fetch_tff_dataset(args)

    if _has_medmnist_dataset(dataset_name):
        return fetch_medmnist_dataset(args)
    if _has_torchtext_dataset(dataset_name):
        return fetch_torchtext_dataset(args)
    if _has_torchaudio_dataset(dataset_name):
        return fetch_torchaudio_dataset(args)

    return fetch_torchvision_dataset(args)


# Backward-compatible wrappers


def load_global_dataset(cfg: Dict[str, Any]):
    _, client_datasets, server_dataset, args = load_dataset(cfg)
    pooled_train = torch.utils.data.ConcatDataset([tr for tr, _ in client_datasets])
    return pooled_train, server_dataset, args.num_classes, args.input_shape


def make_client_splits(train_dataset, cfg: Dict[str, Any]) -> dict[int, Any]:
    args = to_namespace(cfg)

    targets = extract_targets(train_dataset)
    return simulate_split(
        labels=targets,
        num_clients=int(args.num_clients),
        split_type=str(args.split_type),
        seed=int(args.seed),
        min_classes=int(args.min_classes),
        dirichlet_alpha=float(args.dirichlet_alpha),
        unbalanced_keep_min=float(args.unbalanced_keep_min),
    )


def build_local_client_datasets(
    train_dataset,
    split_map: dict[int, Any],
    local_client_ids,
    local_val_ratio: float,
    seed: int,
):
    train_sets = {}
    val_sets = {}
    for cid in local_client_ids:
        tr_ds, te_ds = split_subset_for_client(
            raw_train=train_dataset,
            sample_indices=split_map[int(cid)],
            client_id=int(cid),
            test_size=float(local_val_ratio),
            seed=int(seed),
        )
        train_sets[int(cid)] = tr_ds
        val_sets[int(cid)] = te_ds
    return train_sets, val_sets
