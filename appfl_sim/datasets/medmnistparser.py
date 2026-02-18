from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from appfl_sim.datasets.common import (
    clientize_raw_dataset,
    finalize_dataset_outputs,
    infer_num_classes,
    make_load_tag,
    resolve_dataset_logger,
    to_namespace,
)

try:
    from torchvision import transforms
except Exception:  # pragma: no cover
    transforms = None


logger = logging.getLogger(__name__)


class MedMNISTWrapper(Dataset):
    def __init__(self, base_dataset, name: str):
        self.base_dataset = base_dataset
        labels = np.asarray(base_dataset.labels).reshape(-1)
        self.targets = torch.from_numpy(labels).long()
        self.name = name

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index: int):
        x, y = self.base_dataset[index]
        y = torch.as_tensor(y).long().reshape(-1)[0]
        return x, y

    def __repr__(self):
        return self.name


def fetch_medmnist_dataset(args):
    args = to_namespace(args)
    active_logger = resolve_dataset_logger(args, logger)
    tag = make_load_tag(str(args.dataset), benchmark="MEDMNIST")
    active_logger.info("[%s] resolving dataset metadata.", tag)
    try:
        import medmnist
        from medmnist import INFO
    except Exception as e:  # pragma: no cover
        raise RuntimeError("medmnist is not installed.") from e

    dataset_key = str(args.dataset).lower()
    alias = {k.lower(): k for k in INFO}
    if dataset_key not in alias:
        raise ValueError(f"Unknown MedMNIST dataset: {args.dataset}")

    canonical = alias[dataset_key]
    info = INFO[canonical]
    data_class = getattr(medmnist, info["python_class"])
    transform = transforms.ToTensor() if transforms is not None else None

    data_root = Path(str(args.data_dir)).expanduser()
    data_root.mkdir(parents=True, exist_ok=True)
    if bool(args.download):
        active_logger.info("[%s] downloading (if needed).", tag)

    train_base = data_class(
        split="train",
        root=str(data_root),
        download=bool(args.download),
        transform=transform,
    )
    test_base = data_class(
        split="test",
        root=str(data_root),
        download=bool(args.download),
        transform=transform,
    )

    raw_train = MedMNISTWrapper(train_base, name=f"[{canonical}] TRAIN")
    raw_test = MedMNISTWrapper(test_base, name=f"[{canonical}] TEST")
    active_logger.info("[%s] building federated client splits.", tag)

    split_map, client_datasets = clientize_raw_dataset(raw_train, args)
    split_map, client_datasets, server_dataset, args = finalize_dataset_outputs(
        split_map=split_map,
        client_datasets=client_datasets,
        server_dataset=raw_test,
        args=args,
        raw_train=raw_train,
    )
    args.num_classes = max(int(args.num_classes), int(info.get("n_classes", infer_num_classes(raw_train))))
    args.need_embedding = False
    args.seq_len = None
    args.num_embeddings = None
    active_logger.info("[%s] finished loading (%d clients).", tag, int(args.num_clients))
    return split_map, client_datasets, server_dataset, args
