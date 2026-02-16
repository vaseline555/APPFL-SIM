from __future__ import annotations

import inspect
from pathlib import Path

from appfl_sim.datasets.common import (
    clientize_raw_dataset,
    finalize_dataset_outputs,
    infer_num_classes,
    to_namespace,
)

try:
    from torchvision import datasets as tv_datasets
    from torchvision import transforms
except Exception:  # pragma: no cover
    tv_datasets = None
    transforms = None


def _resolve_torchvision_dataset(name: str):
    if tv_datasets is None:
        return None
    if hasattr(tv_datasets, name):
        return getattr(tv_datasets, name)
    for candidate in dir(tv_datasets):
        if candidate.lower() == name.lower():
            return getattr(tv_datasets, candidate)
    return None


def _ensure_targets(ds):
    if hasattr(ds, "targets"):
        return
    for attr in ["labels", "_labels", "y"]:
        if hasattr(ds, attr):
            setattr(ds, "targets", getattr(ds, attr))
            return
    if hasattr(ds, "_samples"):
        setattr(ds, "targets", [s[-1] for s in ds._samples])


def _instantiate_dataset(dataset_cls, root: str, split: str, download: bool, transform):
    sig = inspect.signature(dataset_cls.__init__)
    kwargs = {}
    if "root" in sig.parameters:
        kwargs["root"] = root
    if "download" in sig.parameters:
        kwargs["download"] = download
    if "transform" in sig.parameters:
        kwargs["transform"] = transform

    if "train" in sig.parameters:
        kwargs["train"] = split == "train"
        return dataset_cls(**kwargs)

    if "split" in sig.parameters:
        candidates = ["train", "training", "trainval"] if split == "train" else ["test", "valid", "val"]
        for cand in candidates:
            try:
                return dataset_cls(split=cand, **kwargs)
            except Exception:
                continue

    return dataset_cls(**kwargs)


def fetch_torchvision_dataset(args):
    args = to_namespace(args)
    if tv_datasets is None:
        raise RuntimeError("torchvision is not installed.")

    dataset_cls = _resolve_torchvision_dataset(args.dataset)
    if dataset_cls is None:
        raise ValueError(f"Unknown torchvision dataset: {args.dataset}")

    data_root = Path(str(args.data_dir)).expanduser()
    data_root.mkdir(parents=True, exist_ok=True)

    transform = transforms.ToTensor() if transforms is not None else None
    raw_train = _instantiate_dataset(
        dataset_cls,
        str(data_root),
        "train",
        bool(args.download),
        transform,
    )
    raw_test = _instantiate_dataset(
        dataset_cls,
        str(data_root),
        "test",
        bool(args.download),
        transform,
    )

    _ensure_targets(raw_train)
    _ensure_targets(raw_test)

    split_map, client_datasets = clientize_raw_dataset(raw_train, args)
    split_map, client_datasets, server_dataset, args = finalize_dataset_outputs(
        split_map=split_map,
        client_datasets=client_datasets,
        server_dataset=raw_test,
        args=args,
        raw_train=raw_train,
    )
    args.need_embedding = False
    args.seq_len = None
    args.num_embeddings = None
    if hasattr(raw_train, "classes"):
        args.num_classes = max(int(args.num_classes), int(len(raw_train.classes)))
    else:
        args.num_classes = int(infer_num_classes(raw_train))
    return split_map, client_datasets, server_dataset, args
