from __future__ import annotations

import importlib
import inspect
import json
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from appfl_sim.datasets.common import (
    TensorBackedDataset,
    clientize_raw_dataset,
    finalize_dataset_outputs,
    make_load_tag,
    package_dataset_outputs,
    resolve_dataset_logger,
    set_common_metadata,
    to_namespace,
)


logger = logging.getLogger(__name__)


def _to_tensor_backed_dataset(payload: Any, name: str) -> Dataset:
    if isinstance(payload, Dataset):
        return payload

    x_data = None
    y_data = None

    if isinstance(payload, dict):
        for key in ["x", "inputs", "features", "data"]:
            if key in payload:
                x_data = payload[key]
                break
        for key in ["y", "targets", "labels", "label"]:
            if key in payload:
                y_data = payload[key]
                break
    elif isinstance(payload, (tuple, list)) and len(payload) == 2:
        x_data, y_data = payload[0], payload[1]

    if x_data is None or y_data is None:
        raise ValueError(
            "Unable to convert payload into a Dataset. Provide Dataset object or (x, y) tensors/arrays."
        )

    x_tensor = torch.as_tensor(np.asarray(x_data))
    y_tensor = torch.as_tensor(np.asarray(y_data)).long().reshape(-1)

    if x_tensor.shape[0] != y_tensor.shape[0]:
        raise ValueError(
            f"Custom dataset payload shape mismatch: x has {x_tensor.shape[0]} rows, y has {y_tensor.shape[0]} rows"
        )

    if x_tensor.dtype in {torch.int16, torch.int32, torch.int64, torch.uint8}:
        x_tensor = x_tensor.long()
    else:
        x_tensor = x_tensor.float()

    return TensorBackedDataset(x_tensor, y_tensor, name=name)


def _normalize_loader_result(result: Any, args: Any):
    if isinstance(result, tuple) and len(result) == 4:
        split_map, client_datasets, server_dataset, out_args = result
        out_args = set_common_metadata(to_namespace(out_args), client_datasets)
        return package_dataset_outputs(split_map, client_datasets, server_dataset, out_args)

    if isinstance(result, tuple) and len(result) == 3:
        split_map, client_datasets, out_args = result
        out_args = set_common_metadata(to_namespace(out_args), client_datasets)
        return package_dataset_outputs(split_map, client_datasets, None, out_args)

    if isinstance(result, dict):
        if {"split_map", "client_datasets"}.issubset(result.keys()):
            out_args = set_common_metadata(
                to_namespace(result.get("args", args)), result["client_datasets"]
            )
            return package_dataset_outputs(
                result["split_map"],
                result["client_datasets"],
                result.get("server_dataset", None),
                out_args,
            )

    raise ValueError(
        "Custom parser expects (split_map, client_datasets, server_dataset, args), "
        "legacy (split_map, client_datasets, args), or dict with split_map/client_datasets."
    )


def _load_from_callable(args):
    loader_spec = str(getattr(args, "custom_dataset_loader", "")).strip()
    if ":" not in loader_spec:
        raise ValueError(
            "custom_dataset_loader must be in 'package.module:function' format."
        )

    module_name, fn_name = loader_spec.split(":", 1)
    fn = getattr(importlib.import_module(module_name), fn_name)

    kwargs_raw = getattr(
        args,
        "custom_dataset_kwargs",
        "{}",
    )
    if isinstance(kwargs_raw, str):
        kwargs = json.loads(kwargs_raw) if kwargs_raw.strip() else {}
    elif isinstance(kwargs_raw, dict):
        kwargs = dict(kwargs_raw)
    else:
        kwargs = {}

    sig = inspect.signature(fn)
    if "args" in sig.parameters:
        result = fn(args=args, **kwargs)
    elif "cfg" in sig.parameters:
        result = fn(cfg=vars(args), **kwargs)
    else:
        result = fn(**kwargs)

    return _normalize_loader_result(result, args)


def _load_train_test_from_directory(data_dir: Path) -> Tuple[Dataset, Dataset | None]:
    train_candidates = [
        data_dir / "train.pt",
        data_dir / "train.pth",
        data_dir / "train.npz",
    ]
    test_candidates = [
        data_dir / "test.pt",
        data_dir / "test.pth",
        data_dir / "test.npz",
    ]

    train_obj = None
    for path in train_candidates:
        if path.exists():
            if path.suffix == ".npz":
                npz = np.load(path)
                train_obj = {"x": npz["x"], "y": npz["y"]}
            else:
                train_obj = torch.load(path, map_location="cpu")
            break

    if train_obj is None:
        raise FileNotFoundError(
            f"No train artifact found under {data_dir}. Expected one of: {[p.name for p in train_candidates]}"
        )

    test_obj = None
    for path in test_candidates:
        if path.exists():
            if path.suffix == ".npz":
                npz = np.load(path)
                test_obj = {"x": npz["x"], "y": npz["y"]}
            else:
                test_obj = torch.load(path, map_location="cpu")
            break

    train_ds = _to_tensor_backed_dataset(train_obj, "[CUSTOM] TRAIN")
    test_ds = (
        _to_tensor_backed_dataset(test_obj, "[CUSTOM] TEST") if test_obj is not None else None
    )
    return train_ds, test_ds


def _load_from_path(args):
    custom_path = Path(str(getattr(args, "custom_dataset_path", "")).strip()).expanduser()
    if not custom_path.exists():
        raise FileNotFoundError(f"custom_dataset_path does not exist: {custom_path}")

    if custom_path.is_file():
        if custom_path.suffix == ".npz":
            npz = np.load(custom_path)
            payload = {"x": npz["x"], "y": npz["y"]}
        else:
            payload = torch.load(custom_path, map_location="cpu")
        try:
            return _normalize_loader_result(payload, args)
        except Exception:
            train_ds = _to_tensor_backed_dataset(payload, "[CUSTOM] TRAIN")
            split_map, client_datasets = clientize_raw_dataset(train_ds, args)
            return finalize_dataset_outputs(
                split_map=split_map,
                client_datasets=client_datasets,
                server_dataset=None,
                args=args,
                raw_train=train_ds,
            )

    candidate = custom_path / "dataset.pt"
    if candidate.exists():
        payload = torch.load(candidate, map_location="cpu")
        try:
            return _normalize_loader_result(payload, args)
        except Exception:
            pass

    client_payload = custom_path / "client_datasets.pt"
    if client_payload.exists():
        payload = torch.load(client_payload, map_location="cpu")
        if isinstance(payload, dict) and {"split_map", "client_datasets"}.issubset(payload.keys()):
            return _normalize_loader_result(payload, args)
        if isinstance(payload, list):
            split_map = {cid: len(pair[0]) for cid, pair in enumerate(payload)}
            out_args = set_common_metadata(to_namespace(args), payload)
            return package_dataset_outputs(split_map, payload, None, out_args)

    train_ds, test_ds = _load_train_test_from_directory(custom_path)
    split_map, client_datasets = clientize_raw_dataset(train_ds, args)
    return finalize_dataset_outputs(
        split_map=split_map,
        client_datasets=client_datasets,
        server_dataset=test_ds,
        args=args,
        raw_train=train_ds,
    )


def fetch_custom_dataset(args):
    """Custom dataset parser.

    Supports two modes:
    - `custom_dataset_loader=package.module:function` callable contract.
    - `custom_dataset_path=/path/to/data_or_artifact` local artifact contract.
    """
    args = to_namespace(args)
    active_logger = resolve_dataset_logger(args, logger)
    tag = make_load_tag(str(getattr(args, "dataset", "custom")), benchmark="CUSTOM")

    loader_spec = str(getattr(args, "custom_dataset_loader", "")).strip()
    dataset_path = str(getattr(args, "custom_dataset_path", "")).strip()

    if loader_spec:
        active_logger.info("[%s] loading via custom callable.", tag)
        out = _load_from_callable(args)
        active_logger.info("[%s] finished loading.", tag)
        return out
    if dataset_path:
        active_logger.info("[%s] loading from custom path.", tag)
        out = _load_from_path(args)
        active_logger.info("[%s] finished loading.", tag)
        return out

    raise ValueError(
        "For dataset_loader=custom, set either custom_dataset_loader or custom_dataset_path."
    )
