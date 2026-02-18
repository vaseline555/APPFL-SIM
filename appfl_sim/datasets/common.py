from __future__ import annotations

import random
import logging
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Subset


class _DatasetLoggerAdapter:
    """Adapter for APPFL custom loggers to stdlib-like interface."""

    def __init__(self, target):
        self._target = target

    @staticmethod
    def _fmt(msg: str, *args) -> str:
        if args:
            try:
                return str(msg) % args
            except Exception:
                return f"{msg} {' '.join(str(a) for a in args)}"
        return str(msg)

    def info(self, msg, *args, **kwargs):
        self._target.info(self._fmt(msg, *args))

    def warning(self, msg, *args, **kwargs):
        if hasattr(self._target, "warning"):
            self._target.warning(self._fmt(msg, *args))
        else:
            self._target.info(self._fmt(msg, *args))

    def error(self, msg, *args, **kwargs):
        if hasattr(self._target, "error"):
            self._target.error(self._fmt(msg, *args))
        else:
            self._target.info(self._fmt(msg, *args))


def resolve_dataset_logger(args: Any, default_logger: logging.Logger):
    candidate = getattr(args, "logger", None)
    if candidate is None:
        return default_logger
    if isinstance(candidate, logging.Logger):
        return candidate
    return _DatasetLoggerAdapter(candidate)


def make_load_tag(dataset_name: str, benchmark: str | None = None) -> str:
    ds = str(dataset_name).strip().upper()
    bench = str(benchmark or "").strip().upper()
    return f"{bench}-{ds}" if bench else ds


class TensorBackedDataset(Dataset):
    def __init__(self, inputs: torch.Tensor, targets: torch.Tensor, name: str = "dataset"):
        self.inputs = inputs
        self.targets = targets.long()
        self.name = name

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int):
        return self.inputs[index], self.targets[index]

    def __repr__(self) -> str:
        return self.name


def to_namespace(args: Any) -> SimpleNamespace:
    if isinstance(args, SimpleNamespace):
        ns = SimpleNamespace(**vars(args))
    elif isinstance(args, dict):
        ns = SimpleNamespace(**args)
    else:
        ns = SimpleNamespace(**vars(args))

    defaults = {
        "num_clients": 20,
        "K": None,
        "seed": 42,
        "data_dir": "./data",
        "test_size": 0.2,
        "split_type": "iid",
        "dirichlet_alpha": 0.3,
        "min_classes": 2,
        "unbalanced_keep_min": 0.5,
        "download": True,
        "dataset_loader": "auto",
        "custom_dataset_loader": "",
        "custom_dataset_kwargs": "{}",
        "custom_dataset_path": "",
        "external_source": "",
        "external_dataset_name": "",
        "external_dataset_config_name": "",
        "external_train_split": "train",
        "external_test_split": "test",
        "external_feature_key": "",
        "external_label_key": "",
        "seq_len": 128,
        "num_embeddings": 10000,
        "use_model_tokenizer": False,
        "use_pt_model": False,
        "model_name": "SimpleCNN",
        "audio_num_frames": 16000,
        "flamby_data_terms_accepted": False,
        "leaf_raw_data_fraction": 1.0,
        "leaf_min_samples_per_client": 2,
        "leaf_image_root": "",
        "infer_num_clients": False,
        "client_subsample_num": 0,
        "client_subsample_ratio": 1.0,
        "client_subsample_mode": "random",
        "client_subsample_seed": None,
        "in_channels": None,
    }
    for k, v in defaults.items():
        if not hasattr(ns, k):
            setattr(ns, k, v)
    if ns.K is None:
        ns.K = int(ns.num_clients)
    return ns


def _prefixed_arg(
    args: SimpleNamespace,
    prefix: str,
    key: str,
    default: Any,
) -> Any:
    if prefix:
        pref_key = f"{prefix}_{key}"
        if hasattr(args, pref_key):
            return getattr(args, pref_key)
    return getattr(args, key, default)


def _safe_int(value: Any, default: int) -> int:
    try:
        if value is None:
            return int(default)
        return int(value)
    except Exception:
        return int(default)


def _safe_float(value: Any, default: float) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _safe_bool(value: Any, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "y", "on"}:
            return True
        if text in {"0", "false", "no", "n", "off"}:
            return False
        return bool(default)
    return bool(value)


def resolve_fixed_pool_clients(
    available_clients: List[Any],
    args: SimpleNamespace,
    prefix: str = "",
) -> List[Any]:
    """Resolve client subset for fixed-pool datasets (LEAF/FLamby/TFF).

    Rules:
    - If `infer_num_clients=true` (global or `<prefix>_infer_num_clients`) or
      `num_clients<=0`, start from full available pool.
    - Otherwise cap to `num_clients`.
    - Then apply optional subsampling controls:
      - `<prefix>_client_subsample_num` / `client_subsample_num`
      - `<prefix>_client_subsample_ratio` / `client_subsample_ratio`
      - `<prefix>_client_subsample_mode` / `client_subsample_mode` (`random|first|last`)
      - `<prefix>_client_subsample_seed` / `client_subsample_seed`
    """
    pool = list(available_clients)
    if not pool:
        return []

    infer_clients = _safe_bool(
        _prefixed_arg(args, prefix, "infer_num_clients", False), False
    )
    requested_num = _safe_int(getattr(args, "num_clients", 0), 0)
    if infer_clients or requested_num <= 0:
        requested_num = len(pool)
    requested_num = max(1, min(requested_num, len(pool)))
    base = pool[:requested_num]

    subsample_num = _safe_int(
        _prefixed_arg(args, prefix, "client_subsample_num", 0),
        0,
    )
    subsample_ratio = _safe_float(
        _prefixed_arg(args, prefix, "client_subsample_ratio", 1.0),
        1.0,
    )
    subsample_mode = str(
        _prefixed_arg(args, prefix, "client_subsample_mode", "random")
    ).strip().lower()
    subsample_seed = _safe_int(
        _prefixed_arg(args, prefix, "client_subsample_seed", getattr(args, "seed", 42)),
        _safe_int(getattr(args, "seed", 42), 42),
    )

    target = len(base)
    if subsample_num > 0:
        target = min(target, subsample_num)
    elif 0.0 < subsample_ratio < 1.0:
        target = max(1, int(round(len(base) * subsample_ratio)))

    if target >= len(base):
        return base

    if subsample_mode in {"first", "head"}:
        return base[:target]
    if subsample_mode in {"last", "tail"}:
        return base[-target:]

    rng = random.Random(subsample_seed)
    chosen_idx = set(rng.sample(range(len(base)), target))
    # Preserve original order for reproducible client-id mapping.
    return [item for idx, item in enumerate(base) if idx in chosen_idx]


def infer_input_shape(dataset: Dataset) -> Tuple[int, ...]:
    x, _ = dataset[0]
    if not torch.is_tensor(x):
        x = torch.as_tensor(np.asarray(x))
    return tuple(x.shape)


def extract_targets(dataset) -> np.ndarray:
    if hasattr(dataset, "targets"):
        targets = dataset.targets
        if torch.is_tensor(targets):
            return targets.detach().cpu().numpy()
        return np.asarray(targets)

    if hasattr(dataset, "tensors") and len(dataset.tensors) >= 2:
        return dataset.tensors[1].detach().cpu().numpy()

    if hasattr(dataset, "dataset") and hasattr(dataset, "indices"):
        parent_targets = extract_targets(dataset.dataset)
        return parent_targets[np.asarray(dataset.indices)]

    labels = []
    for i in range(len(dataset)):
        sample = dataset[i]
        if not isinstance(sample, (tuple, list)) or len(sample) < 2:
            raise ValueError("Unable to extract targets from dataset.")
        y = sample[1]
        if torch.is_tensor(y):
            if y.ndim == 0:
                labels.append(int(y.item()))
            else:
                labels.append(int(y.reshape(-1)[0].item()))
        elif isinstance(y, np.ndarray):
            labels.append(int(np.asarray(y).reshape(-1)[0]))
        else:
            labels.append(int(y))
    return np.asarray(labels, dtype=np.int64)


def infer_num_classes(dataset) -> int:
    targets = extract_targets(dataset)
    return int(np.unique(targets).size) if targets.size > 0 else 0


def concat_targets(datasets: Iterable[Dataset]) -> np.ndarray:
    chunks = []
    for ds in datasets:
        if ds is None or len(ds) == 0:
            continue
        chunks.append(extract_targets(ds))
    if not chunks:
        return np.asarray([], dtype=np.int64)
    return np.concatenate(chunks).astype(np.int64)


def _first_nonempty_train_dataset(client_datasets):
    for train_ds, _ in client_datasets:
        if len(train_ds) > 0:
            return train_ds
    return None


def package_dataset_outputs(split_map, client_datasets, server_dataset, args: SimpleNamespace):
    return split_map, client_datasets, server_dataset, args


def finalize_dataset_outputs(
    split_map,
    client_datasets,
    server_dataset,
    args: Any,
    raw_train: Dataset | None = None,
):
    args_ns = to_namespace(args)
    args_ns = set_common_metadata(
        args_ns,
        client_datasets,
        raw_train=raw_train,
    )
    return package_dataset_outputs(split_map, client_datasets, server_dataset, args_ns)


def _iid_split(num_samples: int, num_clients: int, rng: np.random.Generator) -> dict[int, np.ndarray]:
    perm = rng.permutation(num_samples)
    chunks = np.array_split(perm, num_clients)
    return {cid: chunk.astype(np.int64) for cid, chunk in enumerate(chunks)}


def _unbalanced_split(
    num_samples: int,
    num_clients: int,
    rng: np.random.Generator,
    keep_min: float,
) -> dict[int, np.ndarray]:
    base = _iid_split(num_samples, num_clients, rng)
    out: dict[int, np.ndarray] = {}
    for cid, indices in base.items():
        if len(indices) <= 1:
            out[cid] = indices
            continue
        keep_ratio = rng.uniform(keep_min, 1.0)
        keep_count = max(1, int(len(indices) * keep_ratio))
        out[cid] = indices[:keep_count]
    return out


def _pathological_split(
    labels: np.ndarray,
    num_clients: int,
    min_classes: int,
    rng: np.random.Generator,
) -> dict[int, np.ndarray]:
    indices = np.arange(len(labels))
    sorted_indices = indices[np.argsort(labels)]

    num_shards = max(num_clients * max(min_classes, 1), num_clients)
    shards = np.array_split(sorted_indices, num_shards)
    shard_ids = rng.permutation(num_shards)

    out = {cid: [] for cid in range(num_clients)}
    ptr = 0

    for cid in range(num_clients):
        for _ in range(max(min_classes, 1)):
            if ptr >= len(shard_ids):
                ptr = 0
            sid = shard_ids[ptr]
            ptr += 1
            out[cid].append(shards[sid])

    while ptr < len(shard_ids):
        sid = shard_ids[ptr]
        cid = (ptr - num_clients * max(min_classes, 1)) % num_clients
        out[cid].append(shards[sid])
        ptr += 1

    return {
        cid: np.concatenate(parts).astype(np.int64) if parts else np.array([], dtype=np.int64)
        for cid, parts in out.items()
    }


def _dirichlet_split(
    labels: np.ndarray,
    num_clients: int,
    alpha: float,
    min_size: int,
    rng: np.random.Generator,
    max_retry: int = 20,
) -> dict[int, np.ndarray]:
    classes = np.unique(labels)
    alpha = max(alpha, 1e-3)

    for _ in range(max_retry):
        splits = [[] for _ in range(num_clients)]
        for cls in classes:
            cls_idx = np.where(labels == cls)[0]
            rng.shuffle(cls_idx)
            proportions = rng.dirichlet(np.repeat(alpha, num_clients))
            cut_points = (np.cumsum(proportions)[:-1] * len(cls_idx)).astype(int)
            cls_splits = np.split(cls_idx, cut_points)
            for cid, part in enumerate(cls_splits):
                splits[cid].append(part)

        result = {
            cid: np.concatenate(parts).astype(np.int64) if parts else np.array([], dtype=np.int64)
            for cid, parts in enumerate(splits)
        }
        min_client_size = min(len(v) for v in result.values())
        if min_client_size >= min_size:
            return result

    return _iid_split(len(labels), num_clients, rng)


def simulate_split(
    labels: np.ndarray,
    num_clients: int,
    split_type: str,
    seed: int,
    min_classes: int = 2,
    dirichlet_alpha: float = 0.3,
    unbalanced_keep_min: float = 0.5,
    pre_split_map: dict[int, np.ndarray] | None = None,
) -> dict[int, np.ndarray]:
    rng = np.random.default_rng(seed)
    split_type = split_type.lower()

    if split_type == "iid":
        return _iid_split(len(labels), num_clients, rng)
    if split_type == "unbalanced":
        return _unbalanced_split(len(labels), num_clients, rng, keep_min=unbalanced_keep_min)
    if split_type in {"patho", "pathological"}:
        return _pathological_split(labels, num_clients, min_classes=min_classes, rng=rng)
    if split_type in {"diri", "dirichlet"}:
        return _dirichlet_split(
            labels,
            num_clients,
            alpha=dirichlet_alpha,
            min_size=2,
            rng=rng,
        )
    if split_type == "pre":
        if pre_split_map is None:
            raise ValueError("split_type='pre' requires pre_split_map.")
        return {int(k): np.asarray(v, dtype=np.int64) for k, v in pre_split_map.items()}

    raise ValueError(f"Unsupported split_type: {split_type}")


def split_subset_for_client(
    raw_train: Dataset,
    sample_indices: np.ndarray,
    client_id: int,
    test_size: float,
    seed: int,
):
    sample_indices = np.asarray(sample_indices, dtype=np.int64)
    rng = np.random.default_rng(seed + client_id)
    sample_indices = rng.permutation(sample_indices)

    n = len(sample_indices)
    n_test = int(n * test_size)
    if test_size > 0 and n > 1:
        n_test = max(1, min(n_test, n - 1))
    else:
        n_test = 0

    test_idx = sample_indices[:n_test]
    train_idx = sample_indices[n_test:]

    train_subset = Subset(raw_train, train_idx.tolist())
    test_subset = Subset(raw_train, test_idx.tolist()) if n_test > 0 else Subset(raw_train, [])

    train_targets = extract_targets(train_subset) if len(train_subset) > 0 else np.array([], dtype=np.int64)
    test_targets = extract_targets(test_subset) if len(test_subset) > 0 else np.array([], dtype=np.int64)
    train_subset.targets = torch.from_numpy(train_targets).long()
    test_subset.targets = torch.from_numpy(test_targets).long()
    return train_subset, test_subset


def clientize_raw_dataset(raw_train: Dataset, args: SimpleNamespace):
    targets = extract_targets(raw_train)
    split_map = simulate_split(
        labels=targets,
        num_clients=int(args.num_clients),
        split_type=str(args.split_type),
        seed=int(args.seed),
        min_classes=int(args.min_classes),
        dirichlet_alpha=float(args.dirichlet_alpha),
        unbalanced_keep_min=float(args.unbalanced_keep_min),
    )

    client_datasets = []
    for cid in range(int(args.num_clients)):
        train_ds, test_ds = split_subset_for_client(
            raw_train=raw_train,
            sample_indices=split_map[cid],
            client_id=cid,
            test_size=float(args.test_size),
            seed=int(args.seed),
        )
        client_datasets.append((train_ds, test_ds))

    return split_map, client_datasets


def set_common_metadata(
    args: SimpleNamespace,
    client_datasets,
    raw_train: Dataset | None = None,
):
    args.num_clients = len(client_datasets)
    args.K = len(client_datasets)

    shape_source = raw_train if raw_train is not None and len(raw_train) > 0 else _first_nonempty_train_dataset(client_datasets)
    args.input_shape = infer_input_shape(shape_source) if shape_source is not None else (1,)

    if raw_train is not None and len(raw_train) > 0:
        args.num_classes = infer_num_classes(raw_train)
    else:
        train_targets = concat_targets(train_ds for train_ds, _ in client_datasets)
        args.num_classes = int(np.unique(train_targets).size) if train_targets.size > 0 else 0

    if getattr(args, "in_channels", None) is None:
        if len(args.input_shape) == 1:
            args.in_channels = 1
        elif len(args.input_shape) > 1:
            args.in_channels = int(args.input_shape[0])
    return args
