from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from appfl_sim.datasets.common import package_dataset_outputs, to_namespace


_TEXT_DATASETS = {"shakespeare", "sent140", "reddit"}
_DEFAULT_LEAF_META = {
    "femnist": {"num_classes": 62, "need_embedding": False},
    "shakespeare": {"num_classes": 80, "need_embedding": True, "seq_len": 80, "num_embeddings": 80},
    "sent140": {"num_classes": 2, "need_embedding": True, "seq_len": 25, "num_embeddings": 400001},
    "celeba": {"num_classes": 2, "need_embedding": False},
    "reddit": {"num_classes": 10000, "need_embedding": True, "seq_len": 10, "num_embeddings": 10000},
}


class LeafClientDataset(Dataset):
    def __init__(
        self,
        dataset_key: str,
        split: str,
        user: str,
        records: Dict[str, List[Any]],
        label_to_idx: Dict[str, int],
        seq_len: int | None,
        num_embeddings: int | None,
        image_root: Path | None,
    ):
        self.dataset_key = dataset_key
        self.identifier = f"[LEAF-{dataset_key.upper()}] CLIENT<{user}> ({split})"
        self.x = list(records.get("x", []))
        raw_y = list(records.get("y", []))
        self.targets = torch.tensor(
            [label_to_idx[str(v)] for v in raw_y], dtype=torch.long
        )
        self.seq_len = int(seq_len) if seq_len is not None else None
        self.num_embeddings = int(num_embeddings) if num_embeddings is not None else None
        self.image_root = image_root

    def __len__(self) -> int:
        return len(self.x)

    def __repr__(self) -> str:
        return self.identifier

    def _encode_text(self, value: Any) -> torch.Tensor:
        seq_len = int(self.seq_len or 32)
        vocab = max(8, int(self.num_embeddings or 256))

        if isinstance(value, (list, tuple)) and value and all(
            isinstance(v, (int, np.integer)) for v in value
        ):
            ids = [int(v) % vocab for v in value]
        else:
            if isinstance(value, bytes):
                text = value.decode("utf-8", errors="ignore")
            elif isinstance(value, str):
                text = value
            elif isinstance(value, (list, tuple)):
                text = " ".join(str(v) for v in value)
            else:
                text = str(value)

            tokens = list(text) if self.dataset_key == "shakespeare" else text.split()
            ids = [abs(hash(tok)) % vocab for tok in tokens]

        if len(ids) < seq_len:
            ids += [0] * (seq_len - len(ids))
        return torch.tensor(ids[:seq_len], dtype=torch.long)

    @staticmethod
    def _to_chw_float(arr: np.ndarray) -> torch.Tensor:
        if arr.ndim == 1:
            side = int(round(np.sqrt(arr.size)))
            if side * side == arr.size:
                arr = arr.reshape(side, side)
        if arr.ndim == 2:
            arr = np.expand_dims(arr, axis=0)
        elif arr.ndim == 3 and arr.shape[0] not in {1, 3}:
            arr = np.transpose(arr, (2, 0, 1))

        tensor = torch.from_numpy(arr).float()
        if tensor.max() > 1.0:
            tensor = tensor / 255.0
        return tensor

    def _load_image_from_path(self, rel_or_abs_path: str, rgb: bool) -> torch.Tensor:
        p = Path(rel_or_abs_path)
        if not p.is_absolute() and self.image_root is not None:
            candidate = self.image_root / rel_or_abs_path
            if candidate.exists():
                p = candidate
            else:
                p = self.image_root / Path(rel_or_abs_path).name
        if not p.exists():
            raise FileNotFoundError(f"LEAF image file not found: {p}")

        img = Image.open(p)
        img = img.convert("RGB" if rgb else "L")
        return self._to_chw_float(np.asarray(img))

    def _encode_image_like(self, value: Any, rgb: bool) -> torch.Tensor:
        if isinstance(value, str):
            return self._load_image_from_path(value, rgb=rgb)
        arr = np.asarray(value)
        return self._to_chw_float(arr)

    def __getitem__(self, index: int):
        xi = self.x[index]
        yi = self.targets[index]

        if self.dataset_key in _TEXT_DATASETS:
            x = self._encode_text(xi)
        elif self.dataset_key == "celeba":
            x = self._encode_image_like(xi, rgb=True)
        elif self.dataset_key == "femnist":
            x = self._encode_image_like(xi, rgb=False)
        else:
            arr = np.asarray(xi)
            x = self._to_chw_float(arr) if arr.ndim in {1, 2, 3} else torch.tensor(xi)

        return x, yi


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _merge_leaf_json_dir(folder: Path) -> Dict[str, Any]:
    files = sorted([p for p in folder.glob("*.json") if p.is_file()])
    if not files:
        raise FileNotFoundError(f"No JSON files found in {folder}")

    merged = {"users": [], "num_samples": [], "user_data": {}}
    for fp in files:
        obj = _load_json(fp)
        users = obj.get("users", [])
        user_data = obj.get("user_data", {})
        num_samples = obj.get("num_samples", [])
        for idx, user in enumerate(users):
            if user not in user_data:
                continue
            if user in merged["user_data"]:
                continue
            merged["users"].append(user)
            merged["user_data"][user] = user_data[user]
            if idx < len(num_samples):
                merged["num_samples"].append(int(num_samples[idx]))
            else:
                merged["num_samples"].append(len(user_data[user].get("y", [])))
    return merged


def _sample_users_by_fraction(all_obj: Dict[str, Any], fraction: float, seed: int) -> List[str]:
    users = list(all_obj.get("users", []))
    if fraction >= 1.0 or not users:
        return users

    rng = random.Random(seed)
    rng.shuffle(users)

    total = sum(len(all_obj["user_data"][u].get("y", [])) for u in users)
    target = max(1, int(total * max(0.0, fraction)))

    selected = []
    cum = 0
    for user in users:
        selected.append(user)
        cum += len(all_obj["user_data"][user].get("y", []))
        if cum >= target:
            break
    return selected


def _split_from_all_data(
    dataset_key: str,
    all_obj: Dict[str, Any],
    test_size: float,
    seed: int,
    raw_data_fraction: float,
    min_samples_per_client: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    rng = random.Random(seed)
    users = _sample_users_by_fraction(all_obj, raw_data_fraction, seed)

    train_obj = {"users": [], "num_samples": [], "user_data": {}}
    test_obj = {"users": [], "num_samples": [], "user_data": {}}

    for user in users:
        records = all_obj["user_data"].get(user, {})
        x = list(records.get("x", []))
        y = list(records.get("y", []))

        n = min(len(x), len(y))
        if n < max(2, int(min_samples_per_client)):
            continue

        x = x[:n]
        y = y[:n]
        n_train = max(1, min(int((1.0 - test_size) * n), n - 1))

        if dataset_key == "shakespeare":
            train_idx = list(range(n_train))
            test_idx = list(range(n_train, n))
        else:
            train_idx = sorted(rng.sample(range(n), n_train))
            test_set = set(range(n)) - set(train_idx)
            test_idx = sorted(list(test_set))

        if not train_idx or not test_idx:
            continue

        train_obj["users"].append(user)
        test_obj["users"].append(user)

        train_obj["user_data"][user] = {
            "x": [x[i] for i in train_idx],
            "y": [y[i] for i in train_idx],
        }
        test_obj["user_data"][user] = {
            "x": [x[i] for i in test_idx],
            "y": [y[i] for i in test_idx],
        }
        train_obj["num_samples"].append(len(train_idx))
        test_obj["num_samples"].append(len(test_idx))

    return train_obj, test_obj


def _resolve_leaf_train_test(args, dataset_root: Path, dataset_key: str):
    train_dir = dataset_root / "train"
    test_dir = dataset_root / "test"

    if train_dir.exists() and test_dir.exists():
        return _merge_leaf_json_dir(train_dir), _merge_leaf_json_dir(test_dir)

    all_data_dir = dataset_root / "all_data"
    if not all_data_dir.exists():
        raise FileNotFoundError(
            f"LEAF dataset requires either train/test JSON directories or all_data under {dataset_root}"
        )

    all_obj = _merge_leaf_json_dir(all_data_dir)
    return _split_from_all_data(
        dataset_key=dataset_key,
        all_obj=all_obj,
        test_size=float(getattr(args, "test_size", 0.2)),
        seed=int(getattr(args, "seed", 42)),
        raw_data_fraction=float(getattr(args, "leaf_raw_data_fraction", 1.0)),
        min_samples_per_client=int(getattr(args, "leaf_min_samples_per_client", 2)),
    )


def _resolve_image_root(args, dataset_root: Path, dataset_key: str) -> Path | None:
    user_root = str(getattr(args, "leaf_image_root", "")).strip()
    if user_root:
        p = Path(user_root).expanduser()
        if p.exists():
            return p

    if dataset_key == "celeba":
        candidates = [
            dataset_root / "raw" / "img_align_celeba",
            dataset_root / "img_align_celeba",
            dataset_root / "images",
        ]
        for path in candidates:
            if path.exists():
                return path
    return None


def _build_label_vocab(train_obj: Dict[str, Any], test_obj: Dict[str, Any]) -> Dict[str, int]:
    labels = []
    for obj in [train_obj, test_obj]:
        for user in obj.get("users", []):
            labels.extend(obj["user_data"].get(user, {}).get("y", []))
    unique = sorted({str(v) for v in labels})
    return {label: idx for idx, label in enumerate(unique)}


def fetch_leaf(args):
    """LEAF parser adapted from AAggFF processing flow with compact implementation."""
    args = to_namespace(args)
    dataset_key = str(args.dataset).strip().lower()

    dataset_root = Path(str(args.data_dir)).expanduser() / dataset_key
    if not dataset_root.exists():
        raise FileNotFoundError(f"LEAF dataset root not found: {dataset_root}")

    train_obj, test_obj = _resolve_leaf_train_test(args, dataset_root, dataset_key)
    if not train_obj.get("users"):
        raise ValueError(f"No LEAF users available after processing: {dataset_root}")

    if str(getattr(args, "num_clients", 0)).isdigit() and int(args.num_clients) > 0:
        max_clients = min(int(args.num_clients), len(train_obj["users"]))
        users = train_obj["users"][:max_clients]
    else:
        users = list(train_obj["users"])

    label_to_idx = _build_label_vocab(train_obj, test_obj)
    image_root = _resolve_image_root(args, dataset_root, dataset_key)

    defaults = _DEFAULT_LEAF_META.get(dataset_key, {})
    seq_len = int(getattr(args, "seq_len", defaults.get("seq_len", 32)))
    num_embeddings = int(
        getattr(args, "num_embeddings", defaults.get("num_embeddings", 10000))
    )

    split_map: Dict[int, int] = {}
    client_datasets = []
    for cid, user in enumerate(users):
        tr_records = train_obj["user_data"].get(user, {"x": [], "y": []})
        te_records = test_obj["user_data"].get(user, {"x": [], "y": []})
        tr_ds = LeafClientDataset(
            dataset_key=dataset_key,
            split="train",
            user=str(user),
            records=tr_records,
            label_to_idx=label_to_idx,
            seq_len=seq_len,
            num_embeddings=num_embeddings,
            image_root=image_root,
        )
        te_ds = LeafClientDataset(
            dataset_key=dataset_key,
            split="test",
            user=str(user),
            records=te_records,
            label_to_idx=label_to_idx,
            seq_len=seq_len,
            num_embeddings=num_embeddings,
            image_root=image_root,
        )
        split_map[cid] = len(tr_ds)
        client_datasets.append((tr_ds, te_ds))

    args.num_clients = len(client_datasets)
    args.K = len(client_datasets)
    args.num_classes = (
        len(label_to_idx) if label_to_idx else int(defaults.get("num_classes", 0))
    )

    args.need_embedding = bool(defaults.get("need_embedding", dataset_key in _TEXT_DATASETS))
    if args.need_embedding:
        args.seq_len = seq_len
        args.num_embeddings = num_embeddings
    else:
        args.seq_len = None
        args.num_embeddings = None

    first_train = next((tr for tr, _ in client_datasets if len(tr) > 0), None)
    if first_train is not None:
        x0, _ = first_train[0]
        args.input_shape = tuple(getattr(x0, "shape", (1,)))
    else:
        args.input_shape = (1,)

    if len(args.input_shape) > 1:
        args.in_channels = int(args.input_shape[0])
    else:
        args.in_channels = 1

    return package_dataset_outputs(split_map, client_datasets, None, args)


# backward alias

def fetch_leaf_preprocessed(dataset_name: str, root: str):
    args = to_namespace({"dataset": dataset_name, "data_dir": root})
    return fetch_leaf(args)
