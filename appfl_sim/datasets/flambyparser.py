from __future__ import annotations

import inspect
from typing import Any, Dict, List, Tuple

from appfl_sim.datasets.common import to_namespace


# Keep FLamby dataset scope aligned with AAggFF parser support.
_SUPPORTED_FLAMBY: Dict[str, Dict[str, Any]] = {
    "HEART": {
        "module": "flamby.datasets.fed_heart_disease",
        "class": "FedHeartDisease",
        "max_clients": 4,
        "num_classes": 2,
        "license": "https://archive.ics.uci.edu/dataset/45/heart+disease",
    },
    "ISIC2019": {
        "module": "flamby.datasets.fed_isic2019",
        "class": "FedIsic2019",
        "max_clients": 6,
        "num_classes": 8,
        "license": "https://challenge.isic-archive.com/data/",
    },
    "IXITINY": {
        "module": "flamby.datasets.fed_ixi",
        "class": "FedIXITiny",
        "max_clients": 3,
        "num_classes": 2,
        "license": "https://brain-development.org/ixi-dataset/",
    },
}


def _canonical_flamby_key(dataset_name: str) -> str:
    key = str(dataset_name).strip().upper().replace("-", "").replace("_", "")
    aliases = {
        "HEART": "HEART",
        "HEARTDISEASE": "HEART",
        "ISIC2019": "ISIC2019",
        "IXITINY": "IXITINY",
    }
    return aliases.get(key, "")


def _instantiate_flamby_dataset(ds_class, *, train: bool, center: int | None, pooled: bool):
    sig = inspect.signature(ds_class.__init__)
    kwargs = {}
    if "train" in sig.parameters:
        kwargs["train"] = bool(train)
    if "center" in sig.parameters and center is not None:
        kwargs["center"] = int(center)
    if "pooled" in sig.parameters:
        kwargs["pooled"] = bool(pooled)
    return ds_class(**kwargs)


def fetch_flamby(args):
    """FLamby parser with AAggFF-aligned dataset scope.

    Supported datasets: HEART, ISIC2019, IXITINY.
    Others are rejected because additional data-provider approval is required.
    """
    args = to_namespace(args)
    key = _canonical_flamby_key(str(args.dataset))
    if not key:
        allowed = ", ".join(sorted(_SUPPORTED_FLAMBY.keys()))
        raise PermissionError(
            "Unsupported FLamby dataset for this simulation package. "
            f"Allowed (AAggFF-aligned): {allowed}."
        )

    cfg = _SUPPORTED_FLAMBY[key]
    accepted = bool(getattr(args, "flamby_data_terms_accepted", False))
    if not accepted:
        raise PermissionError(
            f"FLamby dataset '{key}' requires explicit data-term approval. "
            f"Set flamby_data_terms_accepted=true after accepting terms at: {cfg['license']}"
        )

    try:
        module = __import__(cfg["module"], fromlist=[cfg["class"]])
        ds_class = getattr(module, cfg["class"])
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "flamby is not installed or dataset dependencies are missing. "
            "Install required dependencies for FLamby dataset usage."
        ) from e

    max_clients = int(cfg["max_clients"])
    req_clients = int(args.num_clients)
    if req_clients > max_clients:
        raise ValueError(
            f"FLamby dataset '{key}' supports at most {max_clients} clients, got {req_clients}."
        )

    split_map: Dict[int, int] = {}
    client_datasets: List[Tuple[Any, Any]] = []
    for cid in range(req_clients):
        train_ds = _instantiate_flamby_dataset(ds_class, train=True, center=cid, pooled=False)
        test_ds = _instantiate_flamby_dataset(ds_class, train=False, center=cid, pooled=False)
        split_map[cid] = len(train_ds)
        client_datasets.append((train_ds, test_ds))

    # Prefer pooled server eval when supported by dataset API.
    try:
        server_dataset = _instantiate_flamby_dataset(
            ds_class,
            train=False,
            center=None,
            pooled=True,
        )
    except Exception:
        server_dataset = client_datasets[0][1] if client_datasets else None

    args.num_clients = req_clients
    args.K = req_clients
    args.num_classes = int(cfg["num_classes"])
    args.need_embedding = False
    args.seq_len = None
    args.num_embeddings = None

    # Infer tensor shape from first client sample without forcing expensive scans.
    if client_datasets and len(client_datasets[0][0]) > 0:
        sample_x, _ = client_datasets[0][0][0]
        shape = tuple(getattr(sample_x, "shape", ()))
        args.input_shape = shape if shape else (1,)
        if len(args.input_shape) > 1:
            args.in_channels = int(args.input_shape[0])
        else:
            args.in_channels = 1
    else:
        args.input_shape = (1,)
        args.in_channels = 1

    return split_map, client_datasets, server_dataset, args
