from __future__ import annotations

import logging
from pathlib import Path

import torch

from appfl_sim.datasets.common import (
    BasicTensorDataset,
    clientize_raw_dataset,
    finalize_dataset_outputs,
    make_load_tag,
    resolve_dataset_logger,
    to_namespace,
)
from appfl_sim.datasets.externalparser import _encode_text_features


logger = logging.getLogger(__name__)


def _encode_texts(texts, args):
    seq_len = int(getattr(args, "seq_len", 128))
    if bool(getattr(args, "use_model_tokenizer", False)):
        try:
            from transformers import AutoTokenizer

            model_name = str(getattr(args, "model_name", "")).strip()
            tokenizer_name = model_name if "/" in model_name else "distilbert-base-uncased"
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            encoded = tokenizer(
                list(texts),
                truncation=True,
                padding="max_length",
                max_length=seq_len,
                return_tensors="pt",
            )
            args.need_embedding = False
            args.seq_len = int(encoded["input_ids"].shape[1])
            args.num_embeddings = int(getattr(tokenizer, "vocab_size", 0))
            return torch.stack(
                [encoded["input_ids"].long(), encoded["attention_mask"].long()],
                dim=1,
            )
        except Exception:
            pass

    ids, vocab_size = _encode_text_features(list(texts), args)
    args.need_embedding = True
    args.seq_len = int(ids.shape[1]) if ids.ndim >= 2 else int(seq_len)
    args.num_embeddings = int(vocab_size)
    return ids


def fetch_sklearn_dataset(args):
    args = to_namespace(args)
    active_logger = resolve_dataset_logger(args, logger)
    tag = make_load_tag(str(args.dataset_name), benchmark="SKLEARN")

    try:
        from sklearn.datasets import fetch_20newsgroups
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "scikit-learn dataset backend requested but scikit-learn is not installed."
        ) from e

    dataset_name = str(args.dataset_name).strip().lower().replace("_", "")
    if dataset_name not in {"20newsgroups", "twentynewsgroups"}:
        raise ValueError(
            "dataset.backend=sklearn currently supports only the 20 Newsgroups dataset."
        )

    data_root = Path(str(args.data_dir)).expanduser()
    data_root.mkdir(parents=True, exist_ok=True)
    remove = tuple(getattr(args, "sklearn_remove", ()))
    active_logger.info("[%s] loading 20 Newsgroups with remove=%s.", tag, remove)

    fetch_kwargs = {
        "data_home": str(data_root),
        "remove": remove,
        "shuffle": True,
        "random_state": int(args.seed),
        "download_if_missing": bool(args.download),
    }
    raw_train_data = fetch_20newsgroups(subset="train", **fetch_kwargs)
    raw_test_data = fetch_20newsgroups(subset="test", **fetch_kwargs)

    x_train = _encode_texts(raw_train_data.data, args)
    x_test = _encode_texts(raw_test_data.data, args)
    y_train = torch.as_tensor(raw_train_data.target, dtype=torch.long)
    y_test = torch.as_tensor(raw_test_data.target, dtype=torch.long)

    raw_train = BasicTensorDataset(x_train, y_train, name="[SKLEARN:20NG] TRAIN")
    raw_test = BasicTensorDataset(x_test, y_test, name="[SKLEARN:20NG] TEST")

    active_logger.info("[%s] building federated client splits.", tag)
    client_datasets = clientize_raw_dataset(raw_train, args)
    client_datasets, server_dataset, dataset_meta = finalize_dataset_outputs(
        client_datasets=client_datasets,
        server_dataset=raw_test,
        dataset_meta=args,
        raw_train=raw_train,
    )
    dataset_meta.need_embedding = bool(getattr(args, "need_embedding", False))
    dataset_meta.seq_len = getattr(args, "seq_len", None)
    dataset_meta.num_embeddings = getattr(args, "num_embeddings", None)
    dataset_meta.num_classes = int(len(raw_train_data.target_names))
    active_logger.info(
        "[%s] finished loading (%d clients).", tag, int(dataset_meta.num_clients)
    )
    return client_datasets, server_dataset, dataset_meta
