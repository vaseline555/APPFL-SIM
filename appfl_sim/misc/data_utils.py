from __future__ import annotations
import ast
import random
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import ConcatDataset, random_split
from appfl_sim.logger import ServerAgentFileLogger
from appfl_sim.misc.config_utils import _cfg_get, _cfg_set


def _parse_holdout_dataset_ratio(config: DictConfig) -> Optional[List[float]]:
    raw = _cfg_get(config, "eval.configs.dataset_ratio", None)
    if raw is None:
        return None
    if isinstance(raw, str):
        text = raw.strip()
        if text == "":
            return None
        try:
            parsed = ast.literal_eval(text)
        except Exception as exc:
            raise ValueError(
                "eval.configs.dataset_ratio must be a list-like string, e.g. '[80,20]' or '[0.8,0.1,0.1]'"
            ) from exc
    else:
        parsed = raw

    if isinstance(parsed, (int, float)):
        raise ValueError("eval.configs.dataset_ratio must contain 1, 2, or 3 values.")
    ratios = [float(x) for x in parsed]
    if len(ratios) not in {1, 2, 3}:
        raise ValueError("eval.configs.dataset_ratio must have length 1, 2, or 3.")
    if any(x <= 0 for x in ratios):
        raise ValueError("eval.configs.dataset_ratio values must be positive.")
    total = float(sum(ratios))
    if np.isclose(total, 100.0, atol=1e-6):
        ratios = [x / 100.0 for x in ratios]
    elif not np.isclose(total, 1.0, atol=1e-6):
        raise ValueError(
            "eval.configs.dataset_ratio must sum to 1.0 or 100.0, e.g. [0.8,0.2] or [80,20]."
        )
    if len(ratios) == 1 and not np.isclose(ratios[0], 1.0, atol=1e-6):
        raise ValueError(
            "eval.configs.dataset_ratio with a single value only accepts [1.0] or [100]."
        )
    return ratios

def _validate_bandit_dataset_ratio(config: DictConfig) -> None:
    algorithm = str(_cfg_get(config, "algorithm.algorithm", "fedavg")).strip().lower()
    if algorithm not in {"swucb", "swts"}:
        return
    ratios = _parse_holdout_dataset_ratio(config)
    if ratios is None:
        raise ValueError(
            "For algorithm in {swucb, swts}, `eval.configs.dataset_ratio` is required "
            "and must include validation split, e.g. [80,10,10]."
        )
    if len(ratios) < 3:
        raise ValueError(
            "For algorithm in {swucb, swts}, `eval.configs.dataset_ratio` must have "
            "three entries (train/val/test), e.g. [80,10,10]."
        )

def _safe_split_lengths(n: int, ratios: List[float]) -> List[int]:
    lengths = [int(float(n) * r) for r in ratios]
    remain = int(n) - int(sum(lengths))
    for i in range(remain):
        lengths[i % len(lengths)] += 1
    # If possible, ensure each partition has at least one sample.
    if n >= len(lengths):
        for idx in range(len(lengths)):
            if lengths[idx] > 0:
                continue
            donor = int(np.argmax(lengths))
            if lengths[donor] > 1:
                lengths[donor] -= 1
                lengths[idx] = 1
    return lengths

def _normalize_client_tuple(entry) -> Tuple[Optional[object], Optional[object], Optional[object]]:
    if not isinstance(entry, tuple):
        raise ValueError("Each client dataset entry must be a tuple.")
    if len(entry) == 2:
        train_ds, test_ds = entry
        return train_ds, None, test_ds
    if len(entry) == 3:
        train_ds, val_ds, test_ds = entry
        return train_ds, val_ds, test_ds
    raise ValueError("Each client dataset entry must be (train,test) or (train,val,test).")

def _apply_holdout_dataset_ratio(
    client_datasets,
    config: DictConfig,
    logger: Optional[ServerAgentFileLogger] = None,
):
    ratios = _parse_holdout_dataset_ratio(config)
    if ratios is None:
        return client_datasets
    train_only = len(ratios) == 1
    if train_only:
        _cfg_set(config, "eval.do_pre_evaluation", False)
        _cfg_set(config, "eval.do_post_evaluation", False)
        _cfg_set(config, "eval.enable_global_eval", False)
        _cfg_set(config, "eval.enable_federated_eval", False)
        if logger is not None:
            logger.info(
                "eval.configs.dataset_ratio=[1.0|100] detected: disabling validation/test "
                "and global/federated evaluation (training metrics only)."
            )

    seed = int(_cfg_get(config, "experiment.seed", 0))
    out = []
    for cid, entry in enumerate(client_datasets):
        train_ds, val_ds, test_ds = _normalize_client_tuple(entry)
        parts = [ds for ds in (train_ds, val_ds, test_ds) if ds is not None]
        if not parts:
            raise ValueError(f"Client dataset entry {cid} is empty.")
        merged = parts[0] if len(parts) == 1 else ConcatDataset(parts)
        total = len(merged)
        if total <= 0:
            if train_only:
                out.append((merged, None, None))
            elif len(ratios) == 2:
                out.append((merged, merged))
            else:
                out.append((merged, merged, merged))
            continue
        lengths = _safe_split_lengths(total, ratios)
        generator = torch.Generator().manual_seed(seed + 7919 + int(cid))
        splits = random_split(merged, lengths, generator=generator)
        if train_only:
            out.append((splits[0], None, None))
        elif len(ratios) == 2:
            out.append((splits[0], splits[1]))
        else:
            out.append((splits[0], splits[1], splits[2]))
    del logger
    return out

def _dataset_has_eval_split(dataset) -> bool:
    if dataset is None:
        return False
    try:
        return len(dataset) > 0
    except Exception:
        return True

def _validate_loader_output(client_datasets, runtime_cfg: Dict) -> None:
    num_clients = int(runtime_cfg["num_clients"])
    if len(client_datasets) != num_clients:
        raise ValueError(
            f"Loader/client metadata mismatch: len(client_datasets)={len(client_datasets)} "
            f"but num_clients={num_clients}"
        )
    for cid, pair in enumerate(client_datasets):
        if not (isinstance(pair, tuple) and len(pair) in {2, 3}):
            raise ValueError(
                f"client_datasets[{cid}] must be tuple(train,test) or tuple(train,val,test)."
            )

def _build_client_groups(config: DictConfig, num_clients: int) -> Tuple[List[int], List[int]]:
    all_clients = list(range(int(num_clients)))
    scheme = str(_cfg_get(config, "eval.configs.scheme", "dataset")).strip().lower()
    if scheme != "client":
        return all_clients, []

    holdout_num = int(_cfg_get(config, "eval.configs.client_counts", 0))
    holdout_ratio = float(_cfg_get(config, "eval.configs.client_ratio", 0.0))
    if holdout_num <= 0 and holdout_ratio > 0.0:
        holdout_num = max(1, int(round(num_clients * holdout_ratio)))
    holdout_num = max(0, min(holdout_num, max(0, num_clients - 1)))
    if holdout_num == 0:
        return all_clients, []

    rng = random.Random(int(_cfg_get(config, "experiment.seed", 42)) + 2026)
    shuffled = all_clients[:]
    rng.shuffle(shuffled)
    holdout = sorted(shuffled[:holdout_num])
    train_clients = sorted(cid for cid in all_clients if cid not in set(holdout))
    if not train_clients:
        return all_clients, []
    return train_clients, holdout

def _sample_train_clients(train_client_ids: List[int], num_sampled_clients: int) -> List[int]:
    if not train_client_ids:
        return []
    n = max(1, int(num_sampled_clients))
    n = min(n, len(train_client_ids))
    return sorted(random.sample(train_client_ids, n))

def _sample_eval_clients(
    config: DictConfig,
    client_ids: List[int],
    round_idx: int,
) -> List[int]:
    ids = sorted(int(cid) for cid in client_ids)
    total = len(ids)
    if total <= 1:
        return ids

    ratio = float(_cfg_get(config, "eval.configs.client_ratio", 1.0))
    ratio = min(1.0, max(0.0, ratio))
    seed = int(_cfg_get(config, "experiment.seed", 42))

    target = int(round(total * ratio))
    if ratio > 0.0 and target <= 0:
        target = 1
    if target <= 0:
        target = total
    target = max(1, min(total, target))

    if target >= total:
        return ids
    rng = random.Random(seed + 17 * int(round_idx))
    return sorted(rng.sample(ids, target))

def _resolve_client_eval_dataset(
    client_datasets: Sequence,
    client_id: int,
    eval_split: str,
):
    item = client_datasets[int(client_id)]
    if len(item) == 2:
        train_ds, test_ds = item
        val_ds = None
    elif len(item) == 3:
        train_ds, val_ds, test_ds = item
    else:
        raise ValueError(
            "Each client dataset entry must be tuple(train,test) or tuple(train,val,test)."
        )
    del train_ds
    chosen = str(eval_split).strip().lower()
    if chosen in {"val", "validation"}:
        return val_ds if val_ds is not None else test_ds
    return test_ds if test_ds is not None else val_ds
