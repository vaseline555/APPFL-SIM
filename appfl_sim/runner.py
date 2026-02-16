from __future__ import annotations

import copy
import sys
import time
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Sequence, Tuple, List, Optional

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from appfl_sim.agent import (
    ClientAgent,
    ClientAgentConfig,
    ServerAgent,
    ServerAgentConfig,
)
from appfl_sim.logger import ServerAgentFileLogger, create_experiment_tracker
from appfl_sim.loaders import load_dataset, load_model
from appfl_sim.misc.utils import get_local_rank, resolve_rank_device, set_seed_everything


def _default_config_path() -> Path:
    package_root = Path(__file__).resolve().parent
    candidates = [
        package_root / "config" / "examples" / "simulation.yaml",
        package_root.parent / "config" / "examples" / "simulation.yaml",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def _resolve_config_path(config_path: str) -> Path:
    raw = Path(config_path).expanduser()
    package_root = Path(__file__).resolve().parent

    candidates: List[Path] = []
    candidates.append(raw)
    if not raw.is_absolute():
        candidates.append(Path.cwd() / raw)

        raw_posix = raw.as_posix().lstrip("./")
        if raw_posix.startswith("config/"):
            suffix = raw_posix[len("config/") :]
            candidates.append(package_root / "config" / suffix)
        elif raw_posix.startswith("appfl_sim/config/"):
            suffix = raw_posix[len("appfl_sim/config/") :]
            candidates.append(package_root / "config" / suffix)
        else:
            candidates.append(package_root / raw_posix)

    seen = set()
    unique_candidates: List[Path] = []
    for path in candidates:
        key = str(path.resolve(strict=False))
        if key in seen:
            continue
        seen.add(key)
        unique_candidates.append(path)

    for path in unique_candidates:
        if path.exists():
            return path

    tried = "\n".join(f"  - {p}" for p in unique_candidates)
    raise FileNotFoundError(
        f"Config file not found for '{config_path}'. Tried:\n{tried}"
    )


def _print_help() -> None:
    print(
        """
appfl[sim] runner

Usage:
  python -m appfl_sim.runner --config /path/to/config.yaml
  appfl-sim backend=mpi dataset=MNIST num_clients=3 num_rounds=2
""".strip()
    )


def _cfg_to_dict(cfg) -> Dict:
    if isinstance(cfg, DictConfig):
        return dict(OmegaConf.to_container(cfg, resolve=True))
    if isinstance(cfg, SimpleNamespace):
        return dict(vars(cfg))
    if isinstance(cfg, dict):
        return dict(cfg)
    return dict(vars(cfg))


def _weighted_mean(stats: Dict[int, Dict], key: str) -> float:
    total = 0.0
    count = 0
    for values in stats.values():
        n = int(values.get("num_examples", 0))
        total += float(values.get(key, 0.0)) * n
        count += n
    return total / count if count > 0 else 0.0


def _cfg_bool(config: DictConfig, key: str, default: bool) -> bool:
    value = config.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "y", "on"}:
            return True
        if text in {"0", "false", "no", "n", "off"}:
            return False
        return default
    return bool(value)


def _mpi_download_mode(config: DictConfig) -> str:
    raw_mode = str(config.get("mpi_dataset_download_mode", "rank0")).strip().lower()
    aliases = {
        "rank0_then_barrier": "rank0",
        "root": "rank0",
        "local_rank0_then_barrier": "local_rank0",
        "node_leader": "local_rank0",
    }
    mode = aliases.get(raw_mode, raw_mode)
    supported = {"rank0", "local_rank0", "all", "none"}
    if mode not in supported:
        raise ValueError(
            "mpi_dataset_download_mode must be one of: rank0, local_rank0, all, none"
        )
    return mode


def _load_dataset_mpi(config: DictConfig, communicator, rank: int):
    loader_cfg = _cfg_to_dict(config)
    mode = _mpi_download_mode(config)
    local_rank = get_local_rank(default=max(rank - 1, 0))

    if mode == "all":
        loader_cfg["download"] = True
        return load_dataset(loader_cfg)

    if mode == "none":
        loader_cfg["download"] = False
        return load_dataset(loader_cfg)

    if mode == "rank0":
        cached_result = None
        if rank == 0:
            cfg_root = dict(loader_cfg)
            cfg_root["download"] = True
            cached_result = load_dataset(cfg_root)
        communicator.barrier()
        if rank == 0:
            return cached_result
        cfg_other = dict(loader_cfg)
        cfg_other["download"] = False
        return load_dataset(cfg_other)

    # mode == local_rank0
    cached_result = None
    if local_rank == 0:
        cfg_leader = dict(loader_cfg)
        cfg_leader["download"] = True
        cached_result = load_dataset(cfg_leader)
    communicator.barrier()
    if local_rank == 0:
        return cached_result
    cfg_follower = dict(loader_cfg)
    cfg_follower["download"] = False
    return load_dataset(cfg_follower)


def _dataset_has_eval_split(dataset) -> bool:
    if dataset is None:
        return False
    try:
        return len(dataset) > 0
    except Exception:
        return True


def _should_eval_round(round_idx: int, every: int, num_rounds: int) -> bool:
    return round_idx % max(1, int(every)) == 0 or round_idx == int(num_rounds)


def _build_train_cfg(config: DictConfig, device: str) -> Dict:
    return {
        "device": device,
        "batch_size": int(config.batch_size),
        "num_workers": int(config.num_workers),
        "local_epochs": int(config.local_epochs),
        "optimizer": str(config.optimizer),
        "optim_args": {
            "lr": float(config.lr),
            "weight_decay": float(config.weight_decay),
        },
        "max_grad_norm": float(config.max_grad_norm),
        "logging_output_dirname": str(config.log_dir),
        "logging_output_filename": str(config.log_file),
        "experiment_id": str(config.exp_name),
    }


def _build_clients(
    config: DictConfig,
    model,
    client_datasets: Sequence,
    local_client_ids,
    device: str,
):
    train_cfg = _build_train_cfg(config, device=device)
    clients = []
    for cid in local_client_ids:
        train_ds, test_ds = client_datasets[int(cid)]
        client_cfg = ClientAgentConfig(
            train_configs=OmegaConf.create(
                {
                    **train_cfg,
                    "loss_fn": str(config.criterion),
                }
            ),
            model_configs=OmegaConf.create({}),
            data_configs=OmegaConf.create({}),
        )
        client_cfg.client_id = str(int(cid))
        client_cfg.experiment_id = str(config.exp_name)
        client = ClientAgent(client_agent_config=client_cfg)
        client.model = copy.deepcopy(model)
        client.train_dataset = train_ds
        client.val_dataset = test_ds
        client.client_agent_config.train_configs.trainer = "SimVanillaTrainer"
        client._load_trainer()
        client.id = int(cid)
        clients.append(
            client
        )
    return clients


def _build_server(
    config: DictConfig,
    runtime_cfg: Dict,
    model,
    server_dataset,
) -> ServerAgent:
    num_clients = int(runtime_cfg["num_clients"])
    server_cfg = ServerAgentConfig(
        client_configs=OmegaConf.create(
            {
                "train_configs": {
                    "loss_fn": str(config.criterion),
                },
                "model_configs": {},
            }
        ),
        server_configs=OmegaConf.create(
            {
                "num_clients": num_clients,
                "num_global_epochs": int(config.num_rounds),
                "client_fraction": float(config.client_fraction),
                "device": str(config.server_device),
                "eval_batch_size": int(config.get("eval_batch_size", config.batch_size)),
                "num_workers": int(config.num_workers),
                "aggregator": "FedAvgAggregator",
                "aggregator_kwargs": {
                    "client_weights_mode": "sample_size",
                },
                "scheduler": "SyncScheduler",
                "scheduler_kwargs": {
                    "num_clients": num_clients,
                    "same_init_model": False,
                },
            }
        ),
    )
    server = ServerAgent(server_agent_config=server_cfg)
    server.model = model
    if (
        hasattr(server, "aggregator")
        and server.aggregator is not None
        and hasattr(server.aggregator, "model")
        and getattr(server.aggregator, "model", None) is None
    ):
        server.aggregator.model = server.model
    server.loss_fn = torch.nn.__dict__[str(config.criterion)]()
    server._val_dataset = server_dataset
    server._load_val_data()
    return server


def _log_round(
    config: DictConfig,
    round_idx: int,
    selected_count: int,
    stats,
    weights,
    global_eval_metrics: Optional[Dict[str, float]] = None,
    federated_eval_metrics: Optional[Dict[str, float]] = None,
    federated_eval_in_metrics: Optional[Dict[str, float]] = None,
    federated_eval_out_metrics: Optional[Dict[str, float]] = None,
    logger: ServerAgentFileLogger | None = None,
    tracker=None,
):
    train_loss = _weighted_mean(stats, "loss")
    train_acc = _weighted_mean(stats, "accuracy")
    round_metrics: Dict[str, float] = {
        "train/loss": float(train_loss),
        "train/accuracy": float(train_acc),
        "train/selected_clients": float(selected_count),
    }
    log = (
        f"[Round {round_idx:04d}] selected={selected_count} "
        f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}"
    )

    if global_eval_metrics is not None:
        round_metrics["eval/loss"] = float(global_eval_metrics["loss"])
        round_metrics["eval/accuracy"] = float(global_eval_metrics["accuracy"])
        log += (
            f" | global_loss={global_eval_metrics['loss']:.4f}"
            f" global_acc={global_eval_metrics['accuracy']:.4f}"
        )
    if federated_eval_metrics is not None:
        round_metrics["fed_eval/loss"] = float(federated_eval_metrics["loss"])
        round_metrics["fed_eval/accuracy"] = float(federated_eval_metrics["accuracy"])
        round_metrics["fed_eval/num_clients"] = float(
            federated_eval_metrics.get("num_clients", 0)
        )
        log += (
            f" | fed_eval_loss={federated_eval_metrics['loss']:.4f}"
            f" fed_eval_acc={federated_eval_metrics['accuracy']:.4f}"
            f" fed_eval_clients={int(federated_eval_metrics.get('num_clients', 0))}"
        )
    if federated_eval_in_metrics is not None:
        round_metrics["fed_eval_in/loss"] = float(federated_eval_in_metrics["loss"])
        round_metrics["fed_eval_in/accuracy"] = float(
            federated_eval_in_metrics["accuracy"]
        )
        round_metrics["fed_eval_in/num_clients"] = float(
            federated_eval_in_metrics.get("num_clients", 0)
        )
        log += (
            f" | fed_in_loss={federated_eval_in_metrics['loss']:.4f}"
            f" fed_in_acc={federated_eval_in_metrics['accuracy']:.4f}"
            f" fed_in_clients={int(federated_eval_in_metrics.get('num_clients', 0))}"
        )
    if federated_eval_out_metrics is not None:
        round_metrics["fed_eval_out/loss"] = float(federated_eval_out_metrics["loss"])
        round_metrics["fed_eval_out/accuracy"] = float(
            federated_eval_out_metrics["accuracy"]
        )
        round_metrics["fed_eval_out/num_clients"] = float(
            federated_eval_out_metrics.get("num_clients", 0)
        )
        log += (
            f" | fed_out_loss={federated_eval_out_metrics['loss']:.4f}"
            f" fed_out_acc={federated_eval_out_metrics['accuracy']:.4f}"
            f" fed_out_clients={int(federated_eval_out_metrics.get('num_clients', 0))}"
        )

    if weights:
        min_w = min(weights.values())
        max_w = max(weights.values())
        round_metrics["agg/weight_min"] = float(min_w)
        round_metrics["agg/weight_max"] = float(max_w)
        log += f" | agg_w[min,max]=({min_w:.4f},{max_w:.4f})"

    if logger is not None:
        logger.info(log)
    else:
        print(log)
    if tracker is not None:
        tracker.log_metrics(step=round_idx, metrics=round_metrics)


def _new_server_logger(config: DictConfig, mode: str) -> ServerAgentFileLogger:
    file_name = f"{config.log_file}_{mode}"
    return ServerAgentFileLogger(
        file_dir=str(config.log_dir),
        file_name=file_name,
        experiment_id=str(config.exp_name),
    )


def _merge_runtime_cfg(config: DictConfig, loader_args: SimpleNamespace | dict) -> Dict:
    runtime_cfg = _cfg_to_dict(config)
    if isinstance(loader_args, SimpleNamespace):
        runtime_cfg.update(vars(loader_args))
    elif isinstance(loader_args, dict):
        runtime_cfg.update(loader_args)
    runtime_cfg["num_clients"] = int(
        runtime_cfg.get("num_clients", runtime_cfg.get("K", int(config.num_clients)))
    )
    runtime_cfg["K"] = int(runtime_cfg["num_clients"])
    return runtime_cfg


def _validate_loader_output(client_datasets, runtime_cfg: Dict) -> None:
    num_clients = int(runtime_cfg["num_clients"])
    if len(client_datasets) != num_clients:
        raise ValueError(
            f"Loader/client metadata mismatch: len(client_datasets)={len(client_datasets)} "
            f"but num_clients={num_clients}"
        )
    for cid, pair in enumerate(client_datasets):
        if not (isinstance(pair, tuple) and len(pair) == 2):
            raise ValueError(
                f"client_datasets[{cid}] must be a tuple(train_dataset, test_dataset)."
            )


def _build_client_groups(config: DictConfig, num_clients: int) -> Tuple[List[int], List[int]]:
    all_clients = list(range(int(num_clients)))
    scheme = str(config.get("federated_eval_scheme", "holdout_dataset")).strip().lower()
    if scheme != "holdout_client":
        return all_clients, []

    holdout_num = int(config.get("holdout_eval_num_clients", 0))
    holdout_ratio = float(config.get("holdout_eval_client_ratio", 0.0))
    if holdout_num <= 0 and holdout_ratio > 0.0:
        holdout_num = max(1, int(round(num_clients * holdout_ratio)))
    holdout_num = max(0, min(holdout_num, max(0, num_clients - 1)))
    if holdout_num == 0:
        return all_clients, []

    rng = random.Random(int(config.seed) + 2026)
    shuffled = all_clients[:]
    rng.shuffle(shuffled)
    holdout = sorted(shuffled[:holdout_num])
    train_clients = sorted(cid for cid in all_clients if cid not in set(holdout))
    if not train_clients:
        return all_clients, []
    return train_clients, holdout


def _sample_train_clients(train_client_ids: List[int], client_fraction: float) -> List[int]:
    if not train_client_ids:
        return []
    n = max(1, int(float(client_fraction) * len(train_client_ids)))
    n = min(n, len(train_client_ids))
    return sorted(random.sample(train_client_ids, n))


def _build_federated_eval_plan(
    config: DictConfig,
    round_idx: int,
    num_rounds: int,
    selected_train_ids: List[int],
    train_client_ids: List[int],
    holdout_client_ids: List[int],
) -> Dict[str, List[int] | str | bool]:
    scheme = str(config.get("federated_eval_scheme", "holdout_dataset")).strip().lower()
    fed_eval_every = int(config.get("federated_eval_every", int(config.eval_every)))
    checkpoint = _should_eval_round(round_idx, fed_eval_every, num_rounds)

    if scheme == "holdout_client":
        in_ids = sorted(train_client_ids if checkpoint else selected_train_ids)
        out_ids = sorted(holdout_client_ids if checkpoint else [])
        return {
            "scheme": "holdout_client",
            "checkpoint": checkpoint,
            "in_ids": in_ids,
            "out_ids": out_ids,
        }

    # Default: holdout_dataset-based evaluation.
    # During training rounds evaluate selected clients only;
    # at checkpoint rounds evaluate all training clients.
    in_ids = sorted(train_client_ids if checkpoint else selected_train_ids)
    return {
        "scheme": "holdout_dataset",
        "checkpoint": checkpoint,
        "in_ids": in_ids,
        "out_ids": [],
    }


def _run_federated_eval_mpi(
    communicator,
    model_state,
    round_idx: int,
    eval_ids: List[int],
) -> Optional[Dict[str, float]]:
    if not eval_ids:
        return None
    communicator.broadcast_global_model(
        model=model_state,
        args={
            "done": False,
            "mode": "eval",
            "round": int(round_idx),
            "eval_ids": list(eval_ids),
        },
    )
    eval_payloads = communicator.recv_all_local_models_from_clients()
    eval_stats = {}
    for cid, payload in eval_payloads.items():
        if isinstance(payload, dict) and "eval_stats" in payload:
            eval_stats[int(cid)] = payload["eval_stats"]
    return _aggregate_eval_stats(eval_stats)


def _aggregate_eval_stats(stats: Dict[int, Dict]) -> Optional[Dict[str, float]]:
    if not stats:
        return None
    total_examples = sum(int(v.get("num_examples", 0)) for v in stats.values())
    if total_examples <= 0:
        return {
            "loss": -1.0,
            "accuracy": -1.0,
            "num_examples": 0,
            "num_clients": len(stats),
        }
    return {
        "loss": float(_weighted_mean(stats, "loss")),
        "accuracy": float(_weighted_mean(stats, "accuracy")),
        "num_examples": int(total_examples),
        "num_clients": int(len(stats)),
    }


def _run_federated_eval_serial(
    clients,
    global_state,
    eval_client_ids: List[int],
) -> Optional[Dict[str, float]]:
    if not eval_client_ids:
        return None
    eval_set = set(eval_client_ids)
    eval_stats: Dict[int, Dict] = {}
    for client in clients:
        if client.id not in eval_set:
            continue
        client.download(global_state)
        eval_stats[int(client.id)] = client.evaluate()
    return _aggregate_eval_stats(eval_stats)


def _collect_server_updates(rank_payloads: Dict[int, Dict]) -> Tuple[Dict, Dict, Dict]:
    local_states = {}
    sample_sizes = {}
    train_stats = {}
    for cid, payload in rank_payloads.items():
        local_states[int(cid)] = payload["state"]
        sample_sizes[int(cid)] = int(payload.get("num_examples", 0))
        train_stats[int(cid)] = payload.get("stats", {})
    return local_states, sample_sizes, train_stats


def _run_server_mpi(
    config: DictConfig,
    communicator,
    server: ServerAgent,
    train_client_ids: List[int],
    holdout_client_ids: List[int],
    enable_global_eval: bool,
    enable_federated_eval: bool,
    logger: ServerAgentFileLogger | None = None,
    tracker=None,
):
    t0 = time.time()
    fed_scheme = str(config.get("federated_eval_scheme", "holdout_dataset"))
    start_msg = (
        f"[appfl-sim] start(mpi) experiment='{config.exp_name}' algo={config.algorithm} "
        f"dataset={config.dataset} clients={server.num_clients} rounds={config.num_rounds} "
        f"fed_eval_scheme={fed_scheme} train_clients={len(train_client_ids)} "
        f"holdout_eval_clients={len(holdout_client_ids)}"
    )
    if logger is not None:
        logger.info(start_msg)
    else:
        print(start_msg)

    round_idx = 1
    while True:
        if round_idx > int(config.num_rounds):
            communicator.broadcast_global_model(args={"done": True})
            break

        selected_ids = _sample_train_clients(
            train_client_ids=train_client_ids,
            client_fraction=float(server.client_fraction),
        )
        communicator.broadcast_global_model(
            model=server.model.state_dict(),
            args={
                "done": False,
                "mode": "train",
                "round": round_idx,
                "selected_ids": selected_ids,
            },
        )

        rank_payloads = communicator.recv_all_local_models_from_clients()
        local_states, sample_sizes, train_stats = _collect_server_updates(rank_payloads)
        weights = server.aggregate(local_states, sample_sizes)

        global_eval_metrics = None
        if enable_global_eval and _should_eval_round(
            round_idx, int(config.eval_every), int(config.num_rounds)
        ):
            global_eval_metrics = server.evaluate()

        federated_eval_metrics = None
        federated_eval_in_metrics = None
        federated_eval_out_metrics = None
        if enable_federated_eval:
            plan = _build_federated_eval_plan(
                config=config,
                round_idx=round_idx,
                num_rounds=int(config.num_rounds),
                selected_train_ids=selected_ids,
                train_client_ids=train_client_ids,
                holdout_client_ids=holdout_client_ids,
            )
            if plan["scheme"] == "holdout_client":
                federated_eval_in_metrics = _run_federated_eval_mpi(
                    communicator=communicator,
                    model_state=server.model.state_dict(),
                    round_idx=round_idx,
                    eval_ids=list(plan["in_ids"]),
                )
                federated_eval_out_metrics = _run_federated_eval_mpi(
                    communicator=communicator,
                    model_state=server.model.state_dict(),
                    round_idx=round_idx,
                    eval_ids=list(plan["out_ids"]),
                )
            else:
                federated_eval_metrics = _run_federated_eval_mpi(
                    communicator=communicator,
                    model_state=server.model.state_dict(),
                    round_idx=round_idx,
                    eval_ids=list(plan["in_ids"]),
                )

        _log_round(
            config,
            round_idx,
            len(selected_ids),
            train_stats,
            weights,
            global_eval_metrics=global_eval_metrics,
            federated_eval_metrics=federated_eval_metrics,
            federated_eval_in_metrics=federated_eval_in_metrics,
            federated_eval_out_metrics=federated_eval_out_metrics,
            logger=logger,
            tracker=tracker,
        )
        round_idx += 1

    communicator.barrier()
    finish_msg = f"[appfl-sim] finished(mpi) in {time.time() - t0:.2f}s"
    if logger is not None:
        logger.info(finish_msg)
    else:
        print(finish_msg)


def _run_client_mpi(communicator, clients):
    while True:
        incoming = communicator.recv_global_model_from_server(source=0)
        if isinstance(incoming, tuple):
            global_state, args = incoming
        else:
            global_state, args = incoming, {}

        if bool(args.get("done", False)):
            break

        mode = str(args.get("mode", "train")).lower()
        round_idx = int(args.get("round", 0))

        local_payload = {}
        if mode == "eval":
            eval_ids = set(args.get("eval_ids", []))
            for client in clients:
                if client.id not in eval_ids:
                    continue
                client.download(global_state)
                local_payload[int(client.id)] = {"eval_stats": client.evaluate()}
        else:
            selected = set(args.get("selected_ids", []))
            for client in clients:
                if client.id not in selected:
                    continue
                client.download(global_state)
                train_result = client.update(round_idx=round_idx)
                local_payload[int(client.id)] = {
                    "state": client.upload(),
                    "num_examples": int(train_result.get("num_examples", 0)),
                    "stats": train_result,
                }

        communicator.send_local_models_to_server(local_payload, dest=0)

    communicator.barrier()


def run_serial(config) -> None:
    if not isinstance(config, DictConfig):
        config = OmegaConf.create(_cfg_to_dict(config))

    set_seed_everything(int(config.seed))
    t0 = time.time()
    server_logger = _new_server_logger(config, mode="serial")
    tracker = create_experiment_tracker(config)

    loader_cfg = _cfg_to_dict(config)
    loader_cfg["download"] = True
    _, client_datasets, server_dataset, args = load_dataset(loader_cfg)

    runtime_cfg = _merge_runtime_cfg(config, args)
    _validate_loader_output(client_datasets, runtime_cfg)
    num_clients = int(runtime_cfg["num_clients"])
    train_client_ids, holdout_client_ids = _build_client_groups(config, num_clients)
    enable_global_eval = _cfg_bool(config, "enable_global_eval", True) and _dataset_has_eval_split(
        server_dataset
    )
    enable_federated_eval = _cfg_bool(config, "enable_federated_eval", True)

    model = load_model(
        runtime_cfg,
        input_shape=tuple(runtime_cfg["input_shape"]),
        num_classes=int(runtime_cfg["num_classes"]),
    )

    server = _build_server(
        config=config,
        runtime_cfg=runtime_cfg,
        model=model,
        server_dataset=server_dataset,
    )

    clients = _build_clients(
        config=config,
        model=model,
        client_datasets=client_datasets,
        local_client_ids=np.arange(num_clients).astype(int),
        device=resolve_rank_device(str(config.device), rank=1, world_size=2),
    )

    server_logger.info(
        f"[appfl-sim] start(serial) experiment='{config.exp_name}' algo={config.algorithm} "
        f"dataset={config.dataset} clients={num_clients} rounds={config.num_rounds} "
        f"fed_eval_scheme={config.get('federated_eval_scheme', 'holdout_dataset')} "
        f"train_clients={len(train_client_ids)} holdout_eval_clients={len(holdout_client_ids)}"
    )

    for round_idx in range(1, int(config.num_rounds) + 1):
        selected_ids = _sample_train_clients(
            train_client_ids=train_client_ids,
            client_fraction=float(server.client_fraction),
        )
        selected = set(selected_ids)
        global_state = server.model.state_dict()

        updates = {}
        sample_sizes = {}
        stats = {}
        for client in clients:
            if client.id not in selected:
                continue
            client.download(global_state)
            train_result = client.update(round_idx=round_idx)
            updates[client.id] = client.upload()
            sample_sizes[client.id] = int(train_result["num_examples"])
            stats[client.id] = train_result

        weights = server.aggregate(updates, sample_sizes)
        global_eval_metrics = None
        if enable_global_eval and _should_eval_round(
            round_idx, int(config.eval_every), int(config.num_rounds)
        ):
            global_eval_metrics = server.evaluate()

        federated_eval_metrics = None
        federated_eval_in_metrics = None
        federated_eval_out_metrics = None
        if enable_federated_eval:
            plan = _build_federated_eval_plan(
                config=config,
                round_idx=round_idx,
                num_rounds=int(config.num_rounds),
                selected_train_ids=selected_ids,
                train_client_ids=train_client_ids,
                holdout_client_ids=holdout_client_ids,
            )
            if plan["scheme"] == "holdout_client":
                federated_eval_in_metrics = _run_federated_eval_serial(
                    clients=clients,
                    global_state=server.model.state_dict(),
                    eval_client_ids=list(plan["in_ids"]),
                )
                federated_eval_out_metrics = _run_federated_eval_serial(
                    clients=clients,
                    global_state=server.model.state_dict(),
                    eval_client_ids=list(plan["out_ids"]),
                )
            else:
                federated_eval_metrics = _run_federated_eval_serial(
                    clients=clients,
                    global_state=server.model.state_dict(),
                    eval_client_ids=list(plan["in_ids"]),
                )

        _log_round(
            config,
            round_idx,
            len(selected_ids),
            stats,
            weights,
            global_eval_metrics=global_eval_metrics,
            federated_eval_metrics=federated_eval_metrics,
            federated_eval_in_metrics=federated_eval_in_metrics,
            federated_eval_out_metrics=federated_eval_out_metrics,
            logger=server_logger,
            tracker=tracker,
        )

    server_logger.info(f"[appfl-sim] finished(serial) in {time.time() - t0:.2f}s")
    if tracker is not None:
        tracker.close()


def run_mpi(config) -> None:
    if not isinstance(config, DictConfig):
        config = OmegaConf.create(_cfg_to_dict(config))

    from appfl_sim.comm import MpiSyncCommunicator, get_mpi_comm

    communicator = MpiSyncCommunicator(get_mpi_comm())
    rank = communicator.rank
    world_size = communicator.size
    local_rank = get_local_rank(default=max(rank - 1, 0))

    if world_size < 2:
        if rank == 0:
            raise RuntimeError(
                "MPI world size must be >= 2 (rank0 server + at least 1 client rank)."
            )
        return

    set_seed_everything(int(config.seed))
    _, client_datasets, server_dataset, args = _load_dataset_mpi(
        config=config, communicator=communicator, rank=rank
    )

    runtime_cfg = _merge_runtime_cfg(config, args)
    _validate_loader_output(client_datasets, runtime_cfg)

    num_clients = int(runtime_cfg["num_clients"])
    train_client_ids, holdout_client_ids = _build_client_groups(config, num_clients)
    enable_global_eval = _cfg_bool(config, "enable_global_eval", True) and _dataset_has_eval_split(
        server_dataset
    )
    enable_federated_eval = _cfg_bool(config, "enable_federated_eval", True)
    model = load_model(
        runtime_cfg,
        input_shape=tuple(runtime_cfg["input_shape"]),
        num_classes=int(runtime_cfg["num_classes"]),
    )

    set_seed_everything(int(config.seed) + rank)
    use_local_rank_device = _cfg_bool(config, "mpi_use_local_rank_device", True)
    client_local_rank = local_rank if use_local_rank_device else max(rank - 1, 0)

    if rank == 0:
        server_logger = _new_server_logger(config, mode="mpi")
        server_logger.info(
            f"[appfl-sim] mpi world_size={world_size} rank={rank} local_rank={local_rank} "
            f"download_mode={_mpi_download_mode(config)}"
        )
        tracker = create_experiment_tracker(config)
        server = _build_server(
            config=config,
            runtime_cfg=runtime_cfg,
            model=model,
            server_dataset=server_dataset,
        )
        _run_server_mpi(
            config,
            communicator,
            server,
            train_client_ids=train_client_ids,
            holdout_client_ids=holdout_client_ids,
            enable_global_eval=enable_global_eval,
            enable_federated_eval=enable_federated_eval,
            logger=server_logger,
            tracker=tracker,
        )
        if tracker is not None:
            tracker.close()
        return

    client_groups = np.array_split(np.arange(num_clients), world_size - 1)
    local_client_ids = np.asarray(client_groups[rank - 1]).astype(int)

    client_device = resolve_rank_device(
        str(config.device),
        rank=rank,
        world_size=world_size,
        local_rank=client_local_rank,
    )
    if _cfg_bool(config, "mpi_log_rank_mapping", False):
        print(
            f"[appfl-sim][rank={rank}] local_rank={local_rank} "
            f"client_device={client_device} num_local_clients={len(local_client_ids)}"
        )

    clients = _build_clients(
        config=config,
        model=model,
        client_datasets=client_datasets,
        local_client_ids=local_client_ids,
        device=client_device,
    )
    _run_client_mpi(communicator, clients)


def _extract_config_path(argv: list[str]) -> tuple[str | None, list[str]]:
    config_path = None
    remaining: list[str] = []
    idx = 0
    while idx < len(argv):
        token = argv[idx]
        if token in {"--config", "-c"}:
            if idx + 1 >= len(argv):
                raise ValueError("--config requires a file path")
            config_path = argv[idx + 1]
            idx += 2
            continue
        if token.startswith("--config="):
            config_path = token.split("=", 1)[1]
            idx += 1
            continue
        if token.startswith("config="):
            config_path = token.split("=", 1)[1]
            idx += 1
            continue
        remaining.append(token)
        idx += 1
    return config_path, remaining


def _normalize_cli_tokens(tokens: list[str]) -> tuple[str | None, list[str]]:
    backend = None
    out: list[str] = []
    idx = 0
    while idx < len(tokens):
        tok = tokens[idx]
        if tok in {"-h", "--help"}:
            _print_help()
            raise SystemExit(0)
        if tok in {"serial", "mpi"}:
            backend = tok
            idx += 1
            continue
        if tok.startswith("--"):
            keyval = tok[2:]
            if "=" in keyval:
                key, value = keyval.split("=", 1)
                out.append(f"{key.replace('-', '_')}={value}")
                idx += 1
                continue
            key = keyval.replace("-", "_")
            if key == "no_need_embedding":
                out.append("need_embedding=false")
                idx += 1
                continue
            if idx + 1 < len(tokens) and not tokens[idx + 1].startswith("--"):
                out.append(f"{key}={tokens[idx + 1]}")
                idx += 2
            else:
                out.append(f"{key}=true")
                idx += 1
            continue
        if "=" in tok:
            key, value = tok.split("=", 1)
            key = key.replace("-", "_")
            if key == "backend":
                backend = value
            elif key == "no_need_embedding":
                out.append("need_embedding=false")
            else:
                out.append(f"{key}={value}")
        idx += 1
    return backend, out


def parse_config(argv: list[str] | None = None) -> tuple[str, DictConfig]:
    argv = list(sys.argv[1:] if argv is None else argv)
    config_path, remaining = _extract_config_path(argv)
    backend_override, dotlist = _normalize_cli_tokens(remaining)

    default_path = _default_config_path()
    if not default_path.exists():
        raise FileNotFoundError(f"Default config not found: {default_path}")

    cfg = OmegaConf.load(default_path)
    if config_path:
        cfg_path = _resolve_config_path(config_path)
        cfg = OmegaConf.merge(cfg, OmegaConf.load(cfg_path))
    if dotlist:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(dotlist))
    if backend_override is not None:
        cfg.backend = backend_override

    backend = str(cfg.get("backend", "mpi")).lower()
    if backend not in {"serial", "mpi"}:
        raise ValueError("backend must be one of: serial, mpi")

    return backend, cfg


def main() -> None:
    backend, config = parse_config()
    if backend == "serial":
        run_serial(config)
    else:
        run_mpi(config)


if __name__ == "__main__":
    main()
