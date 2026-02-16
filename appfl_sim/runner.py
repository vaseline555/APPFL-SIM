from __future__ import annotations

import copy
import sys
import time
import random
from datetime import datetime
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


def _build_train_cfg(config: DictConfig, device: str, run_log_dir: str) -> Dict:
    return {
        "device": device,
        "mode": "epoch",
        "num_local_epochs": int(config.local_epochs),
        "train_batch_size": int(config.batch_size),
        "val_batch_size": int(config.get("eval_batch_size", config.batch_size)),
        "num_workers": int(config.num_workers),
        "optim": str(config.optimizer),
        "optim_args": {
            "lr": float(config.lr),
            "weight_decay": float(config.weight_decay),
        },
        "max_grad_norm": float(config.max_grad_norm),
        "logging_output_dirname": str(run_log_dir),
        "logging_output_filename": "client",
        "experiment_id": str(config.exp_name),
        "client_logging_enabled": True,
        "client_log_title_every": int(config.get("client_log_title_every", 0)),
        "client_log_show_titles": _cfg_bool(config, "client_log_show_titles", True),
        "client_log_title_each_round": _cfg_bool(
            config, "client_log_title_each_round", True
        ),
        "do_pre_validation": _cfg_bool(config, "do_pre_validation", True),
        "do_validation": _cfg_bool(config, "do_validation", True),
    }


def _resolve_client_logging_policy(
    config: DictConfig,
    num_clients: int,
) -> Dict[str, object]:
    scheme = str(config.get("client_logging_scheme", "auto")).strip().lower()
    threshold = int(config.get("per_client_logging_threshold", 10))
    warning_threshold = int(config.get("per_client_logging_warning_threshold", 50))
    agg_scheme = str(config.get("aggregated_logging_scheme", "server_only")).strip().lower()

    if scheme not in {"auto", "per_client", "aggregated"}:
        raise ValueError(
            "client_logging_scheme must be one of: auto, per_client, aggregated"
        )
    if agg_scheme != "server_only":
        raise ValueError("aggregated_logging_scheme currently supports only: server_only")

    if scheme == "aggregated":
        effective = "aggregated"
    elif scheme == "per_client":
        effective = "per_client"
    else:
        effective = "per_client" if int(num_clients) <= threshold else "aggregated"

    return {
        "requested_scheme": scheme,
        "effective_scheme": effective,
        "client_logging_enabled": effective == "per_client",
        "threshold": threshold,
        "warning_threshold": warning_threshold,
        "aggregated_scheme": agg_scheme,
    }


def _emit_logging_policy_message(
    policy: Dict[str, object],
    num_clients: int,
    logger: Optional[ServerAgentFileLogger] = None,
) -> None:
    requested = str(policy["requested_scheme"])
    effective = str(policy["effective_scheme"])
    threshold = int(policy["threshold"])
    warning_threshold = int(policy["warning_threshold"])
    agg_scheme = str(policy["aggregated_scheme"])

    def _info(msg: str) -> None:
        if logger is not None:
            logger.info(msg)
        else:
            print(msg)

    def _warn(msg: str) -> None:
        if logger is not None:
            logger.warning(msg)
        else:
            print(msg)

    if requested == "auto" and effective == "aggregated":
        _info(
            f"Client logging auto-disabled: num_clients={num_clients} > "
            f"per_client_logging_threshold={threshold}. "
            f"Using aggregated_logging_scheme={agg_scheme} (server-side metrics only)."
        )
        return
    if requested == "aggregated":
        _info(
            f"Using aggregated_logging_scheme={agg_scheme} (server-side metrics only)."
        )
        return
    if requested == "per_client" and int(num_clients) > warning_threshold:
        _warn(
            f"Per-client logging is explicitly enabled with "
            f"num_clients={num_clients} (> {warning_threshold}). "
            "This may produce large I/O overhead. "
            "Suggestion: set client_logging_scheme=auto or aggregated."
        )


def _build_clients(
    config: DictConfig,
    model,
    client_datasets: Sequence,
    local_client_ids,
    device: str,
    run_log_dir: str,
    client_logging_enabled: bool = True,
):
    train_cfg = _build_train_cfg(config, device=device, run_log_dir=run_log_dir)
    train_cfg["client_logging_enabled"] = bool(client_logging_enabled)
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
        client.client_agent_config.train_configs.trainer = "VanillaTrainer"
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
    total_train_clients: int,
    stats,
    weights,
    global_eval_metrics: Optional[Dict[str, float]] = None,
    federated_eval_metrics: Optional[Dict[str, float]] = None,
    federated_eval_in_metrics: Optional[Dict[str, float]] = None,
    federated_eval_out_metrics: Optional[Dict[str, float]] = None,
    logger: ServerAgentFileLogger | None = None,
    tracker=None,
):
    del weights

    def _entity_line(title: str, body: str) -> str:
        return f"  {title:<18} {body}"

    round_metrics: Dict[str, object] = {
        "clients": {
            "selected": int(selected_count),
            "total": int(total_train_clients),
        }
    }
    lines = [
        "--- Round Summary ---",
        _entity_line("Clients:", f"selected={selected_count}/{total_train_clients}"),
    ]

    if stats:
        numeric_train_keys = sorted(
            {
                k
                for values in stats.values()
                for k, v in values.items()
                if isinstance(v, (int, float))
                and k != "num_examples"
                and not k.startswith("pre_val_")
                and not k.startswith("post_val_")
            }
        )
        if numeric_train_keys:
            train_parts = []
            training_metrics: Dict[str, Dict[str, float]] = {}
            for key in numeric_train_keys:
                values = [float(v.get(key, 0.0)) for v in stats.values()]
                avg_value = float(np.mean(values))
                std_value = float(np.std(values))
                training_metrics[key] = {
                    "avg": avg_value,
                    "std": std_value,
                }
                train_parts.append(f"{key}={avg_value:.4f}/{std_value:.4f}")
            round_metrics["training"] = training_metrics
            lines.append(_entity_line("Training:", " | ".join(train_parts)))

    def _append_eval_block(
        title: str,
        json_key: str,
        metrics: Optional[Dict[str, float]],
        with_client_std: bool = False,
    ) -> None:
        if metrics is None:
            return
        parts = []
        section_metrics: Dict[str, object] = {}
        for key, value in sorted(metrics.items()):
            if not isinstance(value, (int, float)):
                continue
            if key in {"num_clients", "num_examples"}:
                continue
            if key.endswith("_min") or key.endswith("_max"):
                continue
            if with_client_std and key in {"loss_std", "accuracy_std"}:
                continue
            if with_client_std and key in {"loss", "accuracy"}:
                std_key = f"{key}_std"
                if std_key in metrics and isinstance(metrics[std_key], (int, float)):
                    section_metrics[key] = {
                        "avg": float(value),
                        "std": float(metrics[std_key]),
                    }
                    parts.append(f"{key}={float(value):.4f}/{float(metrics[std_key]):.4f}")
                else:
                    section_metrics[key] = float(value)
                    parts.append(f"{key}={float(value):.4f}")
            else:
                section_metrics[key] = float(value)
                parts.append(f"{key}={float(value):.4f}")
        if parts:
            lines.append(_entity_line(f"{title}:", " | ".join(parts)))
            round_metrics[json_key] = section_metrics

    def _append_federated_extrema(metrics: Optional[Dict[str, float]]) -> None:
        if metrics is None:
            return
        if not (
            "loss_min" in metrics
            and "loss_max" in metrics
            and "accuracy_min" in metrics
            and "accuracy_max" in metrics
        ):
            return
        lines.append(
            _entity_line(
                "Federated Extrema:",
                f"accuracy[min,max]=[{float(metrics['accuracy_min']):.4f},{float(metrics['accuracy_max']):.4f}]"
                " | "
                f"loss[min,max]=[{float(metrics['loss_min']):.4f},{float(metrics['loss_max']):.4f}]",
            )
        )
        round_metrics["fed_extrema"] = {
            "accuracy": {
                "min": float(metrics["accuracy_min"]),
                "max": float(metrics["accuracy_max"]),
            },
            "loss": {
                "min": float(metrics["loss_min"]),
                "max": float(metrics["loss_max"]),
            },
        }

    do_pre_val = _cfg_bool(config, "do_pre_validation", True)
    do_post_val = _cfg_bool(config, "do_validation", True)
    if do_pre_val and do_post_val and stats:
        if all("pre_val_loss" in v and "pre_val_accuracy" in v for v in stats.values()):
            pre_loss_vals = [float(v["pre_val_loss"]) for v in stats.values()]
            pre_acc_vals = [float(v["pre_val_accuracy"]) for v in stats.values()]
            round_metrics["local_pre_eval"] = {
                "accuracy": {
                    "avg": float(np.mean(pre_acc_vals)),
                    "std": float(np.std(pre_acc_vals)),
                },
                "loss": {
                    "avg": float(np.mean(pre_loss_vals)),
                    "std": float(np.std(pre_loss_vals)),
                },
            }
            lines.append(
                _entity_line(
                    "Local Pre-Eval.:",
                    f"accuracy={np.mean(pre_acc_vals):.4f}/{np.std(pre_acc_vals):.4f} | "
                    f"loss={np.mean(pre_loss_vals):.4f}/{np.std(pre_loss_vals):.4f}",
                )
            )
        if all("post_val_loss" in v and "post_val_accuracy" in v for v in stats.values()):
            post_loss_vals = [float(v["post_val_loss"]) for v in stats.values()]
            post_acc_vals = [float(v["post_val_accuracy"]) for v in stats.values()]
            round_metrics["local_post_eval"] = {
                "accuracy": {
                    "avg": float(np.mean(post_acc_vals)),
                    "std": float(np.std(post_acc_vals)),
                },
                "loss": {
                    "avg": float(np.mean(post_loss_vals)),
                    "std": float(np.std(post_loss_vals)),
                },
            }
            lines.append(
                _entity_line(
                    "Local Post-Eval.:",
                    f"accuracy={np.mean(post_acc_vals):.4f}/{np.std(post_acc_vals):.4f} | "
                    f"loss={np.mean(post_loss_vals):.4f}/{np.std(post_loss_vals):.4f}",
                )
            )

    _append_eval_block(
        "Global Eval.", "global_eval", global_eval_metrics, with_client_std=False
    )
    _append_eval_block(
        "Federated Eval.", "fed_eval", federated_eval_metrics, with_client_std=True
    )
    _append_eval_block(
        "Fed Eval In.", "fed_eval_in", federated_eval_in_metrics, with_client_std=True
    )
    _append_eval_block(
        "Fed Eval Out.",
        "fed_eval_out",
        federated_eval_out_metrics,
        with_client_std=True,
    )
    if federated_eval_metrics is not None:
        _append_federated_extrema(federated_eval_metrics)
    elif federated_eval_in_metrics is not None:
        _append_federated_extrema(federated_eval_in_metrics)

    log = "\n".join(lines)
    if logger is not None:
        logger.info(log, round_label=f"Round {round_idx:04d}")
    else:
        print(log)
    if tracker is not None:
        tracker.log_metrics(step=round_idx, metrics=round_metrics)


def _new_server_logger(config: DictConfig, mode: str) -> ServerAgentFileLogger:
    del mode
    run_ts = str(config.get("run_timestamp", "")).strip()
    run_dir = Path(str(config.log_dir)) / str(config.exp_name) / run_ts
    return ServerAgentFileLogger(
        file_dir=str(run_dir),
        file_name="server.log",
        experiment_id=str(config.exp_name),
    )


def _resolve_run_timestamp(config: DictConfig, preset: Optional[str] = None) -> str:
    run_ts = str(preset if preset is not None else config.get("run_timestamp", "")).strip()
    if run_ts == "":
        run_ts = datetime.now().strftime("%Y%m%d%H%M%S")
    config.run_timestamp = run_ts
    return run_ts


def _start_summary_lines(
    mode: str,
    config: DictConfig,
    num_clients: int,
    train_client_count: int,
    holdout_client_count: int,
) -> str:
    return "\n".join(
        [
            f"Start {mode.upper()} simulation",
            f"  * Experiment: {config.exp_name}",
            f"  * Algorithm: {config.algorithm}",
            f"  * Dataset: {config.dataset}",
            f"  * Rounds: {config.num_rounds}",
            f"  * Total Clients: {num_clients}",
            f"  * Per-round Clients: {train_client_count}",
            f"  * Evaluation Scheme: {config.get('federated_eval_scheme', 'holdout_dataset')}",
            f"  * Holdout Clients (evaluation): {holdout_client_count}\n" \
                if config.get('federated_eval_scheme', 'holdout_dataset') == 'holdout_client' else ''
        ]
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
    loss_vals = [float(v.get("loss", -1.0)) for v in stats.values() if "loss" in v]
    acc_vals = [float(v.get("accuracy", -1.0)) for v in stats.values() if "accuracy" in v]
    if total_examples <= 0:
        return {
            "loss": -1.0,
            "accuracy": -1.0,
            "loss_std": -1.0,
            "accuracy_std": -1.0,
            "num_examples": 0,
            "num_clients": len(stats),
            "loss_min": -1.0,
            "loss_max": -1.0,
            "accuracy_min": -1.0,
            "accuracy_max": -1.0,
        }
    return {
        "loss": float(_weighted_mean(stats, "loss")),
        "accuracy": float(_weighted_mean(stats, "accuracy")),
        "loss_std": float(np.std(loss_vals)) if loss_vals else 0.0,
        "accuracy_std": float(np.std(acc_vals)) if acc_vals else 0.0,
        "num_examples": int(total_examples),
        "num_clients": int(len(stats)),
        "loss_min": float(min(loss_vals)) if loss_vals else -1.0,
        "loss_max": float(max(loss_vals)) if loss_vals else -1.0,
        "accuracy_min": float(min(acc_vals)) if acc_vals else -1.0,
        "accuracy_max": float(max(acc_vals)) if acc_vals else -1.0,
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
        state = payload["state"]
        if isinstance(state, tuple):
            state = state[0]
        local_states[int(cid)] = state
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
    start_msg = _start_summary_lines(
        mode="mpi",
        config=config,
        num_clients=int(server.num_clients),
        train_client_count=len(train_client_ids),
        holdout_client_count=len(holdout_client_ids),
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
            len(train_client_ids),
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
    finish_msg = f"Finished MPI simulation in {time.time() - t0:.2f}s."
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
                state = client.upload()
                if isinstance(state, tuple):
                    state = state[0]
                local_payload[int(client.id)] = {
                    "state": state,
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
    run_ts = _resolve_run_timestamp(config)
    run_log_dir = str(Path(str(config.log_dir)) / str(config.exp_name) / run_ts)
    server_logger = _new_server_logger(config, mode="serial")
    tracker = create_experiment_tracker(config)

    loader_cfg = _cfg_to_dict(config)
    loader_cfg["download"] = True
    _, client_datasets, server_dataset, args = load_dataset(loader_cfg)

    runtime_cfg = _merge_runtime_cfg(config, args)
    _validate_loader_output(client_datasets, runtime_cfg)
    num_clients = int(runtime_cfg["num_clients"])
    logging_policy = _resolve_client_logging_policy(config, num_clients=num_clients)
    _emit_logging_policy_message(
        logging_policy, num_clients=num_clients, logger=server_logger
    )
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
        run_log_dir=run_log_dir,
        client_logging_enabled=bool(logging_policy["client_logging_enabled"]),
    )

    server_logger.info(
        _start_summary_lines(
            mode="serial",
            config=config,
            num_clients=num_clients,
            train_client_count=len(train_client_ids),
            holdout_client_count=len(holdout_client_ids),
        )
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
            state = client.upload()
            if isinstance(state, tuple):
                state = state[0]
            updates[client.id] = state
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
            len(train_client_ids),
            stats,
            weights,
            global_eval_metrics=global_eval_metrics,
            federated_eval_metrics=federated_eval_metrics,
            federated_eval_in_metrics=federated_eval_in_metrics,
            federated_eval_out_metrics=federated_eval_out_metrics,
            logger=server_logger,
            tracker=tracker,
        )

    server_logger.info(f"Finished serial simulation in {time.time() - t0:.2f}s.")
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

    run_ts_root = str(config.get("run_timestamp", "")).strip() if rank == 0 else ""
    run_ts_root = _resolve_run_timestamp(config, preset=run_ts_root) if rank == 0 else ""
    run_ts = communicator.comm.bcast(run_ts_root, root=0)
    _resolve_run_timestamp(config, preset=run_ts)
    run_log_dir = str(Path(str(config.log_dir)) / str(config.exp_name) / run_ts)

    set_seed_everything(int(config.seed))
    _, client_datasets, server_dataset, args = _load_dataset_mpi(
        config=config, communicator=communicator, rank=rank
    )

    runtime_cfg = _merge_runtime_cfg(config, args)
    _validate_loader_output(client_datasets, runtime_cfg)

    num_clients = int(runtime_cfg["num_clients"])
    logging_policy = _resolve_client_logging_policy(config, num_clients=num_clients)
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
        _emit_logging_policy_message(
            logging_policy, num_clients=num_clients, logger=server_logger
        )
        server_logger.info(
            f"MPI context: world_size={world_size} rank={rank} local_rank={local_rank} "
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
            f"MPI rank mapping: rank={rank} local_rank={local_rank} "
            f"client_device={client_device} num_local_clients={len(local_client_ids)}"
        )

    clients = _build_clients(
        config=config,
        model=model,
        client_datasets=client_datasets,
        local_client_ids=local_client_ids,
        device=client_device,
        run_log_dir=run_log_dir,
        client_logging_enabled=bool(logging_policy["client_logging_enabled"]),
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
