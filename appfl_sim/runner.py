from __future__ import annotations

import sys
import copy
import time

import torch
import numpy as np

from pathlib import Path
from typing import List, Optional
from omegaconf import DictConfig, OmegaConf

from appfl_sim.agent import ClientAgent
from appfl_sim.logger import create_experiment_tracker
from appfl_sim.loaders import load_dataset, load_model
from appfl_sim.metrics import parse_metric_names
from appfl_sim.misc.system_utils import (
    get_local_rank,
    resolve_rank_device,
    set_seed_everything,
    validate_backend_device_consistency,
)
from appfl_sim.misc.config_utils import (
    _allow_reusable_on_demand_pool,
    _cfg_bool,
    _cfg_get,
    _cfg_to_dict,
    _default_config_path,
    _ensure_model_cfg,
    _extract_config_path,
    _merge_runtime_cfg,
    _normalize_cli_tokens,
    _resolve_algorithm_components,
    _resolve_client_logging_policy,
    _resolve_client_state_policy,
    _resolve_config_path,
    _resolve_num_sampled_clients,
    _resolve_on_demand_worker_policy,
)
from appfl_sim.misc.data_utils import (
    _apply_holdout_dataset_ratio,
    _build_client_groups,
    _parse_holdout_dataset_ratio,
    _resolve_client_eval_dataset,
    _dataset_has_eval_split,
    _sample_train_clients,
    _validate_loader_output,
)
from appfl_sim.misc.learning_utils import (
    _aggregate_eval_stats,
    _build_federated_eval_plan,
    _evaluate_dataset_direct,
    _run_federated_eval_serial,
    _should_eval_round,
)
from appfl_sim.misc.logging_utils import (
    _emit_client_state_policy_message,
    _emit_federated_eval_policy_message,
    _emit_logging_policy_message,
    _log_round,
    _new_server_logger,
    _resolve_run_timestamp,
    _start_summary_lines,
    _warn_if_workers_pinned_to_single_device,
)
from appfl_sim.misc.runtime_utils import (
    _build_clients,
    _build_on_demand_worker_pool,
    _build_server,
    _rebind_client_for_on_demand_job,
)
from appfl_sim.misc.dist_utils import launch_or_run_distributed
from appfl_sim.misc.system_utils import (
    _client_processing_chunk_size,
    _iter_id_chunks,
    _maybe_force_server_cpu,
    _release_clients,
)


def _maybe_select_round_local_steps(server, round_idx: int):
    scheduler = getattr(server, "scheduler", None)
    if scheduler is None or not hasattr(scheduler, "select_local_steps"):
        return None
    try:
        return int(scheduler.select_local_steps(round_idx=int(round_idx)))
    except Exception:
        return None


def _weighted_global_gen_error(
    stats: dict,
    sample_sizes: dict,
) -> float | None:
    if not stats:
        return None
    total = 0.0
    accum = 0.0
    for cid, client_stats in stats.items():
        if not isinstance(client_stats, dict):
            continue
        if "local_gen_error" not in client_stats:
            continue
        value = client_stats.get("local_gen_error")
        if not isinstance(value, (int, float)):
            continue
        weight = float(sample_sizes.get(cid, 0))
        if weight <= 0.0:
            weight = 1.0
        accum += weight * float(value)
        total += weight
    if total <= 0.0:
        return None
    return float(accum / total)


def _maybe_observe_round_gen_error(server, global_gen_error: float, round_idx: int):
    scheduler = getattr(server, "scheduler", None)
    if scheduler is None or not hasattr(scheduler, "observe_global_gen_error"):
        return None
    try:
        return scheduler.observe_global_gen_error(
            global_gen_error=float(global_gen_error),
            round_idx=int(round_idx),
        )
    except Exception:
        return None


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


def run_serial(config) -> None:
    if not isinstance(config, DictConfig):
        config = OmegaConf.create(_cfg_to_dict(config))
    _validate_bandit_dataset_ratio(config)

    set_seed_everything(int(_cfg_get(config, "experiment.seed", 42)))
    t0 = time.time()
    run_ts = _resolve_run_timestamp(config, preset=None)
    run_log_dir = str(
        Path(str(_cfg_get(config, "logging.path", "./logs")))
        / str(_cfg_get(config, "experiment.name", "appfl-sim"))
        / run_ts
    )
    server_logger = _new_server_logger(config, mode="serial", run_ts=run_ts)
    tracker = create_experiment_tracker(config, run_timestamp=run_ts)

    loader_cfg = _cfg_to_dict(config)
    # Respect user-configured download policy in serial mode.
    loader_cfg["download"] = _cfg_bool(config, "dataset.download", True)
    loader_cfg["logger"] = server_logger
    _, client_datasets, server_dataset, args = load_dataset(loader_cfg)
    client_datasets = _apply_holdout_dataset_ratio(
        client_datasets, config=config, logger=server_logger
    )

    runtime_cfg = _merge_runtime_cfg(config, args)
    _validate_loader_output(client_datasets, runtime_cfg)
    algorithm_components = _resolve_algorithm_components(config)
    num_clients = int(runtime_cfg["num_clients"])
    state_policy = _resolve_client_state_policy(config)
    train_client_ids, holdout_client_ids = _build_client_groups(config, num_clients)
    num_sampled_clients = _resolve_num_sampled_clients(
        config, num_clients=len(train_client_ids)
    )
    logging_policy = _resolve_client_logging_policy(
        config,
        num_clients=num_clients,
        num_sampled_clients=num_sampled_clients,
    )
    _emit_logging_policy_message(
        logging_policy, num_clients=num_clients, logger=server_logger
    )
    _emit_client_state_policy_message(state_policy, logger=server_logger)
    server_logger.info(
        "Algorithm wiring: "
        f"algorithm={algorithm_components['algorithm']} "
        f"aggregator={algorithm_components['aggregator_name']} "
        f"scheduler={algorithm_components['scheduler_name']} "
        f"trainer={algorithm_components['trainer_name']}"
    )
    on_demand_workers = _resolve_on_demand_worker_policy(config, logger=server_logger)
    enable_global_eval = _cfg_bool(config, "eval.enable_global_eval", True) and _dataset_has_eval_split(
        server_dataset
    )
    _maybe_force_server_cpu(config, enable_global_eval, logger=server_logger)
    enable_federated_eval = _cfg_bool(config, "eval.enable_federated_eval", True)

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
        algorithm_components=algorithm_components,
    )

    client_device = resolve_rank_device(
        str(_cfg_get(config, "experiment.device", "cpu")), rank=1, world_size=2
    )
    chunk_size = _client_processing_chunk_size(
        config=config,
        model=model,
        device=client_device,
        total_clients=num_clients,
        phase="train",
    )
    eager_clients = None
    worker_pool: Optional[List[ClientAgent]] = None
    use_on_demand = bool(state_policy["use_on_demand"])
    on_demand_model = copy.deepcopy(model) if use_on_demand else None

    if not use_on_demand:
        eager_clients = _build_clients(
            config=config,
            model=model,
            client_datasets=client_datasets,
            local_client_ids=np.arange(num_clients).astype(int),
            device=client_device,
            run_log_dir=run_log_dir,
            client_logging_enabled=bool(logging_policy["client_logging_enabled"]),
            trainer_name=str(algorithm_components["trainer_name"]),
        )
    else:
        if _allow_reusable_on_demand_pool(
            config,
            client_logging_enabled=bool(logging_policy["client_logging_enabled"]),
        ):
            worker_pool = _build_on_demand_worker_pool(
                config=config,
                model=on_demand_model if on_demand_model is not None else model,
                client_datasets=client_datasets,
                local_client_ids=np.arange(num_clients).astype(int),
                device=client_device,
                run_log_dir=run_log_dir,
                client_logging_enabled=bool(logging_policy["client_logging_enabled"]),
                trainer_name=str(algorithm_components["trainer_name"]),
                pool_size=chunk_size,
                num_workers_override=on_demand_workers["train"],
            )

    server_logger.info(
        _start_summary_lines(
            mode="serial",
            config=config,
            num_clients=num_clients,
            train_client_count=len(train_client_ids),
            holdout_client_count=len(holdout_client_ids),
            num_sampled_clients=num_sampled_clients,
        )
    )
    _emit_federated_eval_policy_message(
        config=config,
        train_client_count=len(train_client_ids),
        holdout_client_count=len(holdout_client_ids),
        logger=server_logger,
    )

    interrupted = False
    try:
        for round_idx in range(1, int(_cfg_get(config, "train.num_rounds", 20)) + 1):
            selected_ids = _sample_train_clients(
                train_client_ids=train_client_ids,
                num_sampled_clients=int(num_sampled_clients),
            )
            round_local_steps = _maybe_select_round_local_steps(server, round_idx)
            global_state = server.model.state_dict()

            updates = {}
            sample_sizes = {}
            stats = {}
            if eager_clients is not None:
                selected = set(selected_ids)
                for client in eager_clients:
                    if client.id not in selected:
                        continue
                    client.download(global_state)
                    if round_local_steps is None:
                        train_result = client.update(round_idx=round_idx)
                    else:
                        train_result = client.update(
                            round_idx=round_idx, local_steps=int(round_local_steps)
                        )
                    state = client.upload()
                    if isinstance(state, tuple):
                        state = state[0]
                    updates[client.id] = state
                    sample_sizes[client.id] = int(train_result["num_examples"])
                    stats[client.id] = train_result
            else:
                for chunk_ids in _iter_id_chunks(selected_ids, chunk_size):
                    if worker_pool:
                        chunk_clients = worker_pool[: len(chunk_ids)]
                        for client, cid in zip(chunk_clients, chunk_ids):
                            _rebind_client_for_on_demand_job(
                                client,
                                client_id=int(cid),
                                client_datasets=client_datasets,
                                num_workers_override=on_demand_workers["train"],
                            )
                    else:
                        chunk_clients = _build_clients(
                            config=config,
                            model=on_demand_model if on_demand_model is not None else model,
                            client_datasets=client_datasets,
                            local_client_ids=np.asarray(chunk_ids).astype(int),
                            device=client_device,
                            run_log_dir=run_log_dir,
                            client_logging_enabled=bool(logging_policy["client_logging_enabled"]),
                            trainer_name=str(algorithm_components["trainer_name"]),
                            share_model=True,
                            num_workers_override=on_demand_workers["train"],
                        )
                    for client in chunk_clients:
                        client.download(global_state)
                        if round_local_steps is None:
                            train_result = client.update(round_idx=round_idx)
                        else:
                            train_result = client.update(
                                round_idx=round_idx, local_steps=int(round_local_steps)
                            )
                        state = client.upload()
                        if isinstance(state, tuple):
                            state = state[0]
                        updates[client.id] = state
                        sample_sizes[client.id] = int(train_result["num_examples"])
                        stats[client.id] = train_result
                    if not worker_pool:
                        _release_clients(chunk_clients)

            weights = server.aggregate(updates, sample_sizes)
            round_gen_error = _weighted_global_gen_error(stats, sample_sizes)
            if round_gen_error is not None:
                _maybe_observe_round_gen_error(
                    server, global_gen_error=round_gen_error, round_idx=round_idx
                )
            global_eval_metrics = None
            if enable_global_eval and _should_eval_round(
                round_idx,
                int(_cfg_get(config, "eval.every", 1)),
                int(_cfg_get(config, "train.num_rounds", 20)),
            ):
                global_eval_metrics = server.evaluate(round_idx=round_idx)

            federated_eval_metrics = None
            federated_eval_in_metrics = None
            federated_eval_out_metrics = None
            if enable_federated_eval:
                plan = _build_federated_eval_plan(
                    config=config,
                    round_idx=round_idx,
                    num_rounds=int(_cfg_get(config, "train.num_rounds", 20)),
                    selected_train_ids=selected_ids,
                    train_client_ids=train_client_ids,
                    holdout_client_ids=holdout_client_ids,
                )
                if eager_clients is not None:
                    state = server.model.state_dict()
                    if plan["scheme"] == "client":
                        eval_in_set = set(plan["in_ids"])
                        eval_out_set = set(plan["out_ids"])
                        eval_in_stats = {}
                        eval_out_stats = {}
                        for client in eager_clients:
                            if client.id in eval_in_set:
                                client.download(state)
                                eval_in_stats[int(client.id)] = client.evaluate(
                                    split="test",
                                    offload_after=False,
                                )
                            elif client.id in eval_out_set:
                                client.download(state)
                                eval_out_stats[int(client.id)] = client.evaluate(
                                    split="test",
                                    offload_after=False,
                                )
                        federated_eval_in_metrics = _aggregate_eval_stats(eval_in_stats)
                        federated_eval_out_metrics = _aggregate_eval_stats(eval_out_stats)
                    else:
                        eval_set = set(plan["in_ids"])
                        eval_stats = {}
                        for client in eager_clients:
                            if client.id not in eval_set:
                                continue
                            client.download(state)
                            eval_stats[int(client.id)] = client.evaluate(
                                split="test",
                                offload_after=False,
                            )
                        federated_eval_metrics = _aggregate_eval_stats(eval_stats)
                else:
                    if plan["scheme"] == "client":
                        federated_eval_in_metrics = _run_federated_eval_serial(
                            config=config,
                            model=on_demand_model if on_demand_model is not None else model,
                            client_datasets=client_datasets,
                            run_log_dir=run_log_dir,
                            client_logging_enabled=bool(
                                logging_policy["client_logging_enabled"]
                            ),
                            device=client_device,
                            global_state=server.model.state_dict(),
                            eval_client_ids=list(plan["in_ids"]),
                            round_idx=round_idx,
                            eval_tag="fed-in",
                            eval_split="test",
                            eval_num_workers_override=on_demand_workers["eval"],
                        )
                        federated_eval_out_metrics = _run_federated_eval_serial(
                            config=config,
                            model=on_demand_model if on_demand_model is not None else model,
                            client_datasets=client_datasets,
                            run_log_dir=run_log_dir,
                            client_logging_enabled=bool(
                                logging_policy["client_logging_enabled"]
                            ),
                            device=client_device,
                            global_state=server.model.state_dict(),
                            eval_client_ids=list(plan["out_ids"]),
                            round_idx=round_idx,
                            eval_tag="fed-out",
                            eval_split="test",
                            eval_num_workers_override=on_demand_workers["eval"],
                        )
                    else:
                        federated_eval_metrics = _run_federated_eval_serial(
                            config=config,
                            model=on_demand_model if on_demand_model is not None else model,
                            client_datasets=client_datasets,
                            run_log_dir=run_log_dir,
                            client_logging_enabled=bool(
                                logging_policy["client_logging_enabled"]
                            ),
                            device=client_device,
                            global_state=server.model.state_dict(),
                            eval_client_ids=list(plan["in_ids"]),
                            round_idx=round_idx,
                            eval_tag="fed",
                            eval_split="test",
                            eval_num_workers_override=on_demand_workers["eval"],
                        )

            _log_round(
                config,
                round_idx,
                len(selected_ids),
                len(train_client_ids),
                stats,
                weights,
                round_local_steps=round_local_steps,
                global_eval_metrics=global_eval_metrics,
                federated_eval_metrics=federated_eval_metrics,
                federated_eval_in_metrics=federated_eval_in_metrics,
                federated_eval_out_metrics=federated_eval_out_metrics,
                logger=server_logger,
                tracker=tracker,
            )
    except KeyboardInterrupt:
        interrupted = True
        server_logger.info("Interrupted by user; shutting down serial backend.")
    finally:
        if eager_clients is not None:
            _release_clients(eager_clients)
        if worker_pool is not None:
            _release_clients(worker_pool)
        if tracker is not None:
            tracker.close()

    if interrupted:
        raise SystemExit(130)
    server_logger.info(f"Finished serial simulation in {time.time() - t0:.2f}s.")
    server_logger.info(f"Saved resulting metrics in a log folder.")
    server_logger.info(f"Good Bye!")


def _load_dataset_distributed(
    config: DictConfig,
    rank: int,
    logger=None,
):
    import torch.distributed as dist

    def _set_download(cfg_dict: dict, value: bool) -> None:
        ds = cfg_dict.get("dataset", None)
        if not isinstance(ds, dict):
            ds = {}
            cfg_dict["dataset"] = ds
        ds["download"] = bool(value)

    loader_cfg = _cfg_to_dict(config)
    if logger is not None:
        loader_cfg["logger"] = logger
    if rank == 0:
        cfg_root = dict(loader_cfg)
        _set_download(cfg_root, True)
        cached = load_dataset(cfg_root)
    else:
        cached = None
    dist.barrier()
    if rank == 0:
        return cached
    cfg_other = dict(loader_cfg)
    _set_download(cfg_other, False)
    return load_dataset(cfg_other)


def _run_federated_eval_distributed(
    config: DictConfig,
    model,
    client_datasets,
    device: str,
    global_state,
    eval_client_ids: list[int],
    rank: int,
    world_size: int,
    eval_split: str = "test",
):
    import torch.distributed as dist

    if not eval_client_ids:
        # All ranks must still join the collective.
        if rank == 0:
            gathered: list[dict] = [None] * world_size
            dist.gather_object({}, object_gather_list=gathered, dst=0)
        else:
            dist.gather_object({}, object_gather_list=None, dst=0)
        return None if rank != 0 else None

    eval_model = model.to(device)
    eval_model.load_state_dict(global_state)
    eval_model.eval()
    eval_loss_fn = getattr(
        torch.nn, str(_cfg_get(config, "optimization.criterion", "CrossEntropyLoss"))
    )().to(device)
    eval_metric_names = parse_metric_names(_cfg_get(config, "eval.metrics", ["acc1"]))
    eval_batch_size = int(
        _cfg_get(config, "train.eval_batch_size", _cfg_get(config, "train.batch_size", 32))
    )
    eval_workers = max(0, int(_cfg_get(config, "train.num_workers", 0)))
    local_client_set = {
        int(cid)
        for cid in np.asarray(np.array_split(np.arange(len(client_datasets)), world_size)[rank]).astype(int)
    }
    local_stats = {}
    for client_id in sorted(int(cid) for cid in eval_client_ids):
        if client_id not in local_client_set:
            continue
        eval_ds = _resolve_client_eval_dataset(
            client_datasets=client_datasets,
            client_id=int(client_id),
            eval_split=str(eval_split),
        )
        local_stats[int(client_id)] = _evaluate_dataset_direct(
            model=eval_model,
            dataset=eval_ds,
            device=device,
            loss_fn=eval_loss_fn,
            eval_metric_names=eval_metric_names,
            batch_size=eval_batch_size,
            num_workers=eval_workers,
        )

    if rank == 0:
        gathered = [None] * world_size
        dist.gather_object(local_stats, object_gather_list=gathered, dst=0)
    else:
        dist.gather_object(local_stats, object_gather_list=None, dst=0)
        gathered = None
    if rank != 0:
        return None

    merged = {}
    for payload in gathered or []:
        if isinstance(payload, dict):
            merged.update(payload)
    return _aggregate_eval_stats(merged)


def _broadcast_model_state_inplace(model, *, src: int = 0) -> None:
    import torch.distributed as dist

    state = model.state_dict()
    backend = str(dist.get_backend()).strip().lower()
    for key in sorted(state.keys()):
        tensor = state[key]
        if not torch.is_tensor(tensor):
            continue
        if backend == "nccl" and tensor.device.type == "cpu":
            # NCCL does not support CPU tensors. Broadcast via CUDA staging buffer.
            device = torch.device("cuda", torch.cuda.current_device())
            with torch.no_grad():
                staged = tensor.detach().to(device=device, non_blocking=True)
                dist.broadcast(staged, src=src)
                tensor.copy_(staged.to(device="cpu", non_blocking=False))
            del staged
            continue
        dist.broadcast(tensor, src=src)


def run_distributed(config, backend: str) -> None:
    import torch.distributed as dist

    if not isinstance(config, DictConfig):
        config = OmegaConf.create(_cfg_to_dict(config))
    _validate_bandit_dataset_ratio(config)
    if backend not in {"nccl", "gloo"}:
        raise ValueError("backend must be one of: serial, nccl, gloo")
    if not dist.is_initialized():
        raise RuntimeError("torch.distributed process group is not initialized.")

    rank = int(dist.get_rank())
    world_size = int(dist.get_world_size())
    local_rank = get_local_rank(default=rank)

    if backend == "nccl":
        if not torch.cuda.is_available():
            raise RuntimeError("backend=nccl requires CUDA.")
        torch.cuda.set_device(local_rank % max(1, torch.cuda.device_count()))

    set_seed_everything(int(_cfg_get(config, "experiment.seed", 42)) + rank)
    run_ts_root = _resolve_run_timestamp(config, preset=None) if rank == 0 else ""
    run_ts_payload = [run_ts_root]
    dist.broadcast_object_list(run_ts_payload, src=0)
    run_ts = str(run_ts_payload[0])
    run_log_dir = str(
        Path(str(_cfg_get(config, "logging.path", "./logs")))
        / str(_cfg_get(config, "experiment.name", "appfl-sim"))
        / run_ts
    )

    bootstrap_logger = (
        _new_server_logger(config, mode=f"{backend}-rank{rank}", run_ts=run_ts)
        if rank == 0
        else None
    )
    _, client_datasets, server_dataset, args = _load_dataset_distributed(
        config=config,
        rank=rank,
        logger=bootstrap_logger if rank == 0 else None,
    )
    client_datasets = _apply_holdout_dataset_ratio(
        client_datasets, config=config, logger=bootstrap_logger if rank == 0 else None
    )

    runtime_cfg = _merge_runtime_cfg(config, args)
    _validate_loader_output(client_datasets, runtime_cfg)
    algorithm_components = _resolve_algorithm_components(config)
    num_clients = int(runtime_cfg["num_clients"])
    state_policy = _resolve_client_state_policy(config)
    train_client_ids, holdout_client_ids = _build_client_groups(config, num_clients)
    num_sampled_clients = _resolve_num_sampled_clients(
        config, num_clients=len(train_client_ids)
    )
    logging_policy = _resolve_client_logging_policy(
        config,
        num_clients=num_clients,
        num_sampled_clients=num_sampled_clients,
    )
    on_demand_workers = _resolve_on_demand_worker_policy(
        config, logger=bootstrap_logger if rank == 0 else None
    )
    enable_global_eval = _cfg_bool(config, "eval.enable_global_eval", True) and _dataset_has_eval_split(
        server_dataset
    )
    enable_federated_eval = _cfg_bool(config, "eval.enable_federated_eval", True)
    model = load_model(
        runtime_cfg,
        input_shape=tuple(runtime_cfg["input_shape"]),
        num_classes=int(runtime_cfg["num_classes"]),
    )

    client_device = resolve_rank_device(
        str(_cfg_get(config, "experiment.device", "cpu")),
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
    )
    if backend == "nccl" and not str(client_device).startswith("cuda"):
        client_device = f"cuda:{local_rank % max(1, torch.cuda.device_count())}"

    client_groups = np.array_split(np.arange(num_clients), world_size)
    local_client_ids = np.asarray(client_groups[rank]).astype(int)
    local_client_set = {int(cid) for cid in local_client_ids}

    chunk_size = _client_processing_chunk_size(
        config=config,
        model=model,
        device=client_device,
        total_clients=max(1, len(local_client_ids)),
        phase="train",
    )
    eager_clients = None
    worker_pool: Optional[List[ClientAgent]] = None
    use_on_demand = bool(state_policy["use_on_demand"])
    on_demand_model = copy.deepcopy(model) if use_on_demand else None
    # Under NCCL backend, keep logging artifacts only on rank0 to reduce I/O noise.
    local_client_logging_enabled = bool(logging_policy["client_logging_enabled"])
    if backend == "nccl" and rank != 0:
        local_client_logging_enabled = False

    if not use_on_demand:
        eager_clients = _build_clients(
            config=config,
            model=model,
            client_datasets=client_datasets,
            local_client_ids=local_client_ids,
            device=client_device,
            run_log_dir=run_log_dir,
            client_logging_enabled=local_client_logging_enabled,
            trainer_name=str(algorithm_components["trainer_name"]),
        )
    else:
        if _allow_reusable_on_demand_pool(
            config,
            client_logging_enabled=bool(logging_policy["client_logging_enabled"]),
        ):
            worker_pool = _build_on_demand_worker_pool(
                config=config,
                model=on_demand_model if on_demand_model is not None else model,
                client_datasets=client_datasets,
                local_client_ids=local_client_ids,
                device=client_device,
                run_log_dir=run_log_dir,
                client_logging_enabled=local_client_logging_enabled,
                trainer_name=str(algorithm_components["trainer_name"]),
                pool_size=chunk_size,
                num_workers_override=on_demand_workers["train"],
            )

    tracker = None
    server = None
    server_logger = bootstrap_logger if rank == 0 else None
    if rank == 0:
        _maybe_force_server_cpu(config, enable_global_eval, logger=server_logger)
        _warn_if_workers_pinned_to_single_device(
            config=config,
            world_size=world_size,
            logger=server_logger,
        )
        _emit_logging_policy_message(
            logging_policy, num_clients=num_clients, logger=server_logger
        )
        _emit_client_state_policy_message(state_policy, logger=server_logger)
        server_logger.info(
            "Algorithm wiring: "
            f"algorithm={algorithm_components['algorithm']} "
            f"aggregator={algorithm_components['aggregator_name']} "
            f"scheduler={algorithm_components['scheduler_name']} "
            f"trainer={algorithm_components['trainer_name']}"
        )
        server_logger.info(
            f"Distributed context: backend={backend} world_size={world_size} rank={rank} local_rank={local_rank}"
        )
        server_logger.info(
            _start_summary_lines(
                mode=backend,
                config=config,
                num_clients=num_clients,
                train_client_count=len(train_client_ids),
                holdout_client_count=len(holdout_client_ids),
                num_sampled_clients=num_sampled_clients,
            )
        )
        _emit_federated_eval_policy_message(
            config=config,
            train_client_count=len(train_client_ids),
            holdout_client_count=len(holdout_client_ids),
            logger=server_logger,
        )
        tracker = create_experiment_tracker(config, run_timestamp=run_ts)
        server = _build_server(
            config=config,
            runtime_cfg=runtime_cfg,
            model=model,
            server_dataset=server_dataset,
            algorithm_components=algorithm_components,
        )
    # Ensure every rank starts from the exact same global model state.
    sync_model = server.model if rank == 0 else model
    _broadcast_model_state_inplace(sync_model, src=0)
    if on_demand_model is not None and sync_model is not on_demand_model:
        on_demand_model.load_state_dict(sync_model.state_dict())

    t0 = time.time()
    interrupted = False
    try:
        for round_idx in range(1, int(_cfg_get(config, "train.num_rounds", 20)) + 1):
            if rank == 0:
                selected_ids = _sample_train_clients(
                    train_client_ids=train_client_ids,
                    num_sampled_clients=int(num_sampled_clients),
                )
                local_steps = _maybe_select_round_local_steps(server, round_idx)
                payload = {"selected_ids": selected_ids, "local_steps": local_steps}
            else:
                payload = None
            bcast_obj = [payload]
            dist.broadcast_object_list(bcast_obj, src=0)
            payload = bcast_obj[0]
            selected_ids = list(payload["selected_ids"])
            round_local_steps = payload.get("local_steps", None)
            sync_model = server.model if rank == 0 else model
            _broadcast_model_state_inplace(sync_model, src=0)
            if on_demand_model is not None and sync_model is not on_demand_model:
                on_demand_model.load_state_dict(sync_model.state_dict())
            global_state = sync_model.state_dict()
    
            selected_local_ids = sorted(int(cid) for cid in selected_ids if int(cid) in local_client_set)
            local_payload = {}
            if eager_clients is not None:
                selected_set = set(selected_local_ids)
                for client in eager_clients:
                    if client.id not in selected_set:
                        continue
                    client.download(global_state)
                    if round_local_steps is None:
                        train_result = client.update(round_idx=round_idx)
                    else:
                        train_result = client.update(
                            round_idx=round_idx, local_steps=int(round_local_steps)
                        )
                    state = client.upload()
                    if isinstance(state, tuple):
                        state = state[0]
                    local_payload[int(client.id)] = {
                        "state": state,
                        "num_examples": int(train_result.get("num_examples", 0)),
                        "stats": train_result,
                    }
            else:
                for chunk_ids in _iter_id_chunks(selected_local_ids, chunk_size):
                    if worker_pool:
                        chunk_clients = worker_pool[: len(chunk_ids)]
                        for client, cid in zip(chunk_clients, chunk_ids):
                            _rebind_client_for_on_demand_job(
                                client,
                                client_id=int(cid),
                                client_datasets=client_datasets,
                                num_workers_override=on_demand_workers["train"],
                            )
                    else:
                        chunk_clients = _build_clients(
                            config=config,
                            model=on_demand_model if on_demand_model is not None else model,
                            client_datasets=client_datasets,
                            local_client_ids=np.asarray(chunk_ids).astype(int),
                            device=client_device,
                            run_log_dir=run_log_dir,
                            client_logging_enabled=local_client_logging_enabled,
                            trainer_name=str(algorithm_components["trainer_name"]),
                            share_model=True,
                            num_workers_override=on_demand_workers["train"],
                        )
                    for client in chunk_clients:
                        client.download(global_state)
                        if round_local_steps is None:
                            train_result = client.update(round_idx=round_idx)
                        else:
                            train_result = client.update(
                                round_idx=round_idx, local_steps=int(round_local_steps)
                            )
                        state = client.upload()
                        if isinstance(state, tuple):
                            state = state[0]
                        local_payload[int(client.id)] = {
                            "state": state,
                            "num_examples": int(train_result.get("num_examples", 0)),
                            "stats": train_result,
                        }
                    if not worker_pool:
                        _release_clients(chunk_clients)
    
            if rank == 0:
                gathered = [None] * world_size
                dist.gather_object(local_payload, object_gather_list=gathered, dst=0)
            else:
                dist.gather_object(local_payload, object_gather_list=None, dst=0)
                gathered = None
            weights = None
            stats = {}
            if rank == 0:
                updates = {}
                sample_sizes = {}
                for payload_map in gathered or []:
                    if not isinstance(payload_map, dict):
                        continue
                    for cid, payload_item in payload_map.items():
                        state = payload_item["state"]
                        if isinstance(state, tuple):
                            state = state[0]
                        updates[int(cid)] = state
                        sample_sizes[int(cid)] = int(payload_item.get("num_examples", 0))
                        stats[int(cid)] = payload_item.get("stats", {})
                weights = server.aggregate(updates, sample_sizes)
                round_gen_error = _weighted_global_gen_error(stats, sample_sizes)
                if round_gen_error is not None:
                    _maybe_observe_round_gen_error(
                        server, global_gen_error=round_gen_error, round_idx=round_idx
                    )
    
            sync_model = server.model if rank == 0 else model
            _broadcast_model_state_inplace(sync_model, src=0)
            if on_demand_model is not None and sync_model is not on_demand_model:
                on_demand_model.load_state_dict(sync_model.state_dict())
            next_global_state = sync_model.state_dict()
    
            global_eval_metrics = None
            if rank == 0 and enable_global_eval and _should_eval_round(
                round_idx,
                int(_cfg_get(config, "eval.every", 1)),
                int(_cfg_get(config, "train.num_rounds", 20)),
            ):
                global_eval_metrics = server.evaluate(round_idx=round_idx)
    
            plan_payload = [None]
            if rank == 0 and enable_federated_eval:
                plan_payload[0] = _build_federated_eval_plan(
                    config=config,
                    round_idx=round_idx,
                    num_rounds=int(_cfg_get(config, "train.num_rounds", 20)),
                    selected_train_ids=selected_ids,
                    train_client_ids=train_client_ids,
                    holdout_client_ids=holdout_client_ids,
                )
            dist.broadcast_object_list(plan_payload, src=0)
            plan = plan_payload[0]
    
            federated_eval_metrics = None
            federated_eval_in_metrics = None
            federated_eval_out_metrics = None
            if enable_federated_eval and isinstance(plan, dict):
                if plan["scheme"] == "client":
                    federated_eval_in_metrics = _run_federated_eval_distributed(
                        config=config,
                        model=on_demand_model if on_demand_model is not None else model,
                        client_datasets=client_datasets,
                        device=client_device,
                        global_state=next_global_state,
                        eval_client_ids=list(plan["in_ids"]),
                        rank=rank,
                        world_size=world_size,
                        eval_split="test",
                    )
                    federated_eval_out_metrics = _run_federated_eval_distributed(
                        config=config,
                        model=on_demand_model if on_demand_model is not None else model,
                        client_datasets=client_datasets,
                        device=client_device,
                        global_state=next_global_state,
                        eval_client_ids=list(plan["out_ids"]),
                        rank=rank,
                        world_size=world_size,
                        eval_split="test",
                    )
                else:
                    federated_eval_metrics = _run_federated_eval_distributed(
                        config=config,
                        model=on_demand_model if on_demand_model is not None else model,
                        client_datasets=client_datasets,
                        device=client_device,
                        global_state=next_global_state,
                        eval_client_ids=list(plan["in_ids"]),
                        rank=rank,
                        world_size=world_size,
                        eval_split="test",
                    )
                if rank == 0:
                    _log_round(
                        config,
                        round_idx,
                        len(selected_ids),
                        len(train_client_ids),
                        stats,
                        weights,
                        round_local_steps=round_local_steps,
                        global_eval_metrics=global_eval_metrics,
                        federated_eval_metrics=federated_eval_metrics,
                        federated_eval_in_metrics=federated_eval_in_metrics,
                        federated_eval_out_metrics=federated_eval_out_metrics,
                        logger=server_logger,
                        tracker=tracker,
                    )
    except KeyboardInterrupt:
        interrupted = True
        if rank == 0 and server_logger is not None:
            server_logger.info(
                f"Interrupted by user; shutting down distributed backend ({backend})."
            )
    finally:
        if eager_clients is not None:
            _release_clients(eager_clients)
        if worker_pool is not None:
            _release_clients(worker_pool)
        if tracker is not None:
            tracker.close()
        if dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception:
                pass

    if interrupted:
        raise SystemExit(130)
    if rank == 0 and server_logger is not None:
        server_logger.info(f"Finished {backend} simulation in {time.time() - t0:.2f}s.")
        server_logger.info("Saved resulting metrics in a log folder.")
        server_logger.info("Good Bye!")


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
    _ensure_model_cfg(cfg)
    if backend_override is not None:
        cfg.experiment.backend = backend_override

    backend = str(_cfg_get(cfg, "experiment.backend", "serial")).lower()
    if backend not in {"serial", "nccl", "gloo"}:
        raise ValueError("backend must be one of: serial, nccl, gloo")

    return backend, cfg


def main(argv: list[str] | None = None) -> None:
    cli_argv = list(sys.argv[1:] if argv is None else argv)
    backend, config = parse_config(cli_argv)
    validate_backend_device_consistency(backend=backend, config=config)
    if backend == "serial":
        run_serial(config)
    else:
        launch_or_run_distributed(backend=backend, config=config, entry_fn=run_distributed)


if __name__ == "__main__":
    welcome_message = r"""
                  _____  _____  ______ _          _____ _____ __  __ 
            /\   |  __ \|  __ \|  ____| |        / ____|_   _|  \/  |
           /  \  | |__) | |__) | |__  | |  _____| (___   | | | \  / |
          / /\ \ |  ___/|  ___/|  __| | | |______\___ \  | | | |\/| |
         / ____ \| |    | |    | |    | |____    ____) |_| |_| |  | |
        /_/    \_\_|    |_|    |_|    |______|  |_____/|_____|_|  |_|
                                                                    
    Copyright © 2022-2026, UChicago Argonne, LLC and the APPFL Development Team
    """
    print(welcome_message)
    main()
