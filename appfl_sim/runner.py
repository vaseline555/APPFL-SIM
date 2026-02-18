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
from appfl_sim.misc.system_utils import get_local_rank, resolve_rank_device, set_seed_everything
from appfl_sim.misc.config_utils import (
    _allow_reusable_on_demand_pool,
    _cfg_bool,
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
    _dataset_has_eval_split,
    _sample_train_clients,
    _validate_loader_output,
)
from appfl_sim.misc.learning_utils import (
    _aggregate_eval_stats,
    _build_federated_eval_plan,
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
    _warn_if_workers_pinned_to_single_gpu,
)
from appfl_sim.misc.mpi_utils import (
    _load_dataset_mpi,
    _maybe_autolaunch_mpi,
    _mpi_download_mode,
    _run_client_mpi,
    _run_server_mpi,
)
from appfl_sim.misc.runtime_utils import (
    _build_clients,
    _build_on_demand_worker_pool,
    _build_server,
    _rebind_client_for_on_demand_job,
)
from appfl_sim.misc.system_utils import (
    _client_processing_chunk_size,
    _iter_id_chunks,
    _maybe_force_server_cpu,
    _release_clients,
)

def run_serial(config) -> None:
    if not isinstance(config, DictConfig):
        config = OmegaConf.create(_cfg_to_dict(config))

    set_seed_everything(int(config.seed))
    t0 = time.time()
    run_ts = _resolve_run_timestamp(config, preset=None)
    run_log_dir = str(Path(str(config.log_dir)) / str(config.exp_name) / run_ts)
    server_logger = _new_server_logger(config, mode="serial", run_ts=run_ts)
    tracker = create_experiment_tracker(config, run_timestamp=run_ts)

    loader_cfg = _cfg_to_dict(config)
    # Respect user-configured download policy in serial mode.
    loader_cfg["download"] = _cfg_bool(config, "download", True)
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
    enable_global_eval = _cfg_bool(config, "enable_global_eval", True) and _dataset_has_eval_split(
        server_dataset
    )
    _maybe_force_server_cpu(config, enable_global_eval, logger=server_logger)
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
        algorithm_components=algorithm_components,
    )

    client_device = resolve_rank_device(str(config.device), rank=1, world_size=2)
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

    for round_idx in range(1, int(config.num_rounds) + 1):
        selected_ids = _sample_train_clients(
            train_client_ids=train_client_ids,
            num_sampled_clients=int(num_sampled_clients),
        )
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
                train_result = client.update(round_idx=round_idx)
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
                    train_result = client.update(round_idx=round_idx)
                    state = client.upload()
                    if isinstance(state, tuple):
                        state = state[0]
                    updates[client.id] = state
                    sample_sizes[client.id] = int(train_result["num_examples"])
                    stats[client.id] = train_result
                if not worker_pool:
                    _release_clients(chunk_clients)

        weights = server.aggregate(updates, sample_sizes)
        global_eval_metrics = None
        if enable_global_eval and _should_eval_round(
            round_idx, int(config.eval_every), int(config.num_rounds)
        ):
            global_eval_metrics = server.evaluate(round_idx=round_idx)

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
            if eager_clients is not None:
                state = server.model.state_dict()
                if plan["scheme"] == "holdout_client":
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
                if plan["scheme"] == "holdout_client":
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
            global_eval_metrics=global_eval_metrics,
            federated_eval_metrics=federated_eval_metrics,
            federated_eval_in_metrics=federated_eval_in_metrics,
            federated_eval_out_metrics=federated_eval_out_metrics,
            logger=server_logger,
            tracker=tracker,
        )

    server_logger.info(f"Finished serial simulation in {time.time() - t0:.2f}s.")
    server_logger.info(f"Saved resulting metrics in a log folder.")
    server_logger.info(f"Good Bye!")
    if eager_clients is not None:
        _release_clients(eager_clients)
    if worker_pool is not None:
        _release_clients(worker_pool)
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

    run_ts_root = _resolve_run_timestamp(config, preset=None) if rank == 0 else ""
    run_ts = communicator.comm.bcast(run_ts_root, root=0)
    run_log_dir = str(Path(str(config.log_dir)) / str(config.exp_name) / run_ts)

    set_seed_everything(int(config.seed))
    bootstrap_logger = _new_server_logger(config, mode=f"mpi-rank{rank}", run_ts=run_ts)
    _, client_datasets, server_dataset, args = _load_dataset_mpi(
        config=config, communicator=communicator, rank=rank, logger=bootstrap_logger
    )
    client_datasets = _apply_holdout_dataset_ratio(
        client_datasets, config=config, logger=bootstrap_logger
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
        config, logger=bootstrap_logger
    )
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
    respect_explicit_cuda_index = _cfg_bool(
        config, "mpi_respect_explicit_cuda_index", False
    )

    if rank == 0:
        server_logger = bootstrap_logger
        _maybe_force_server_cpu(config, enable_global_eval, logger=server_logger)
        _warn_if_workers_pinned_to_single_gpu(
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
        on_demand_workers = _resolve_on_demand_worker_policy(config, logger=server_logger)
        server_logger.info(
            f"MPI context: world_size={world_size} rank={rank} local_rank={local_rank} "
            f"download_mode={_mpi_download_mode(config)}"
        )
        tracker = create_experiment_tracker(config, run_timestamp=run_ts)
        server = _build_server(
            config=config,
            runtime_cfg=runtime_cfg,
            model=model,
            server_dataset=server_dataset,
            algorithm_components=algorithm_components,
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
        (
            str(config.device)
            if (respect_explicit_cuda_index or not use_local_rank_device)
            else (
                "cuda"
                if str(config.device).strip().lower().startswith("cuda:")
                else str(config.device)
            )
        ),
        rank=rank,
        world_size=world_size,
        local_rank=client_local_rank,
    )
    if _cfg_bool(config, "mpi_log_rank_mapping", False):
        print(
            f"MPI rank mapping: rank={rank} local_rank={local_rank} "
            f"client_device={client_device} num_local_clients={len(local_client_ids)}"
        )

    _run_client_mpi(
        communicator=communicator,
        config=config,
        model=model,
        client_datasets=client_datasets,
        local_client_ids=local_client_ids,
        device=client_device,
        run_log_dir=run_log_dir,
        client_logging_enabled=bool(logging_policy["client_logging_enabled"]),
        trainer_name=str(algorithm_components["trainer_name"]),
        use_on_demand=bool(state_policy["use_on_demand"]),
        on_demand_train_num_workers=int(on_demand_workers["train"]),
        on_demand_eval_num_workers=int(on_demand_workers["eval"]),
    )

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
        cfg.backend = backend_override

    backend = str(cfg.get("backend", "mpi")).lower()
    if backend not in {"serial", "mpi"}:
        raise ValueError("backend must be one of: serial, mpi")

    return backend, cfg

def main(argv: list[str] | None = None) -> None:
    cli_argv = list(sys.argv[1:] if argv is None else argv)
    backend, config = parse_config(cli_argv)
    _maybe_autolaunch_mpi(backend=backend, config=config, cli_argv=cli_argv)
    if backend == "serial":
        run_serial(config)
    else:
        run_mpi(config)


if __name__ == "__main__":
    welcome_message = """
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
