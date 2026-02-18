from __future__ import annotations
import copy
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from omegaconf import DictConfig
from appfl_sim.agent import ClientAgent, ServerAgent
from appfl_sim.logger import ServerAgentFileLogger
from appfl_sim.loaders import load_dataset
from appfl_sim.metrics import parse_metric_names
from appfl_sim.misc.config_utils import _allow_reusable_on_demand_pool, _cfg_bool, _cfg_to_dict, _resolve_num_sampled_clients
from appfl_sim.misc.learning_utils import _build_federated_eval_plan, _evaluate_dataset_direct, _run_federated_eval_mpi, _should_eval_round
from appfl_sim.misc.logging_utils import _emit_federated_eval_policy_message, _log_round, _start_summary_lines
from appfl_sim.misc.runtime_utils import _build_clients, _build_on_demand_worker_pool, _rebind_client_for_on_demand_job
from appfl_sim.misc.system_utils import _client_processing_chunk_size, _iter_id_chunks, _release_clients
from appfl_sim.misc.data_utils import _resolve_client_eval_dataset, _sample_train_clients
from appfl_sim.misc.system_utils import get_local_rank


_MPI_AUTO_LAUNCH_ENV = "APPFL_SIM_MPI_AUTOLAUNCHED"

def _mpi_download_mode(config: DictConfig) -> str:
    raw_mode = str(config.get("mpi_dataset_download_mode", "rank0")).strip().lower()
    mode = raw_mode
    supported = {"rank0", "local_rank0", "all", "none"}
    if mode not in supported:
        raise ValueError(
            "mpi_dataset_download_mode must be one of: rank0, local_rank0, all, none"
        )
    return mode

def _load_dataset_mpi(
    config: DictConfig,
    communicator,
    rank: int,
    logger: Optional[ServerAgentFileLogger] = None,
):
    loader_cfg = _cfg_to_dict(config)
    if logger is not None:
        loader_cfg["logger"] = logger
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
    num_sampled_clients = _resolve_num_sampled_clients(
        config, num_clients=len(train_client_ids)
    )
    start_msg = _start_summary_lines(
        mode="mpi",
        config=config,
        num_clients=int(server.num_clients),
        train_client_count=len(train_client_ids),
        holdout_client_count=len(holdout_client_ids),
        num_sampled_clients=num_sampled_clients,
    )
    if logger is not None:
        logger.info(start_msg)
        _emit_federated_eval_policy_message(
            config=config,
            train_client_count=len(train_client_ids),
            holdout_client_count=len(holdout_client_ids),
            logger=logger,
        )
    else:
        print(start_msg)

    round_idx = 1
    while True:
        if round_idx > int(config.num_rounds):
            communicator.broadcast_global_model(args={"done": True})
            break

        selected_ids = _sample_train_clients(
            train_client_ids=train_client_ids,
            num_sampled_clients=int(num_sampled_clients),
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
            if plan["scheme"] == "holdout_client":
                federated_eval_in_metrics = _run_federated_eval_mpi(
                    config=config,
                    communicator=communicator,
                    model_state=server.model.state_dict(),
                    model=server.model,
                    device=str(config.device),
                    round_idx=round_idx,
                    eval_ids=list(plan["in_ids"]),
                    eval_tag="fed-in",
                    eval_split="test",
                )
                federated_eval_out_metrics = _run_federated_eval_mpi(
                    config=config,
                    communicator=communicator,
                    model_state=server.model.state_dict(),
                    model=server.model,
                    device=str(config.device),
                    round_idx=round_idx,
                    eval_ids=list(plan["out_ids"]),
                    eval_tag="fed-out",
                    eval_split="test",
                )
            else:
                federated_eval_metrics = _run_federated_eval_mpi(
                    config=config,
                    communicator=communicator,
                    model_state=server.model.state_dict(),
                    model=server.model,
                    device=str(config.device),
                    round_idx=round_idx,
                    eval_ids=list(plan["in_ids"]),
                    eval_tag="fed",
                    eval_split="test",
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
    finish_msg = f"Finished MPI simulation in {time.time() - t0:.2f}s.\nSaved resulting metrics in a log folder.\nGood Bye!"
    if logger is not None:
        logger.info(finish_msg)
    else:
        print(finish_msg)

def _run_client_mpi(
    communicator,
    config: DictConfig,
    model,
    client_datasets,
    local_client_ids,
    device: str,
    run_log_dir: str,
    client_logging_enabled: bool,
    trainer_name: str,
    use_on_demand: bool,
    on_demand_train_num_workers: Optional[int] = None,
    on_demand_eval_num_workers: Optional[int] = None,
):
    local_client_set = {int(cid) for cid in local_client_ids}
    on_demand_model = copy.deepcopy(model) if use_on_demand else None
    eval_model = on_demand_model if on_demand_model is not None else model
    eval_loss_fn = getattr(torch.nn, str(config.criterion))().to(device)
    eval_metric_names = parse_metric_names(config.get("eval_metrics", ["acc1"]))
    eval_batch_size = int(config.get("eval_batch_size", config.batch_size))
    eval_workers = (
        int(config.num_workers)
        if on_demand_eval_num_workers is None
        else max(0, int(on_demand_eval_num_workers))
    )
    chunk_size = _client_processing_chunk_size(
        config=config,
        model=on_demand_model if on_demand_model is not None else model,
        device=device,
        total_clients=len(local_client_set),
        phase="train",
    )
    eager_clients = None
    worker_pool: Optional[List[ClientAgent]] = None
    if not use_on_demand:
        eager_clients = _build_clients(
            config=config,
            model=model,
            client_datasets=client_datasets,
            local_client_ids=np.asarray(sorted(local_client_set)).astype(int),
            device=device,
            run_log_dir=run_log_dir,
            client_logging_enabled=client_logging_enabled,
            trainer_name=trainer_name,
        )
    else:
        if _allow_reusable_on_demand_pool(
            config,
            client_logging_enabled=client_logging_enabled,
        ):
            worker_pool = _build_on_demand_worker_pool(
                config=config,
                model=on_demand_model if on_demand_model is not None else model,
                client_datasets=client_datasets,
                local_client_ids=sorted(local_client_set),
                device=device,
                run_log_dir=run_log_dir,
                client_logging_enabled=client_logging_enabled,
                trainer_name=trainer_name,
                pool_size=chunk_size,
                num_workers_override=on_demand_train_num_workers,
            )

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
            eval_split = str(args.get("eval_split", "test"))
            eval_ids = sorted(
                int(cid) for cid in args.get("eval_ids", []) if int(cid) in local_client_set
            )
            if global_state is not None:
                eval_model.load_state_dict(global_state)
            eval_model = eval_model.to(device)
            eval_model.eval()
            for client_id in eval_ids:
                eval_ds = _resolve_client_eval_dataset(
                    client_datasets=client_datasets,
                    client_id=int(client_id),
                    eval_split=eval_split,
                )
                local_payload[int(client_id)] = {
                    "eval_stats": _evaluate_dataset_direct(
                        model=eval_model,
                        dataset=eval_ds,
                        device=device,
                        loss_fn=eval_loss_fn,
                        eval_metric_names=eval_metric_names,
                        batch_size=eval_batch_size,
                        num_workers=eval_workers,
                    )
                }
        else:
            selected = set(
                int(cid)
                for cid in args.get("selected_ids", [])
                if int(cid) in local_client_set
            )
            if eager_clients is not None:
                for client in eager_clients:
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
            else:
                for chunk_ids in _iter_id_chunks(sorted(selected), chunk_size):
                    if worker_pool:
                        chunk_clients = worker_pool[: len(chunk_ids)]
                        for client, cid in zip(chunk_clients, chunk_ids):
                            _rebind_client_for_on_demand_job(
                                client,
                                client_id=int(cid),
                                client_datasets=client_datasets,
                                num_workers_override=on_demand_train_num_workers,
                            )
                    else:
                        chunk_clients = _build_clients(
                            config=config,
                            model=on_demand_model if on_demand_model is not None else model,
                            client_datasets=client_datasets,
                            local_client_ids=np.asarray(chunk_ids).astype(int),
                            device=device,
                            run_log_dir=run_log_dir,
                            client_logging_enabled=client_logging_enabled,
                            trainer_name=trainer_name,
                            share_model=True,
                            num_workers_override=on_demand_train_num_workers,
                        )
                    for client in chunk_clients:
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
                    if not worker_pool:
                        _release_clients(chunk_clients)

        communicator.send_local_models_to_server(local_payload, dest=0)

    if eager_clients is not None:
        _release_clients(eager_clients)
    if worker_pool is not None:
        _release_clients(worker_pool)
    communicator.barrier()

def _is_running_under_mpi_launcher() -> bool:
    env_markers = (
        "OMPI_COMM_WORLD_SIZE",
        "OMPI_COMM_WORLD_RANK",
        "PMI_SIZE",
        "PMI_RANK",
        "PMIX_RANK",
        "MV2_COMM_WORLD_SIZE",
        "MP_CHILD",
        "HYDI_CONTROL_FD",
    )
    return any(key in os.environ for key in env_markers)

def _resolve_mpi_nproc(config: DictConfig) -> int:
    def _parse_int_prefix(text: str, default: int) -> int:
        digits = []
        for ch in str(text):
            if ch.isdigit():
                digits.append(ch)
            else:
                break
        if not digits:
            return default
        return int("".join(digits))

    def _detect_available_worker_slots() -> int:
        # Best-effort scheduler hints before falling back to local CPU count.
        for key in ("SLURM_CPUS_ON_NODE", "PBS_NP", "LSB_DJOB_NUMPROC"):
            raw = os.environ.get(key, "").strip()
            if raw:
                value = _parse_int_prefix(raw, 0)
                if value > 1:
                    return max(1, value - 1)
                if value == 1:
                    return 1
        return max(1, (os.cpu_count() or 2) - 1)

    workers = int(config.get("mpi_num_workers", 0))
    if workers <= 0:
        cpu_workers = _detect_available_worker_slots()
        configured_clients = int(config.get("num_clients", 0))
        if configured_clients > 0:
            workers = min(cpu_workers, configured_clients)
        else:
            workers = cpu_workers
    workers = max(1, int(workers))
    nproc = workers + 1  # server + worker ranks
    return nproc

def _find_mpi_launcher() -> Tuple[Optional[str], Dict[str, str]]:
    launcher = shutil.which("mpiexec") or shutil.which("mpirun")
    if launcher is not None:
        return launcher, {}

    candidate_bins: List[Path] = []
    home = Path.home()
    candidate_bins.append(home / "openmpi-install" / "bin")
    for env_key in ("MPI_HOME", "OPENMPI_HOME", "OMPI_HOME", "I_MPI_ROOT"):
        raw = os.environ.get(env_key, "").strip()
        if raw:
            candidate_bins.append(Path(raw).expanduser() / "bin")

    for bin_dir in candidate_bins:
        mpiexec_path = bin_dir / "mpiexec"
        mpirun_path = bin_dir / "mpirun"
        if mpiexec_path.exists():
            lib_dir = bin_dir.parent / "lib"
            updates = {}
            if lib_dir.exists():
                updates["LD_LIBRARY_PATH"] = (
                    f"{lib_dir}:{os.environ.get('LD_LIBRARY_PATH', '')}".rstrip(":")
                )
            return str(mpiexec_path), updates
        if mpirun_path.exists():
            lib_dir = bin_dir.parent / "lib"
            updates = {}
            if lib_dir.exists():
                updates["LD_LIBRARY_PATH"] = (
                    f"{lib_dir}:{os.environ.get('LD_LIBRARY_PATH', '')}".rstrip(":")
                )
            return str(mpirun_path), updates
    return None, {}

def _maybe_autolaunch_mpi(
    backend: str,
    config: DictConfig,
    cli_argv: list[str],
) -> None:
    if backend != "mpi":
        return
    if os.environ.get(_MPI_AUTO_LAUNCH_ENV, "0") == "1":
        return
    if _is_running_under_mpi_launcher():
        return

    launcher, env_updates = _find_mpi_launcher()
    if launcher is None:
        raise RuntimeError(
            "backend=mpi requested, but no MPI launcher found in PATH "
            "(expected 'mpiexec' or 'mpirun')."
        )

    nproc = _resolve_mpi_nproc(config)
    cmd = [launcher]
    if _cfg_bool(config, "mpi_oversubscribe", False):
        cmd.append("--oversubscribe")
    cmd.extend(["-n", str(nproc), sys.executable, "-m", "appfl_sim.runner", *cli_argv])
    env = os.environ.copy()
    env[_MPI_AUTO_LAUNCH_ENV] = "1"
    env.update(env_updates)
    completed = subprocess.run(cmd, env=env, check=False)
    raise SystemExit(int(completed.returncode))
