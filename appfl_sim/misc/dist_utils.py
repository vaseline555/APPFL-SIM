from __future__ import annotations

import os
import socket
import time
from typing import Callable

import torch
from omegaconf import DictConfig, OmegaConf
from appfl_sim.misc.config_utils import _cfg_get


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _resolve_distributed_world_size(config: DictConfig, backend: str) -> int:
    if backend == "nccl":
        gpus = int(torch.cuda.device_count())
        if gpus <= 0:
            raise RuntimeError("backend=nccl requested, but no CUDA devices are available.")
        return gpus
    cpu_slots = max(1, (os.cpu_count() or 2) - 1)
    configured_clients = int(_cfg_get(config, "train.num_clients", 0))
    if configured_clients > 0:
        return max(1, min(cpu_slots, configured_clients))
    return cpu_slots


def _distributed_worker_entry(
    rank: int,
    world_size: int,
    backend: str,
    config_dict: dict,
    master_port: int,
    entry_fn: Callable[[DictConfig, str], None],
) -> None:
    import torch.distributed as dist

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    dist.init_process_group(
        backend=backend,
        init_method="env://",
        rank=rank,
        world_size=world_size,
    )
    try:
        entry_fn(OmegaConf.create(config_dict), backend)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def launch_or_run_distributed(
    backend: str,
    config: DictConfig,
    entry_fn: Callable[[DictConfig, str], None],
) -> None:
    import torch.distributed as dist
    import torch.multiprocessing as mp

    if dist.is_available() and dist.is_initialized():
        entry_fn(config, backend)
        return

    env_world_size = os.environ.get("WORLD_SIZE", "").strip()
    env_rank = os.environ.get("RANK", "").strip()
    if env_world_size and env_rank:
        dist.init_process_group(backend=backend, init_method="env://")
        try:
            entry_fn(config, backend)
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()
        return

    world_size = _resolve_distributed_world_size(config, backend)
    if world_size <= 0:
        raise RuntimeError("Unable to resolve distributed world size.")

    master_port = _find_free_port()
    ctx = mp.get_context("spawn")
    processes = []
    for rank in range(world_size):
        proc = ctx.Process(
            target=_distributed_worker_entry,
            args=(rank, world_size, backend, dict(OmegaConf.to_container(config, resolve=True)), master_port, entry_fn),
        )
        proc.start()
        processes.append(proc)

    def _terminate_processes(procs):
        for p in procs:
            if p.is_alive():
                p.terminate()
        for p in procs:
            p.join(timeout=2.0)
        for p in procs:
            if p.is_alive():
                p.kill()

    try:
        while processes:
            alive = []
            failed_exit = None
            for proc in processes:
                proc.join(timeout=0.2)
                if proc.exitcode is None:
                    alive.append(proc)
                elif proc.exitcode != 0:
                    failed_exit = int(proc.exitcode)
            if failed_exit is not None:
                _terminate_processes(alive)
                raise RuntimeError(f"Distributed worker exited with code {failed_exit}.")
            processes = alive
            if processes:
                time.sleep(0.05)
    except KeyboardInterrupt:
        _terminate_processes(processes)
        raise SystemExit(130)
    except Exception:
        _terminate_processes(processes)
        raise
