from __future__ import annotations
import gc
import os
import random
import re
from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from appfl_sim.logger import ServerAgentFileLogger



def set_seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _read_int_env(keys: List[str]) -> Optional[int]:
    for key in keys:
        value = os.environ.get(key, "")
        if value == "":
            continue
        try:
            return int(value)
        except Exception:
            continue
    return None


def get_local_rank(default: int = 0) -> int:
    """Best-effort local rank detection across common MPI launchers."""
    detected = _read_int_env(
        [
            "LOCAL_RANK",
            "OMPI_COMM_WORLD_LOCAL_RANK",
            "MV2_COMM_WORLD_LOCAL_RANK",
            "MPI_LOCALRANKID",
            "SLURM_LOCALID",
            "PMI_LOCAL_RANK",
        ]
    )
    if detected is None:
        return int(default)
    return max(0, int(detected))


def resolve_rank_device(
    base_device: str,
    rank: int,
    world_size: int,
    local_rank: Optional[int] = None,
) -> str:
    del world_size  # Reserved for future placement strategies.

    base = str(base_device).strip().lower()
    if not base.startswith("cuda") or not torch.cuda.is_available():
        return "cpu"

    if ":" in base:
        suffix = base.split(":", 1)[1].strip()
        if suffix and suffix.isdigit():
            return f"cuda:{int(suffix)}"
        if suffix not in {"", "local", "auto"}:
            return "cpu"

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        return "cpu"

    if local_rank is None:
        local_rank = get_local_rank(default=max(rank - 1, 0))
    gpu_idx = int(local_rank) % num_gpus
    return f"cuda:{gpu_idx}"


def parse_device_str(devices_str: str):
    """
    Parse `cpu`, `cuda:0`, or `cuda:0,cuda:1` device strings.
    """
    devices = [d.strip().lower() for d in devices_str.split(",")]

    if len(devices) == 1:
        dev = devices[0]
        if dev == "cpu":
            return ({"device_type": "cpu", "device_ids": []}, "cpu")
        if dev == "cuda":
            return ({"device_type": "gpu-single", "device_ids": []}, "cuda")
        if dev.startswith("cuda:"):
            match = re.match(r"cuda:(\d+)$", dev)
            if not match:
                raise ValueError(
                    f"Invalid device format: '{dev}'. Expected 'cuda:<index>' or 'cpu'"
                )
            index = int(match.group(1))
            if index < 0 or index >= torch.cuda.device_count():
                raise ValueError(
                    f"Requested {dev}, but only {torch.cuda.device_count()} GPUs available."
                )
            return ({"device_type": "gpu-single", "device_ids": [index]}, dev)
        raise ValueError(
            f"Unsupported device string: '{dev}'. Use 'cpu' or 'cuda:<index>'."
        )

    device_ids = []
    for d in devices:
        if d == "cpu":
            raise ValueError("Cannot mix 'cpu' with other devices in multi-device usage.")
        match = re.match(r"cuda:(\d+)$", d)
        if not match:
            raise ValueError(f"Invalid device format: '{d}'. Expected 'cuda:<index>'.")
        index = int(match.group(1))
        if index < 0 or index >= torch.cuda.device_count():
            raise ValueError(
                f"Requested {d}, but only {torch.cuda.device_count()} GPUs available."
            )
        device_ids.append(index)

    device_ids = sorted(set(device_ids))
    if not device_ids:
        raise ValueError("No valid CUDA devices parsed from string.")

    first_dev = f"cuda:{device_ids[0]}"
    return ({"device_type": "gpu-multi", "device_ids": device_ids}, first_dev)


def apply_model_device(model, config: dict, xy_device: str):
    """
    Extend `model.to()` by handling optional DataParallel wrapping.
    """
    device_type = config["device_type"]

    if device_type == "cpu":
        model.to("cpu")
        return model

    if device_type == "gpu-single":
        if len(config["device_ids"]) == 0:
            model.to(xy_device)
        else:
            model.to(torch.device(f"cuda:{config['device_ids'][0]}"))
        return model

    if device_type == "gpu-multi":
        model = nn.DataParallel(model, device_ids=config["device_ids"])
        model.to(torch.device(f"cuda:{config['device_ids'][0]}"))
        return model

    raise ValueError(f"Unknown device_type: {device_type}")


def clone_state_dict_optimized(state_dict, include_buffers: bool = True) -> Dict[str, torch.Tensor]:
    del include_buffers  # Included for compatibility with previous signature.
    result: Dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for name, tensor in state_dict.items():
            if tensor is not None:
                result[name] = tensor.clone().detach()
    gc.collect()
    return result


def extract_model_state_optimized(
    model: torch.nn.Module,
    include_buffers: bool = True,
    cpu_transfer: bool = False,
) -> Dict[str, torch.Tensor]:
    state: Dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for name, param in model.named_parameters():
            tensor = param.clone().detach()
            if cpu_transfer and tensor.device.type != "cpu":
                tensor = tensor.cpu()
            state[name] = tensor

        if include_buffers:
            for name, buffer in model.named_buffers():
                if buffer is None:
                    continue
                tensor = buffer.clone().detach()
                if cpu_transfer and tensor.device.type != "cpu":
                    tensor = tensor.cpu()
                state[name] = tensor

    gc.collect()
    return state


def safe_inplace_operation(
    tensor: torch.Tensor,
    operation: str,
    operand: Union[torch.Tensor, float, int],
    alpha: Optional[float] = None,
) -> torch.Tensor:
    if tensor.dtype in [torch.int32, torch.int64, torch.long]:
        if operation == "add":
            return tensor + (alpha * operand if alpha is not None else operand)
        if operation == "sub":
            return tensor - (alpha * operand if alpha is not None else operand)
        if operation == "mul":
            return tensor * operand
        if operation == "div":
            return torch.div(tensor, operand).type(tensor.dtype)
        raise ValueError(f"Unsupported operation: {operation}")

    if operation == "add":
        tensor.add_(operand, alpha=alpha) if alpha is not None else tensor.add_(operand)
    elif operation == "sub":
        tensor.sub_(operand, alpha=alpha) if alpha is not None else tensor.sub_(operand)
    elif operation == "mul":
        tensor.mul_(operand)
    elif operation == "div":
        tensor.div_(operand)
    else:
        raise ValueError(f"Unsupported operation: {operation}")
    return tensor


def optimize_memory_cleanup(
    *objects: Any,
    force_gc: bool = True,
    clear_cuda_cache: bool = False,
) -> None:
    for obj in objects:
        if obj is not None:
            del obj
    if force_gc:
        gc.collect()
    if clear_cuda_cache and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _maybe_force_server_cpu(
    config: DictConfig,
    enable_global_eval: bool,
    logger: Optional[ServerAgentFileLogger] = None,
) -> None:
    if enable_global_eval:
        return
    current = str(config.get("server_device", "cpu")).strip().lower()
    if not current.startswith("cuda"):
        return
    config.server_device = "cpu"
    msg = (
        "Global eval is disabled; forcing `server_device=cpu` to avoid unnecessary "
        "server-side GPU memory usage."
    )
    if logger is not None:
        logger.info(msg)
    else:
        print(msg)


def _resolve_cuda_index(device: str) -> int:
    text = str(device).strip().lower()
    if not text.startswith("cuda"):
        return 0
    if ":" not in text:
        return 0
    suffix = text.split(":", 1)[1].strip()
    if suffix.isdigit():
        return int(suffix)
    return 0


def _model_bytes(model) -> int:
    total = 0
    for p in model.parameters():
        total += int(p.numel()) * int(p.element_size())
    for b in model.buffers():
        total += int(b.numel()) * int(b.element_size())
    return int(total)


def _client_processing_chunk_size(
    config: DictConfig,
    model=None,
    device: str = "cpu",
    total_clients: int = 0,
    phase: str = "train",
) -> int:
    phase_name = str(phase).strip().lower()
    configured = int(config.get("client_processing_chunk_size", 0))
    if configured > 0:
        return max(1, configured)

    if str(device).strip().lower().startswith("cuda") and torch.cuda.is_available():
        try:
            dev_idx = _resolve_cuda_index(device)
            free_bytes, _ = torch.cuda.mem_get_info(dev_idx)
        except Exception:
            free_bytes = 0
        model_bytes = _model_bytes(model) if model is not None else 64 * 1024 * 1024
        per_client = max(256 * 1024 * 1024, model_bytes * (10 if phase_name == "train" else 4))
        budget = int(float(free_bytes) * 0.35) if free_bytes > 0 else 0
        auto_chunk = int(budget // per_client) if budget > 0 else 1
        auto_chunk = max(1, min(64, auto_chunk))
    else:
        cpu = max(1, (os.cpu_count() or 1))
        auto_chunk = max(1, min(64, cpu // 2))

    if int(total_clients) > 0:
        auto_chunk = min(auto_chunk, int(total_clients))
    return max(1, int(auto_chunk))


def _iter_id_chunks(ids: Sequence[int], chunk_size: int):
    ordered = list(ids)
    for start in range(0, len(ordered), chunk_size):
        yield ordered[start : start + chunk_size]


def _release_clients(clients, clear_cuda_cache: bool = False) -> None:
    if clients is None:
        return
    if isinstance(clients, list):
        clients.clear()
    del clients
    gc.collect()
    if clear_cuda_cache and torch.cuda.is_available():
        torch.cuda.empty_cache()
