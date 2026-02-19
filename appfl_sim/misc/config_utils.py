from __future__ import annotations
import importlib
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from omegaconf import DictConfig, OmegaConf
from appfl_sim.logger import ServerAgentFileLogger


def _default_config_path() -> Path:
    package_root = Path(__file__).resolve().parent.parent
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
    package_root = Path(__file__).resolve().parent.parent

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
        APPFL-SIM runner

        Usage:
        python -m appfl_sim.runner --config /path/to/config.yaml
        appfl-sim backend=serial dataset=MNIST num_clients=3 num_rounds=2

        Distributed notes:
        - backend=nccl uses one process per visible GPU.
        - backend=gloo uses CPU processes (auto-sized by CPU capacity and num_clients).
        - backend=serial is the default for lightweight experiments.
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

def _ensure_model_cfg(cfg: DictConfig) -> None:
    del cfg

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


def _build_train_cfg(
    config: DictConfig,
    device: str,
    run_log_dir: str,
    num_workers_override: Optional[int] = None,
) -> Dict:
    if num_workers_override is None:
        num_workers = int(config.num_workers)
    else:
        num_workers = max(0, int(num_workers_override))
    device_text = str(device).strip().lower()
    default_pin_memory = device_text.startswith("cuda")
    pin_memory = _cfg_bool(config, "pin_memory", default_pin_memory)
    update_base_raw = str(config.get("update_base", "epoch")).strip().lower()
    if update_base_raw in {"iter", "step"}:
        mode = "step"
        local_iters = int(config.get("local_iters", 1))
        local_iters = max(1, local_iters)
    else:
        mode = "epoch"
        local_epochs = int(config.get("local_epochs", 1))
        local_epochs = max(1, local_epochs)
    train_cfg = {
        "device": device,
        "mode": mode,
        "batch_size": int(config.batch_size),
        "eval_batch_size": int(config.get("eval_batch_size", config.batch_size)),
        "num_workers": int(num_workers),
        "train_pin_memory": _cfg_bool(config, "train_pin_memory", pin_memory),
        "eval_pin_memory": _cfg_bool(config, "eval_pin_memory", pin_memory),
        "dataloader_persistent_workers": _cfg_bool(
            config, "dataloader_persistent_workers", False
        ),
        "dataloader_prefetch_factor": int(config.get("dataloader_prefetch_factor", 2)),
        "optim": str(config.optimizer),
        "lr": float(config.lr),
        "weight_decay": float(config.weight_decay),
        "max_grad_norm": float(config.max_grad_norm),
        "logging_output_dirname": str(run_log_dir),
        "logging_output_filename": "client",
        "experiment_id": str(config.exp_name),
        "client_logging_enabled": True,
        "do_pre_validation": _cfg_bool(config, "do_pre_validation", True),
        "do_validation": _cfg_bool(config, "do_validation", True),
        "eval_metrics": config.get("eval_metrics", ["acc1"]),
    }
    if mode == "epoch":
        train_cfg["num_local_epochs"] = local_epochs
    else:
        train_cfg["num_local_steps"] = local_iters
    return train_cfg

def _resolve_client_logging_policy(
    config: DictConfig,
    num_clients: int,
    num_sampled_clients: int,
) -> Dict[str, object]:
    scheme = str(config.get("logging_scheme", "auto")).strip().lower()
    warning_threshold = int(config.get("per_client_logging_warning_threshold", 50))

    if scheme not in {"auto", "both", "server_only"}:
        raise ValueError(
            "logging_scheme must be one of: auto, both, server_only"
        )

    basis_clients = max(1, int(num_sampled_clients))

    if scheme == "server_only":
        effective = "server_only"
    else:
        effective = "both"

    forced_server_only = int(num_sampled_clients) < int(num_clients)
    if forced_server_only:
        effective = "server_only"

    return {
        "requested_scheme": scheme,
        "effective_scheme": effective,
        "client_logging_enabled": effective == "both",
        "warning_threshold": warning_threshold,
        "basis_clients": basis_clients,
        "total_clients": int(num_clients),
        "forced_server_only": bool(forced_server_only),
    }

def _resolve_client_state_policy(config: DictConfig) -> Dict[str, object]:
    """Client lifecycle policy.

    Default is stateless (on-demand): instantiate sampled client(s), run, then free.
    Stateful mode is explicit via `stateful_clients=true`.

    """
    stateful = _cfg_bool(config, "stateful_clients", False)
    source = "stateful_clients"

    return {
        "stateful_clients": bool(stateful),
        "use_on_demand": not bool(stateful),
        "source": source,
    }

def _resolve_on_demand_worker_policy(
    config: DictConfig,
    logger: Optional[ServerAgentFileLogger] = None,
) -> Dict[str, int]:
    base_workers = max(0, int(config.get("num_workers", 0)))
    train_workers = max(
        0,
        int(config.get("on_demand_num_workers", 0)),
    )
    eval_workers = max(0, int(config.get("on_demand_eval_num_workers", 0)))
    if logger is not None and base_workers > 0:
        if "on_demand_num_workers" not in config:
            logger.info(
                f"On-demand training dataloaders default to 0 workers (num_workers={base_workers} ignored). "
                "Set on_demand_num_workers to override."
            )
        if "on_demand_eval_num_workers" not in config:
            logger.info(
                f"On-demand eval dataloaders default to 0 workers (num_workers={base_workers} ignored). "
                "Set on_demand_eval_num_workers to override."
            )
    return {"train": train_workers, "eval": eval_workers}

def _to_pascal_case(name: str) -> str:
    text = str(name or "").strip()
    if text == "":
        return ""
    chunks = re.split(r"[^0-9a-zA-Z]+", text)
    chunks = [c for c in chunks if c]
    if not chunks:
        return text
    return "".join(c[:1].upper() + c[1:] for c in chunks)

def _module_has_class(module_path: str, class_name: str) -> bool:
    if str(class_name).strip() == "":
        return False
    try:
        mod = importlib.import_module(module_path)
    except Exception:
        return False
    if hasattr(mod, class_name):
        return True

    class_text = str(class_name)
    snake = re.sub(r"(?<!^)(?=[A-Z])", "_", class_text).lower()
    compact = snake.replace("_", "")
    candidates = [snake, compact]

    suffix_map = {
        "Aggregator": "_aggregator",
        "Scheduler": "_scheduler",
        "Trainer": "_trainer",
    }
    for suffix, file_suffix in suffix_map.items():
        if class_text.endswith(suffix):
            base = class_text[: -len(suffix)]
            base_snake = re.sub(r"(?<!^)(?=[A-Z])", "_", base).lower()
            base_compact = base_snake.replace("_", "")
            candidates.append(f"{base_snake}{file_suffix}")
            candidates.append(f"{base_compact}{file_suffix}")
            break

    unique_candidates = []
    seen = set()
    for item in candidates:
        key = str(item).strip(".")
        if not key or key in seen:
            continue
        seen.add(key)
        unique_candidates.append(key)

    module_candidates = [f"{module_path}.{name}" for name in unique_candidates]
    for submodule_path in module_candidates:
        try:
            submodule = importlib.import_module(submodule_path)
        except Exception:
            continue
        if hasattr(submodule, class_name):
            setattr(mod, class_name, getattr(submodule, class_name))
            return True
    return False

def _resolve_algorithm_components(config: DictConfig) -> Dict[str, Any]:
    algorithm = str(config.get("algorithm", "fedavg")).strip().lower()
    explicit_aggregator = str(config.get("aggregator", "")).strip()
    explicit_scheduler = str(config.get("scheduler", "")).strip()
    explicit_trainer = str(config.get("trainer", "")).strip()

    if explicit_aggregator:
        aggregator_name = explicit_aggregator
    elif algorithm == "fedavg":
        aggregator_name = "FedAvgAggregator"
    else:
        aggregator_name = f"{_to_pascal_case(algorithm)}Aggregator"

    if explicit_scheduler:
        scheduler_name = explicit_scheduler
    elif algorithm == "fedavg":
        scheduler_name = "SyncScheduler"
    else:
        scheduler_name = f"{_to_pascal_case(algorithm)}Scheduler"

    if explicit_trainer:
        trainer_name = explicit_trainer
    elif algorithm == "fedavg":
        trainer_name = "VanillaTrainer"
    else:
        trainer_name = f"{_to_pascal_case(algorithm)}Trainer"

    if not _module_has_class("appfl_sim.algorithm.aggregator", aggregator_name):
        if algorithm == "fedavg":
            aggregator_name = "FedAvgAggregator"
        else:
            raise ValueError(
                f"Aggregator class '{aggregator_name}' not found for algorithm='{algorithm}'. "
                "Implement it under appfl_sim/algorithm/aggregator and expose/import it."
            )
    if not _module_has_class("appfl_sim.algorithm.scheduler", scheduler_name):
        scheduler_name = "SyncScheduler"
    if not _module_has_class("appfl_sim.algorithm.trainer", trainer_name):
        trainer_name = "VanillaTrainer"

    agg_kwargs_raw = config.get("aggregator_kwargs", {})
    sched_kwargs_raw = config.get("scheduler_kwargs", {})
    aggregator_kwargs = (
        _cfg_to_dict(agg_kwargs_raw) if agg_kwargs_raw is not None else {}
    )
    scheduler_kwargs = _cfg_to_dict(sched_kwargs_raw) if sched_kwargs_raw is not None else {}

    if aggregator_name == "FedAvgAggregator":
        aggregator_kwargs.setdefault("client_weights_mode", "sample_ratio")

    return {
        "algorithm": algorithm,
        "aggregator_name": aggregator_name,
        "aggregator_kwargs": aggregator_kwargs,
        "scheduler_name": scheduler_name,
        "scheduler_kwargs": scheduler_kwargs,
        "trainer_name": trainer_name,
    }

def _allow_reusable_on_demand_pool(
    config: DictConfig,
    *,
    client_logging_enabled: bool,
) -> bool:
    if bool(client_logging_enabled):
        return False
    if _cfg_bool(config, "use_secure_agg", False):
        return False
    if _cfg_bool(config, "use_dp", False):
        mechanism = str(config.get("dp_mechanism", "laplace")).strip().lower()
        if mechanism == "opacus":
            return False
    return True

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

def _resolve_num_sampled_clients(config: DictConfig, num_clients: int) -> int:
    if int(num_clients) <= 0:
        return 0

    if "num_sampled_clients" in config:
        try:
            n = int(config.get("num_sampled_clients", 0))
        except Exception:
            n = 0
        if n > 0:
            return max(1, min(int(num_clients), n))

    return int(num_clients)

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
        if tok in {"serial", "nccl", "gloo"}:
            backend = tok
            idx += 1
            continue
        if tok.startswith("--"):
            keyval = tok[2:]
            if "=" in keyval:
                key, value = keyval.split("=", 1)
                key = key.replace("-", "_")
                out.append(f"{key.replace('-', '_')}={value}")
                idx += 1
                continue
            key = keyval.replace("-", "_")
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
            else:
                out.append(f"{key}={value}")
        idx += 1
    return backend, out
