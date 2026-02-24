from __future__ import annotations
import ast
import importlib
import importlib.util
import inspect
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional

import torch
from omegaconf import DictConfig, OmegaConf


def _default_config_path() -> Path:
    package_root = Path(__file__).resolve().parent.parent
    return package_root / "config" / "examples" / "simulation.yaml"

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

def _cfg_to_dict(cfg) -> Dict:
    if isinstance(cfg, DictConfig):
        return dict(OmegaConf.to_container(cfg, resolve=True))
    if isinstance(cfg, SimpleNamespace):
        return dict(vars(cfg))
    if isinstance(cfg, dict):
        return dict(cfg)
    if hasattr(cfg, "__dict__"):
        return dict(vars(cfg))
    return {}

def _cfg_get(config: DictConfig | dict, path: str, default: Any = None) -> Any:
    parts = [p for p in str(path).split(".") if p]
    current: Any = config
    for part in parts:
        if isinstance(current, DictConfig):
            if part not in current:
                return default
            current = current.get(part)
            continue
        if isinstance(current, dict):
            if part not in current:
                return default
            current = current.get(part)
            continue
        return default
    return default if current is None else current


def _cfg_set(config: DictConfig | dict, path: str, value: Any) -> None:
    parts = [p for p in str(path).split(".") if p]
    if not parts:
        return
    current: Any = config
    for part in parts[:-1]:
        if isinstance(current, DictConfig):
            if part not in current or current.get(part) is None:
                current[part] = OmegaConf.create({})
            current = current.get(part)
            continue
        if isinstance(current, dict):
            if part not in current or current.get(part) is None:
                current[part] = {}
            current = current.get(part)
            continue
        return
    last = parts[-1]
    if isinstance(current, DictConfig):
        current[last] = value
    elif isinstance(current, dict):
        current[last] = value


def _resolve_component_backend(kind: str, backend: str, path: str) -> str:
    mode = str(backend or "auto").strip().lower()
    if mode == "auto":
        return "custom" if str(path or "").strip() else "torch"
    if mode not in {"torch", "custom"}:
        raise ValueError(f"{kind}.backend must be one of: auto, torch, custom")
    return mode


def _load_named_symbol(path: str, name: str):
    file_path = Path(str(path)).expanduser().resolve()
    if not file_path.is_file():
        raise FileNotFoundError(f"Custom component file not found: {file_path}")
    module_name = f"_appfl_sim_custom_{file_path.stem}_{abs(hash(str(file_path)))}"
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import custom component from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if str(name).strip() == "":
        exported: list[Any] = []
        for obj in module.__dict__.values():
            if inspect.isclass(obj) and getattr(obj, "__module__", "") == module.__name__:
                exported.append(obj)
        for obj in module.__dict__.values():
            if inspect.isfunction(obj) and getattr(obj, "__module__", "") == module.__name__:
                exported.append(obj)
        if not exported:
            raise ValueError(f"No class/function found in custom component file: {file_path}")
        return exported[-1]
    if not hasattr(module, name):
        raise ValueError(f"Custom component `{name}` not found in {file_path}")
    return getattr(module, name)

def _get_last_class_name_from_file(file_path: str) -> Optional[str]:
    with open(file_path) as file:
        file_content = file.read()
    tree = ast.parse(file_content)
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    if classes:
        return classes[-1].name
    return None


def _get_last_function_name_from_file(file_path: str) -> Optional[str]:
    with open(file_path) as file:
        file_content = file.read()
    tree = ast.parse(file_content)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if functions:
        return functions[-1].name
    return None


def _load_module_from_file(file_path: str):
    file_path = os.path.abspath(file_path)
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    module_dir, module_file = os.path.split(file_path)
    module_name, _ = os.path.splitext(module_file)
    if module_dir not in sys.path:
        sys.path.append(module_dir)
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load python module from: {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _create_instance_from_file(file_path, class_name=None, *args, **kwargs):
    if class_name is None:
        class_name = _get_last_class_name_from_file(file_path)
    if class_name is None:
        raise ValueError(f"No class found in file: {file_path}")
    module = _load_module_from_file(file_path)
    cls = getattr(module, class_name)
    return cls(*args, **kwargs)


def _run_function_from_file(file_path, function_name=None, *args, **kwargs):
    if function_name is None:
        function_name = _get_last_function_name_from_file(file_path)
    if function_name is None:
        raise ValueError(f"No function found in file: {file_path}")
    module = _load_module_from_file(file_path)
    function = getattr(module, function_name)
    return function(*args, **kwargs)


def build_loss_from_config(config: DictConfig | dict):
    name = str(_cfg_get(config, "loss.name", "CrossEntropyLoss"))
    backend = _resolve_component_backend(
        kind="loss",
        backend=str(_cfg_get(config, "loss.backend", "auto")),
        path=str(_cfg_get(config, "loss.path", "")),
    )
    kwargs = _cfg_to_dict(_cfg_get(config, "loss.configs", {}))
    if backend == "torch":
        if not hasattr(torch.nn, name):
            raise ValueError(f"Loss {name} not found in torch.nn")
        return getattr(torch.nn, name)(**kwargs)
    target = _load_named_symbol(str(_cfg_get(config, "loss.path", "")), name)
    if inspect.isclass(target):
        return target(**kwargs)
    if callable(target):
        return target(**kwargs)
    raise TypeError("Custom loss target must be a class or callable")


def build_loss_from_train_cfg(train_cfg: DictConfig | dict):
    payload = {
        "loss": {
            "name": _cfg_get(train_cfg, "loss.name", "CrossEntropyLoss"),
            "backend": _cfg_get(train_cfg, "loss.backend", "auto"),
            "path": _cfg_get(train_cfg, "loss.path", ""),
            "configs": _cfg_get(train_cfg, "loss.configs", {}),
        }
    }
    return build_loss_from_config(payload)


def build_optimizer_from_train_cfg(train_cfg: DictConfig | dict, params: Iterable):
    name = str(_cfg_get(train_cfg, "optimizer.name", "SGD"))
    path = str(_cfg_get(train_cfg, "optimizer.path", ""))
    backend = _resolve_component_backend(
        kind="optimizer",
        backend=str(_cfg_get(train_cfg, "optimizer.backend", "auto")),
        path=path,
    )
    kwargs = _cfg_to_dict(_cfg_get(train_cfg, "optimizer.configs", {}))
    kwargs.setdefault("lr", float(_cfg_get(train_cfg, "optimizer.lr", 0.01)))
    if "weight_decay" not in kwargs:
        kwargs["weight_decay"] = float(
            _cfg_get(train_cfg, "optimizer.configs.weight_decay", 0.0)
        )

    if backend == "torch":
        if not hasattr(torch.optim, name):
            raise ValueError(f"Optimizer {name} not found in torch.optim")
        return getattr(torch.optim, name)(params, **kwargs)

    target = _load_named_symbol(path, name)
    if inspect.isclass(target):
        return target(params, **kwargs)
    if callable(target):
        return target(params, **kwargs)
    raise TypeError("Custom optimizer target must be a class or callable")


def _ensure_model_cfg(cfg: DictConfig) -> None:
    if _cfg_get(cfg, "model.configs", None) is None:
        _cfg_set(cfg, "model.configs", OmegaConf.create({}))
    if _cfg_get(cfg, "dataset.configs", None) is None:
        _cfg_set(cfg, "dataset.configs", OmegaConf.create({}))
    if _cfg_get(cfg, "split.configs", None) is None:
        _cfg_set(cfg, "split.configs", OmegaConf.create({}))
    if _cfg_get(cfg, "eval.configs", None) is None:
        _cfg_set(cfg, "eval.configs", OmegaConf.create({}))
    if _cfg_get(cfg, "logging.configs", None) is None:
        _cfg_set(cfg, "logging.configs", OmegaConf.create({}))
    if _cfg_get(cfg, "loss.configs", None) is None:
        _cfg_set(cfg, "loss.configs", OmegaConf.create({}))
    if _cfg_get(cfg, "optimizer.configs", None) is None:
        _cfg_set(cfg, "optimizer.configs", OmegaConf.create({}))
    if _cfg_get(cfg, "privacy.kwargs", None) is None:
        _cfg_set(cfg, "privacy.kwargs", OmegaConf.create({}))

def _cfg_bool(config: DictConfig, key: str, default: bool) -> bool:
    value = _cfg_get(config, key, default)
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
        num_workers = int(_cfg_get(config, "train.num_workers", 0))
    else:
        num_workers = max(0, int(num_workers_override))
    device_text = str(device).strip().lower()
    default_pin_memory = device_text.startswith("cuda")
    pin_memory = _cfg_bool(config, "train.pin_memory", default_pin_memory)
    update_base_raw = str(_cfg_get(config, "train.update_base", "epoch")).strip().lower()
    if update_base_raw == "iter":
        mode = "step"
        local_iters = int(_cfg_get(config, "train.local_iters", 1))
        local_iters = max(1, local_iters)
    else:
        mode = "epoch"
        local_epochs = int(_cfg_get(config, "train.local_epochs", 1))
        local_epochs = max(1, local_epochs)
    optimizer_configs = _cfg_to_dict(_cfg_get(config, "optimizer.configs", {}))
    loss_configs = _cfg_to_dict(_cfg_get(config, "loss.configs", {}))
    train_cfg = {
        "device": device,
        "mode": mode,
        "batch_size": int(_cfg_get(config, "train.batch_size", 32)),
        "train_data_shuffle": _cfg_bool(config, "train.shuffle", True),
        "eval_batch_size": int(
            _cfg_get(config, "train.eval_batch_size", _cfg_get(config, "train.batch_size", 32))
        ),
        "num_workers": int(num_workers),
        "train_pin_memory": _cfg_bool(config, "train.train_pin_memory", pin_memory),
        "eval_pin_memory": _cfg_bool(config, "train.eval_pin_memory", pin_memory),
        "dataloader_persistent_workers": _cfg_bool(
            config, "train.dataloader_persistent_workers", False
        ),
        "dataloader_prefetch_factor": int(_cfg_get(config, "train.dataloader_prefetch_factor", 2)),
        "optimizer": {
            "name": str(_cfg_get(config, "optimizer.name", "SGD")),
            "backend": str(_cfg_get(config, "optimizer.backend", "auto")),
            "path": str(_cfg_get(config, "optimizer.path", "")),
            "configs": optimizer_configs,
            "lr": float(_cfg_get(config, "optimizer.lr", 0.01)),
            "lr_decay": {
                "enable": _cfg_bool(config, "optimizer.lr_decay.enable", False),
                "type": str(_cfg_get(config, "optimizer.lr_decay.type", "none")),
                "gamma": float(_cfg_get(config, "optimizer.lr_decay.gamma", 0.99)),
                "t_max": int(_cfg_get(config, "optimizer.lr_decay.t_max", 0)),
                "eta_min": float(_cfg_get(config, "optimizer.lr_decay.eta_min", 0.0)),
                "min_lr": float(_cfg_get(config, "optimizer.lr_decay.min_lr", 0.0)),
            },
        },
        "loss": {
            "name": str(_cfg_get(config, "loss.name", "CrossEntropyLoss")),
            "backend": str(_cfg_get(config, "loss.backend", "auto")),
            "path": str(_cfg_get(config, "loss.path", "")),
            "configs": loss_configs,
        },
        "num_rounds": int(_cfg_get(config, "train.num_rounds", 20)),
        "max_grad_norm": float(_cfg_get(config, "optimizer.clip_grad_norm", 0.0)),
        "logging_output_dirname": str(run_log_dir),
        "logging_output_filename": "client",
        "experiment_id": str(_cfg_get(config, "experiment.name", "appfl-sim")),
        "client_logging_enabled": True,
        "do_pre_evaluation": _cfg_bool(config, "eval.do_pre_evaluation", True),
        "do_post_evaluation": _cfg_bool(config, "eval.do_post_evaluation", True),
        "eval_metrics": _cfg_get(config, "eval.metrics", ["acc1"]),
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
    scheme = str(_cfg_get(config, "logging.type", "auto")).strip().lower()

    if scheme not in {"auto", "both", "server_only"}:
        raise ValueError(
            "logging.type must be one of: auto, both, server_only"
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
        "basis_clients": basis_clients,
        "total_clients": int(num_clients),
        "forced_server_only": bool(forced_server_only),
    }

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
    algorithm = str(_cfg_get(config, "algorithm.name", "fedavg")).strip().lower()
    explicit_aggregator = str(_cfg_get(config, "algorithm.aggregator", "")).strip()
    explicit_scheduler = str(_cfg_get(config, "algorithm.scheduler", "")).strip()
    explicit_trainer = str(_cfg_get(config, "algorithm.trainer", "")).strip()
    aggregator_name = explicit_aggregator or f"{_to_pascal_case(algorithm)}Aggregator"
    scheduler_name = explicit_scheduler or f"{_to_pascal_case(algorithm)}Scheduler"
    trainer_name = explicit_trainer or f"{_to_pascal_case(algorithm)}Trainer"

    if not _module_has_class("appfl_sim.algorithm.aggregator", aggregator_name):
        raise ValueError(
            f"Aggregator class '{aggregator_name}' not found for algorithm='{algorithm}'. "
            "Implement it under appfl_sim/algorithm/aggregator and expose/import it."
        )
    if not _module_has_class("appfl_sim.algorithm.scheduler", scheduler_name):
        raise ValueError(
            f"Scheduler class '{scheduler_name}' not found for algorithm='{algorithm}'. "
            "Implement it under appfl_sim/algorithm/scheduler and expose/import it."
        )
    if not _module_has_class("appfl_sim.algorithm.trainer", trainer_name):
        raise ValueError(
            f"Trainer class '{trainer_name}' not found for algorithm='{algorithm}'. "
            "Implement it under appfl_sim/algorithm/trainer and expose/import it."
        )

    agg_kwargs_raw = _cfg_get(config, "algorithm.aggregator_kwargs", {})
    sched_kwargs_raw = _cfg_get(config, "algorithm.scheduler_kwargs", {})
    trainer_kwargs_raw = _cfg_get(config, "algorithm.trainer_kwargs", {})
    aggregator_kwargs = (
        _cfg_to_dict(agg_kwargs_raw) if agg_kwargs_raw is not None else {}
    )
    scheduler_kwargs = _cfg_to_dict(sched_kwargs_raw) if sched_kwargs_raw is not None else {}
    trainer_kwargs = _cfg_to_dict(trainer_kwargs_raw) if trainer_kwargs_raw is not None else {}

    if aggregator_name == "FedavgAggregator":
        aggregator_kwargs.setdefault("client_weights_mode", "sample_ratio")

    return {
        "algorithm_name": algorithm,
        "aggregator_name": aggregator_name,
        "aggregator_kwargs": aggregator_kwargs,
        "scheduler_name": scheduler_name,
        "scheduler_kwargs": scheduler_kwargs,
        "trainer_name": trainer_name,
        "trainer_kwargs": trainer_kwargs,
    }

def _allow_reusable_on_demand_pool(
    config: DictConfig,
    *,
    client_logging_enabled: bool,
) -> bool:
    if bool(client_logging_enabled):
        return False
    if _cfg_bool(config, "secure_aggregation.use_sec_agg", False):
        return False
    if _cfg_bool(config, "privacy.use_dp", False):
        mechanism = str(_cfg_get(config, "privacy.mechanism", "laplace")).strip().lower()
        if mechanism == "opacus":
            return False
    return True

def _merge_runtime_cfg(config: DictConfig, loader_args: Any) -> Dict:
    runtime_cfg = _cfg_to_dict(config)
    if isinstance(loader_args, dict):
        runtime_cfg.update(loader_args)
    elif hasattr(loader_args, "__dict__"):
        runtime_cfg.update(vars(loader_args))
    runtime_cfg["num_clients"] = int(
        _cfg_get(config, "train.num_clients", runtime_cfg.get("num_clients", 0))
    )
    return runtime_cfg

def _parse_cli_tokens(argv: list[str]) -> tuple[str | None, str | None, list[str]]:
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
    backend = None
    out: list[str] = []
    idx = 0
    while idx < len(remaining):
        tok = remaining[idx]
        if tok in {"-h", "--help"}:
            print(
                """
                APPFL-SIM runner

                Usage:
                python -m appfl_sim.runner --config /path/to/config.yaml
                appfl-sim experiment.backend=serial dataset.name=MNIST train.num_clients=3 train.num_rounds=2

                Distributed notes:
                - backend=nccl uses one process per visible GPU.
                - backend=gloo uses CPU processes (auto-sized by CPU capacity and num_clients).
                - backend=serial is the default for lightweight experiments.
                """.strip()
            )
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
            if idx + 1 < len(remaining) and not remaining[idx + 1].startswith("--"):
                out.append(f"{key}={remaining[idx + 1]}")
                idx += 2
            else:
                out.append(f"{key}=true")
                idx += 1
            continue
        if "=" in tok:
            key, value = tok.split("=", 1)
            key = key.replace("-", "_")
            if key == "experiment.backend":
                backend = value
            else:
                out.append(f"{key}={value}")
        idx += 1
    return config_path, backend, out
