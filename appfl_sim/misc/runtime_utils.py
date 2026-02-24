from __future__ import annotations
import ast
import copy
import importlib
import importlib.util
import os
import os.path as osp
import random
import string
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from appfl_sim.misc.config_utils import (
    _build_train_cfg,
    _cfg_bool,
    _cfg_get,
    _resolve_algorithm_components,
    _resolve_client_logging_policy,
    _resolve_num_sampled_clients,
)
from appfl_sim.misc.config_utils import build_loss_from_config
from appfl_sim.misc.data_utils import _build_client_groups

def _maybe_select_round_local_steps(server, round_idx: int):
    scheduler = getattr(server, "scheduler", None)
    if scheduler is None or not hasattr(scheduler, "pull"):
        return None
    try:
        return int(scheduler.pull(round_idx=int(round_idx)))
    except Exception:
        return None

def _normalize_uploaded_state(uploaded):
    if isinstance(uploaded, tuple):
        return uploaded[0]
    return uploaded

def _run_local_client_update(
    client,
    *,
    global_state,
    round_idx: int,
    round_local_steps: Optional[int],
):
    client.download(global_state)
    if round_local_steps is None:
        train_result = client.update(round_idx=round_idx)
    else:
        train_result = client.update(
            round_idx=round_idx, local_steps=int(round_local_steps)
        )
    state = _normalize_uploaded_state(client.upload())
    return train_result, state


def _resolve_runtime_policies(config: DictConfig, runtime_cfg: Dict[str, Any]) -> Dict[str, Any]:
    num_clients = int(runtime_cfg["num_clients"])
    algorithm_components = _resolve_algorithm_components(config)
    state_policy = {
        "stateful": bool(_cfg_bool(config, "experiment.stateful", False)),
        "source": "experiment.stateful",
    }
    train_client_ids, holdout_client_ids = _build_client_groups(config, num_clients)
    num_sampled_clients = _resolve_num_sampled_clients(
        config, num_clients=len(train_client_ids)
    )
    logging_policy = _resolve_client_logging_policy(
        config,
        num_clients=num_clients,
        num_sampled_clients=num_sampled_clients,
    )
    return {
        "algorithm_components": algorithm_components,
        "num_clients": num_clients,
        "state_policy": state_policy,
        "train_client_ids": train_client_ids,
        "holdout_client_ids": holdout_client_ids,
        "num_sampled_clients": num_sampled_clients,
        "logging_policy": logging_policy,
    }


def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return "".join(random.choice(chars) for _ in range(size))

def get_last_class_name(file_path):
    with open(file_path) as file:
        file_content = file.read()
    tree = ast.parse(file_content)
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    if classes:
        return classes[-1].name
    return None

def get_last_function_name(file_path):
    with open(file_path) as file:
        file_content = file.read()
    tree = ast.parse(file_content)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if functions:
        return functions[-1].name
    return None

def create_instance_from_file(file_path, class_name=None, *args, **kwargs):
    if class_name is None:
        class_name = get_last_class_name(file_path)
    file_path = os.path.abspath(file_path)
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    module_dir, module_file = os.path.split(file_path)
    module_name, _ = os.path.splitext(module_file)
    if module_dir not in sys.path:
        sys.path.append(module_dir)
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cls = getattr(module, class_name)
    instance = cls(*args, **kwargs)
    return instance

def get_function_from_file(file_path, function_name=None):
    try:
        if function_name is None:
            function_name = get_last_function_name(file_path)
        file_path = os.path.abspath(file_path)
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        module_dir, module_file = os.path.split(file_path)
        module_name, _ = os.path.splitext(module_file)
        if module_dir not in sys.path:
            sys.path.append(module_dir)
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        function = getattr(module, function_name)
        return function
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def run_function_from_file(file_path, function_name=None, *args, **kwargs):
    try:
        if function_name is None:
            function_name = get_last_function_name(file_path)
        file_path = os.path.abspath(file_path)
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        module_dir, module_file = os.path.split(file_path)
        module_name, _ = os.path.splitext(module_file)
        if module_dir not in sys.path:
            sys.path.append(module_dir)
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        function = getattr(module, function_name)
        result = function(*args, **kwargs)
        return result
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def create_instance_from_file_source(source, class_name=None, *args, **kwargs):
    _home = Path.home()
    dirname = osp.join(_home, ".appfl", "tmp")
    try:
        if not osp.exists(dirname):
            Path(dirname).mkdir(parents=True, exist_ok=True)
    except OSError:
        dirname = osp.join("/tmp", ".appfl", "tmp")
        if not osp.exists(dirname):
            Path(dirname).mkdir(parents=True, exist_ok=True)
    file_path = osp.join(dirname, f"{id_generator()}.py")
    with open(file_path, "w") as file:
        file.write(source)
    instance = create_instance_from_file(file_path, class_name, *args, **kwargs)
    os.remove(file_path)
    return instance

def get_function_from_file_source(source, function_name=None):
    _home = Path.home()
    dirname = osp.join(_home, ".appfl", "tmp")
    try:
        if not osp.exists(dirname):
            Path(dirname).mkdir(parents=True, exist_ok=True)
    except OSError:
        dirname = osp.join("/tmp", ".appfl", "tmp")
        if not osp.exists(dirname):
            Path(dirname).mkdir(parents=True, exist_ok=True)
    file_path = osp.join(dirname, f"{id_generator()}.py")
    with open(file_path, "w") as file:
        file.write(source)
    function = get_function_from_file(file_path, function_name)
    os.remove(file_path)
    return function

def run_function_from_file_source(source, function_name=None, *args, **kwargs):
    function = get_function_from_file_source(source, function_name)
    if function is None:
        return None
    result = function(*args, **kwargs)
    return result

def get_appfl_aggregator(
    aggregator_name: str,
    model: Optional[Any],
    aggregator_config: DictConfig,
    logger: Optional[Any] = None,
):
    try:
        appfl_module = importlib.import_module("appfl_sim.algorithm.aggregator")
        AggregatorClass = getattr(appfl_module, aggregator_name)
        return AggregatorClass(model, aggregator_config, logger)
    except AttributeError:
        raise ValueError(f"Invalid aggregator name: {aggregator_name}")

def get_appfl_scheduler(
    scheduler_name: str,
    scheduler_config: DictConfig,
    aggregator: Optional[Any] = None,
    logger: Optional[Any] = None,
):
    try:
        appfl_module = importlib.import_module("appfl_sim.algorithm.scheduler")
        SchedulerClass = getattr(appfl_module, scheduler_name)
        return SchedulerClass(scheduler_config, aggregator, logger)
    except AttributeError:
        raise ValueError(f"Invalid scheduler name: {scheduler_name}")

def _rebind_client_for_on_demand_job(
    client: ClientAgent,
    *,
    client_id: int,
    client_datasets: Sequence,
    num_workers_override: Optional[int] = None,
) -> None:
    dataset_entry = client_datasets[int(client_id)]
    if len(dataset_entry) == 1:
        train_ds = dataset_entry[0]
        val_ds = None
        test_ds = None
    elif len(dataset_entry) == 2:
        train_ds, test_ds = dataset_entry
        val_ds = None
    elif len(dataset_entry) == 3:
        train_ds, val_ds, test_ds = dataset_entry
    else:
        raise ValueError(
            "Each client dataset entry must be tuple(train), tuple(train,test), or tuple(train,val,test)."
        )

    client.id = int(client_id)
    client.client_agent_config.client_id = str(int(client_id))
    client.train_dataset = train_ds
    client.val_dataset = val_ds
    client.test_dataset = test_ds

    trainer = getattr(client, "trainer", None)
    if trainer is None:
        return
    trainer.client_id = str(int(client_id))
    trainer.train_dataset = train_ds
    trainer.val_dataset = val_ds
    trainer.test_dataset = test_ds

    cfg = client.client_agent_config.train_configs
    if num_workers_override is None:
        num_workers = int(cfg.get("num_workers", 0))
    else:
        num_workers = max(0, int(num_workers_override))
    train_bs = int(cfg.get("batch_size", 32))
    val_bs = int(cfg.get("eval_batch_size", train_bs))
    train_shuffle = bool(cfg.get("train_data_shuffle", True))
    train_pin_memory = bool(cfg.get("train_pin_memory", False))
    eval_pin_memory = bool(cfg.get("eval_pin_memory", train_pin_memory))
    persistent_workers = bool(cfg.get("dataloader_persistent_workers", False))
    prefetch_factor = int(cfg.get("dataloader_prefetch_factor", 2))

    common_train_kwargs = {
        "batch_size": max(1, train_bs),
        "shuffle": train_shuffle,
        "num_workers": num_workers,
        "pin_memory": train_pin_memory,
    }
    common_eval_kwargs = {
        "batch_size": max(1, val_bs),
        "num_workers": num_workers,
        "pin_memory": eval_pin_memory,
    }
    if num_workers > 0:
        common_train_kwargs["persistent_workers"] = persistent_workers
        common_eval_kwargs["persistent_workers"] = persistent_workers
        common_train_kwargs["prefetch_factor"] = max(2, prefetch_factor)
        common_eval_kwargs["prefetch_factor"] = max(2, prefetch_factor)

    trainer.train_dataloader = DataLoader(
        train_ds,
        **common_train_kwargs,
    )
    trainer.val_dataloader = (
        DataLoader(
            val_ds,
            shuffle=False,
            **common_eval_kwargs,
        )
        if val_ds is not None
        else None
    )
    trainer.test_dataloader = (
        DataLoader(
            test_ds,
            shuffle=False,
            **common_eval_kwargs,
        )
        if test_ds is not None
        else None
    )

def _build_on_demand_worker_pool(
    config: DictConfig,
    model,
    client_datasets: Sequence,
    local_client_ids: Sequence[int],
    device: str,
    run_log_dir: str,
    client_logging_enabled: bool,
    trainer_name: str,
    pool_size: int,
    num_workers_override: Optional[int] = None,
) -> List[ClientAgent]:
    if int(pool_size) <= 0:
        return []
    available_ids = [int(cid) for cid in local_client_ids]
    if not available_ids:
        return []
    ids: List[int] = []
    for idx in range(int(pool_size)):
        ids.append(int(available_ids[idx % len(available_ids)]))
    return _build_clients(
        config=config,
        model=model,
        client_datasets=client_datasets,
        local_client_ids=np.asarray(ids).astype(int),
        device=device,
        run_log_dir=run_log_dir,
        client_logging_enabled=client_logging_enabled,
        trainer_name=trainer_name,
        share_model=True,
        num_workers_override=num_workers_override,
    )

def _build_clients(
    config: DictConfig,
    model,
    client_datasets: Sequence,
    local_client_ids,
    device: str,
    run_log_dir: str,
    client_logging_enabled: bool = True,
    trainer_name: str = "VanillaTrainer",
    share_model: bool = False,
    num_workers_override: Optional[int] = None,
):
    from appfl_sim.agent import ClientAgent

    train_cfg = _build_train_cfg(
        config,
        device=device,
        run_log_dir=run_log_dir,
        num_workers_override=num_workers_override,
    )
    train_cfg["client_logging_enabled"] = bool(client_logging_enabled)
    clients = []
    for cid in local_client_ids:
        dataset_entry = client_datasets[int(cid)]
        if len(dataset_entry) == 1:
            train_ds = dataset_entry[0]
            val_ds = None
            test_ds = None
        elif len(dataset_entry) == 2:
            train_ds, test_ds = dataset_entry
            val_ds = None
        elif len(dataset_entry) == 3:
            train_ds, val_ds, test_ds = dataset_entry
        else:
            raise ValueError(
                "Each client dataset entry must be tuple(train), tuple(train,test), or tuple(train,val,test)."
            )
        client_cfg = OmegaConf.create(
            {
                "train_configs": {
                    **train_cfg,
                    "trainer": str(trainer_name),
                },
                "model_configs": {},
                "data_configs": {},
            }
        )
        client_cfg.client_id = str(int(cid))
        client_cfg.experiment_id = str(_cfg_get(config, "experiment.name", "appfl-sim"))
        client = ClientAgent(client_agent_config=client_cfg)
        client.model = model if share_model else copy.deepcopy(model)
        client.train_dataset = train_ds
        client.val_dataset = val_ds
        client.test_dataset = test_ds
        client.trainer = None
        client._load_trainer()
        client.id = int(cid)
        clients.append(
            client
        )
    return clients

def _build_single_client(
    config: DictConfig,
    model,
    client_datasets: Sequence,
    client_id: int,
    device: str,
    run_log_dir: str,
    client_logging_enabled: bool = True,
    trainer_name: str = "VanillaTrainer",
    share_model: bool = False,
    num_workers_override: Optional[int] = None,
):
    clients = _build_clients(
        config=config,
        model=model,
        client_datasets=client_datasets,
        local_client_ids=np.asarray([int(client_id)]).astype(int),
        device=device,
        run_log_dir=run_log_dir,
        client_logging_enabled=bool(client_logging_enabled),
        trainer_name=str(trainer_name),
        share_model=bool(share_model),
        num_workers_override=num_workers_override,
    )
    if not clients:
        raise RuntimeError(f"Failed to construct client for id={client_id}")
    return clients[0]

def _build_server(
    config: DictConfig,
    runtime_cfg: Dict,
    model,
    server_dataset,
    algorithm_components: Optional[Dict[str, Any]] = None,
) -> ServerAgent:
    from appfl_sim.agent import ServerAgent

    if algorithm_components is None:
        algorithm_components = _resolve_algorithm_components(config)
    num_clients = int(runtime_cfg["num_clients"])
    num_sampled_clients = _resolve_num_sampled_clients(config, num_clients=num_clients)
    loss_configs = _cfg_get(config, "loss.configs", {})
    if isinstance(loss_configs, DictConfig):
        loss_configs = dict(OmegaConf.to_container(loss_configs, resolve=True))
    elif isinstance(loss_configs, dict):
        loss_configs = dict(loss_configs)
    else:
        loss_configs = {}
    server_cfg = OmegaConf.create(
        {
            "client_configs": {
                "train_configs": {
                    "eval_metrics": _cfg_get(config, "eval.metrics", ["acc1"]),
                    "loss_name": str(_cfg_get(config, "loss.name", "CrossEntropyLoss")),
                    "loss_backend": str(_cfg_get(config, "loss.backend", "auto")),
                    "loss_path": str(_cfg_get(config, "loss.path", "")),
                    "loss_configs": loss_configs,
                },
                "model_configs": {},
            },
            "server_configs": {
                "num_clients": num_clients,
                "num_global_epochs": int(_cfg_get(config, "train.num_rounds", 20)),
                "num_sampled_clients": int(num_sampled_clients),
                "device": str(_cfg_get(config, "experiment.server_device", "cpu")),
                "eval_show_progress": _cfg_bool(config, "eval.show_eval_progress", True),
                "eval_batch_size": int(
                    _cfg_get(
                        config, "train.eval_batch_size", _cfg_get(config, "train.batch_size", 32)
                    )
                ),
                "num_workers": int(_cfg_get(config, "train.num_workers", 0)),
                "eval_metrics": _cfg_get(config, "eval.metrics", ["acc1"]),
                "aggregator": str(algorithm_components["aggregator_name"]),
                "aggregator_kwargs": dict(algorithm_components["aggregator_kwargs"]),
                "scheduler": str(algorithm_components["scheduler_name"]),
                "scheduler_kwargs": {
                    **dict(algorithm_components["scheduler_kwargs"]),
                    "num_clients": num_clients,
                },
            },
        }
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
    server.loss_fn = build_loss_from_config(config)
    server._val_dataset = server_dataset
    server._load_val_data()
    return server
