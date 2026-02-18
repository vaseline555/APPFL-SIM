import io
import gc
import os
import torch
import pathlib
import warnings
import threading
import numpy as np
import torch.nn as nn
from appfl_sim.agent.config import ServerAgentConfig
from appfl_sim.logger import ServerAgentFileLogger
from appfl_sim.algorithm.scheduler import BaseScheduler
from appfl_sim.algorithm.aggregator import BaseAggregator
from appfl_sim.metrics import MetricsManager, parse_metric_names
from appfl_sim.misc.utils import (
    create_instance_from_file,
    create_instance_from_file_source,
    get_function_from_file,
    run_function_from_file,
    run_function_from_file_source,
    get_appfl_aggregator,
    get_appfl_scheduler,
)
from concurrent.futures import Future
from torch.utils.data import DataLoader
from omegaconf import OmegaConf, DictConfig
from typing import Union, Dict, OrderedDict, Tuple, Optional, Any

try:
    from tqdm.auto import tqdm as _tqdm
except Exception:  # pragma: no cover
    _tqdm = None

try:
    from appfl_sim.misc.data_readiness.report import (
        get_unique_file_path,
        generate_html_content,
        save_html_report,
    )
except Exception:  # pragma: no cover
    def _missing_data_readiness(*args, **kwargs):
        raise ImportError(
            "Data-readiness report utilities require optional dependencies (e.g., matplotlib)."
        )

    get_unique_file_path = _missing_data_readiness
    generate_html_content = _missing_data_readiness
    save_html_report = _missing_data_readiness


class ServerAgent:
    """
    `ServerAgent` should act on behalf of the FL server to:
    - provide configurations that are shared among all clients to the clients (e.g. trainer, model, etc.) `ServerAgent.get_client_configs`
    - take the local model from a client, update the global model, and return it `ServerAgent.global_update`
    - provide the global model to the clients (no input and no aggregation) `ServerAgent.get_parameters`

    User can overwrite any class method to customize the behavior of the server agent.
    """

    def __init__(
        self, server_agent_config: Optional[ServerAgentConfig] = None
    ) -> None:
        self.server_agent_config = (
            server_agent_config if server_agent_config is not None else ServerAgentConfig()
        )
        # Check for optimize_memory in server_configs, default to True
        self.optimize_memory = getattr(
            self.server_agent_config.server_configs, "optimize_memory", True
        )
        if hasattr(self.server_agent_config.client_configs, "comm_configs"):
            self.server_agent_config.server_configs.comm_configs = (
                OmegaConf.merge(
                    self.server_agent_config.server_configs.comm_configs,
                    self.server_agent_config.client_configs.comm_configs,
                )
                if hasattr(self.server_agent_config.server_configs, "comm_configs")
                else self.server_agent_config.client_configs.comm_configs
            )
        self._set_num_clients()
        self._prepare_configs()
        self._create_logger()
        self._load_model()
        self._load_loss()
        self._load_metric()
        self._load_trainer()
        self._load_scheduler()
        self._load_val_data()

    def get_num_clients(self) -> int:
        """
        Get the number of clients.
        """
        if not hasattr(self, "num_clients"):
            self._set_num_clients()
        return self.num_clients

    def get_client_configs(self, **kwargs) -> DictConfig:
        """Return the FL configurations that are shared among all clients."""
        return self.server_agent_config.client_configs

    def global_update(
        self,
        client_id: Union[int, str],
        local_model: Union[Dict, OrderedDict, bytes],
        blocking: bool = False,
        **kwargs,
    ) -> Union[Future, Dict, OrderedDict, Tuple[Union[Dict, OrderedDict], Dict]]:
        """
        Update the global model using the local model from a client and return the updated global model.
        :param: client_id: A unique client id for server to distinguish clients, which be obtained via `ClientAgent.get_id()`.
        :param: local_model: The local model from a client, can be serialized bytes.
        :param: blocking: The global model may not be immediately available for certain aggregation methods (e.g. any synchronous method).
            Setting `blocking` to `True` will block the client until the global model is available.
            Otherwise, the method may return a `Future` object if the most up-to-date global model is not yet available.
        :return: The updated global model (as a Dict or OrderedDict), and optional metadata (as a Dict) if `blocking` is `True`.
            Otherwise, return the `Future` object of the updated global model and optional metadata.
        """
        if self.training_finished():
            global_model = self.scheduler.get_parameters(init_model=False)
            return global_model
        else:
            if isinstance(local_model, bytes):
                local_model = self._bytes_to_model(local_model)
            global_model = self.scheduler.schedule(client_id, local_model, **kwargs)

            # Memory optimization: Clean up local model after scheduling
            if self.optimize_memory:
                del local_model
                gc.collect()
            if not isinstance(global_model, Future):
                return global_model
            if blocking:
                return global_model.result()  # blocking until the `Future` is done
            else:
                return global_model  # return the `Future` object

    def get_parameters(
        self, blocking: bool = False, **kwargs
    ) -> Union[Future, Dict, OrderedDict, Tuple[Union[Dict, OrderedDict], Dict]]:
        """
        Return the global model to the clients.
        :param: `blocking`: The global model may not be immediately available (e.g. if the server wants to wait for all client
            to send the `get_parameters` request before returning the global model for same model initialization).
            Setting `blocking` to `True` will block the client until the global model is available.
        :param: `kwargs`: Additional arguments for the method. Specifically,
            - `init_model`: whether getting the initial model (which should be same among all clients, thus blocking)
            - `serial_run`: set `True` if for serial simulation run, thus no blocking is needed.
            - `globus_compute_run`: set `True` if for globus compute run, thus no blocking is needed.
        """
        global_model = self.scheduler.get_parameters(**kwargs)
        if not isinstance(global_model, Future):
            return global_model
        if blocking:
            return global_model.result()  # blocking until the `Future` is done
        else:
            return global_model  # return the `Future` object

    def set_sample_size(
        self,
        client_id: Union[int, str],
        sample_size: int,
        sync: bool = False,
        blocking: bool = False,
        **kwargs,
    ) -> Optional[Union[Dict, Future]]:
        """
        Set the size of the local dataset of a client.
        :param: client_id: A unique client id for server to distinguish clients, which can be obtained via `ClientAgent.get_id()`.
        :param: sample_size: The size of the local dataset of a client.
        :param: sync: Whether to synchronize the sample size among all clients. If `True`, the method can return the relative weight of the client.
        :param: blocking: Whether to block the client until the sample size of all clients is synchronized.
            If `True`, the method will return the relative weight of the client.
            Otherwise, the method may return a `Future` object of the relative weight, which will be resolved
            when the sample size of all clients is synchronized.
        """
        self.aggregator.set_client_sample_size(client_id, sample_size)
        if sync:
            if not hasattr(self, "_client_sample_size"):
                self._client_sample_size = {}
                self._client_sample_size_future = {}
                self._client_sample_size_lock = threading.Lock()
            with self._client_sample_size_lock:
                self._client_sample_size[client_id] = sample_size
                future = Future()
                self._client_sample_size_future[client_id] = future
                if len(self._client_sample_size) == self.get_num_clients():
                    total_sample_size = sum(self._client_sample_size.values())
                    for client_id in self._client_sample_size_future:
                        self._client_sample_size_future[client_id].set_result(
                            {
                                "client_weight": self._client_sample_size[client_id]
                                / total_sample_size
                            }
                        )
                    self._client_sample_size = {}
                    self._client_sample_size_future = {}
            if blocking:
                return future.result()
            else:
                return future
        return None

    def aggregate(
        self,
        local_states: Dict[Union[int, str], Union[Dict, OrderedDict]],
        sample_sizes: Dict[Union[int, str], int],
    ) -> Dict[int, float]:
        """
        Aggregate local client updates using the configured APPFL aggregator.
        Returns normalized aggregation weights for logging.
        """
        if not local_states:
            return {}
        if not hasattr(self, "aggregator") or self.aggregator is None:
            raise RuntimeError("ServerAgent aggregator is not initialized.")

        if (
            hasattr(self.aggregator, "model")
            and getattr(self.aggregator, "model", None) is None
            and hasattr(self, "model")
            and self.model is not None
        ):
            self.aggregator.model = self.model

        total = float(sum(int(sample_sizes.get(cid, 0)) for cid in local_states))
        if total <= 0.0:
            weights = {int(cid): 1.0 / len(local_states) for cid in local_states}
        else:
            weights = {
                int(cid): float(int(sample_sizes.get(cid, 0))) / total
                for cid in local_states
            }

        for cid, size in sample_sizes.items():
            self.aggregator.set_client_sample_size(cid, int(size))

        aggregated = self.aggregator.aggregate(local_states)
        if isinstance(aggregated, tuple):
            aggregated = aggregated[0]
        if (
            isinstance(aggregated, dict)
            and hasattr(self, "model")
            and self.model is not None
        ):
            self.model.load_state_dict(aggregated, strict=False)
        return weights

    @torch.no_grad()
    def evaluate(self, round_idx: Optional[int] = None) -> Dict[str, Any]:
        return self._evaluate_metrics(round_idx=round_idx)

    def _evaluate_metrics(self, round_idx: Optional[int] = None) -> Dict[str, Any]:
        if not hasattr(self, "_val_dataset") or self._val_dataset is None:
            return {"loss": -1.0, "accuracy": -1.0, "num_examples": 0, "metrics": {}}
        if len(self._val_dataset) == 0:
            return {"loss": -1.0, "accuracy": -1.0, "num_examples": 0, "metrics": {}}
        if not hasattr(self, "model") or self.model is None:
            return {"loss": -1.0, "accuracy": -1.0, "num_examples": 0, "metrics": {}}

        if not hasattr(self, "loss_fn") or self.loss_fn is None:
            self.loss_fn = torch.nn.CrossEntropyLoss()

        if not hasattr(self, "_val_dataloader"):
            self._val_dataloader = DataLoader(
                self._val_dataset,
                batch_size=int(
                    self.server_agent_config.server_configs.get("eval_batch_size", 128)
                ),
                shuffle=False,
                num_workers=int(
                    self.server_agent_config.server_configs.get("num_workers", 0)
                ),
            )

        device = torch.device(
            str(self.server_agent_config.server_configs.get("device", "cpu"))
        )
        eval_metric_names = parse_metric_names(
            self.server_agent_config.server_configs.get(
                "eval_metrics",
                self.server_agent_config.client_configs.train_configs.get(
                    "eval_metrics", None
                )
                if hasattr(self.server_agent_config.client_configs, "train_configs")
                else None,
            )
        )
        default_eval_metric = str(
            self.server_agent_config.server_configs.get(
                "default_eval_metric",
                self.server_agent_config.client_configs.train_configs.get(
                    "default_eval_metric", "acc1"
                )
                if hasattr(self.server_agent_config.client_configs, "train_configs")
                else "acc1",
            )
        )
        if default_eval_metric.strip().lower() in {"none", "null"}:
            default_eval_metric = ""

        manager = MetricsManager(
            eval_metrics=eval_metric_names,
            default_eval_metric=default_eval_metric,
        )
        was_training = self.model.training
        self.model.to(device)
        if hasattr(self.loss_fn, "to"):
            self.loss_fn = self.loss_fn.to(device)
        self.model.eval()

        total_correct = 0
        total_has_logits = False
        total_examples = 0
        target_pred = []
        target_true = []
        show_progress = bool(
            self.server_agent_config.server_configs.get("eval_show_progress", True)
        )
        progress_bar = None
        iterator = self._val_dataloader
        if show_progress and _tqdm is not None:
            try:
                total_batches = len(self._val_dataloader)
            except Exception:
                total_batches = None
            if round_idx is None:
                desc = "appfl-sim: ✅[Server | Global Eval]"
            else:
                desc = f"appfl-sim: ✅[Server | Round {int(round_idx):04d} | Global Eval]"
            progress_bar = _tqdm(
                self._val_dataloader,
                total=total_batches,
                desc=desc,
                leave=False,
                dynamic_ncols=True,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            )
            iterator = progress_bar

        try:
            for inputs, targets in iterator:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                logits = self.model(inputs)
                loss = self.loss_fn(logits, targets)
                logits_cpu = logits.detach().cpu()
                targets_cpu = targets.detach().cpu()

                manager.track(float(loss.item()), logits_cpu, targets_cpu)
                bs = targets_cpu.size(0)
                total_examples += bs
                target_pred.append(logits_cpu.numpy())
                target_true.append(targets_cpu.numpy())
                if logits_cpu.ndim > 1:
                    total_has_logits = True
                    total_correct += int(
                        torch.eq(torch.argmax(logits_cpu, dim=1), targets_cpu.view(-1))
                        .sum()
                        .item()
                    )
        finally:
            if progress_bar is not None:
                progress_bar.close()

        result = manager.aggregate(total_len=total_examples)
        if float(result.get("accuracy", -1.0)) < 0.0:
            if total_has_logits:
                result["accuracy"] = float(total_correct / max(total_examples, 1))
            elif self.metric is not None and target_true and target_pred:
                try:
                    result["accuracy"] = float(
                        self.metric(np.concatenate(target_true), np.concatenate(target_pred))
                    )
                except Exception:
                    pass

        if was_training:
            self.model.train()
        self.model.to("cpu")
        if hasattr(self.loss_fn, "to"):
            self.loss_fn = self.loss_fn.to("cpu")
        return result

    def server_validate(self):
        """
        Validate the server model using the validation dataset.
        """
        if not hasattr(self, "_val_dataset"):
            self.logger.info("No validation dataset is provided.")
            return None
        else:
            return self._validate()

    def training_finished(self, **kwargs) -> bool:
        """Indicate whether the training is finished."""
        return (
            self.server_agent_config.server_configs.num_global_epochs
            <= self.scheduler.get_num_global_epochs()
        )

    def close_connection(self, client_id: Union[int, str]) -> None:
        """Record the client that has finished the communication with the server."""
        if not hasattr(self, "closed_clients"):
            self.closed_clients = set()
            self._close_connection_lock = threading.Lock()
        with self._close_connection_lock:
            self.closed_clients.add(client_id)

    def data_readiness_report(self, readiness_report: Dict) -> None:
        """
        Generate the data readiness report and save it to the output directory.
        """
        output_dir = self.server_agent_config.client_configs.data_readiness_configs.get(
            "output_dirname", "./output"
        )
        output_filename = (
            self.server_agent_config.client_configs.data_readiness_configs.get(
                "output_filename", "data_readiness_report"
            )
        )

        if not os.path.exists(output_dir):
            pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Save JSON report
        # json_file_path = get_unique_file_path(output_dir, output_filename, "json")
        # save_json_report(json_file_path, readiness_report, self.logger)

        # Generate and save HTML report
        html_file_path = get_unique_file_path(output_dir, output_filename, "html")
        html_content = generate_html_content(readiness_report)
        save_html_report(html_file_path, html_content, self.logger)

        self._data_readiness_reports = {}

    def server_terminated(self):
        """Indicate whether the server can be terminated from listening to the clients."""
        if not hasattr(self, "closed_clients"):
            return False
        with self._close_connection_lock:
            terminated = len(self.closed_clients) >= self.get_num_clients()
        if terminated:
            self.clean_up()
        return terminated

    def clean_up(self) -> None:
        """
        Nececessary clean-up operations.
        No need to call this method if using `server_terminated` to check the termination status.
        """
        if not hasattr(self, "cleaned"):
            self.cleaned = False
        if not self.cleaned:
            self.cleaned = True
            if hasattr(self.scheduler, "clean_up"):
                self.scheduler.clean_up()

    def _create_logger(self) -> None:
        kwargs = {}
        if hasattr(self.server_agent_config.server_configs, "logging_output_dirname"):
            kwargs["file_dir"] = (
                self.server_agent_config.server_configs.logging_output_dirname
            )
        if hasattr(self.server_agent_config.server_configs, "logging_output_filename"):
            kwargs["file_name"] = (
                self.server_agent_config.server_configs.logging_output_filename
            )
        self.logger = ServerAgentFileLogger(**kwargs)

    def _load_model(self) -> None:
        """
        Load model from the definition file, and read the source code of the model for sendind to the client.
        User can overwrite this method to load the model from other sources.
        """
        if hasattr(self, "model") and self.model is not None:
            return
        if hasattr(self.server_agent_config.client_configs, "model_configs"):
            self._set_seed()
            model_configs = (
                self.server_agent_config.client_configs.model_configs
                if hasattr(self.server_agent_config.client_configs, "model_configs")
                else self.server_agent_config.server_configs.model_configs
            )
            # Allow runtime-injected models (e.g., simulator sets `server.model` later)
            # without requiring `model_path` in configs.
            if not (
                hasattr(model_configs, "model_path")
                or hasattr(model_configs, "model_source")
            ):
                self.model = None
                return
            if hasattr(model_configs, "model_name"):
                if hasattr(model_configs, "model_path"):
                    self.model = create_instance_from_file(
                        model_configs.model_path,
                        model_configs.model_name,
                        **(
                            model_configs.model_kwargs
                            if hasattr(model_configs, "model_kwargs")
                            else {}
                        ),
                    )
                else:
                    self.model = create_instance_from_file_source(
                        model_configs.model_source,
                        model_configs.model_name,
                        **(
                            model_configs.model_kwargs
                            if hasattr(model_configs, "model_kwargs")
                            else {}
                        ),
                    )
            else:
                if hasattr(model_configs, "model_path"):
                    self.model = run_function_from_file(
                        model_configs.model_path,
                        None,
                        **(
                            model_configs.model_kwargs
                            if hasattr(model_configs, "model_kwargs")
                            else {}
                        ),
                    )
                else:
                    self.model = run_function_from_file_source(
                        model_configs.model_source,
                        None,
                        **(
                            model_configs.model_kwargs
                            if hasattr(model_configs, "model_kwargs")
                            else {}
                        ),
                    )
            # load the model source file and delete model path
            if hasattr(self.server_agent_config.client_configs, "model_configs") and hasattr(
                model_configs, "model_path"
            ):
                with open(model_configs.model_path) as f:
                    self.server_agent_config.client_configs.model_configs.model_source = f.read()
                del self.server_agent_config.client_configs.model_configs.model_path
        else:
            self.model = None

    def _load_loss(self) -> None:
        """
        Load loss function from various sources.
        - `loss_fn_path` and `loss_fn_name`: load the loss function from a file.
        - `loss_fn`: load the loss function from `torch.nn` module.
        - Users can define their own way to load the loss function from other sources.
        """
        if not hasattr(self.server_agent_config, "client_configs") or not hasattr(
            self.server_agent_config.client_configs, "train_configs"
        ):
            self.loss_fn = None
            return
        if hasattr(
            self.server_agent_config.client_configs.train_configs, "loss_fn_path"
        ):
            kwargs = self.server_agent_config.client_configs.train_configs.get(
                "loss_fn_kwargs", {}
            )
            self.loss_fn = create_instance_from_file(
                self.server_agent_config.client_configs.train_configs.loss_fn_path,
                self.server_agent_config.client_configs.train_configs.loss_fn_name
                if hasattr(
                    self.server_agent_config.client_configs.train_configs,
                    "loss_fn_name",
                )
                else None,
                **kwargs,
            )
            with open(
                self.server_agent_config.client_configs.train_configs.loss_fn_path
            ) as f:
                self.server_agent_config.client_configs.train_configs.loss_fn_source = (
                    f.read()
                )
            del self.server_agent_config.client_configs.train_configs.loss_fn_path
        elif hasattr(self.server_agent_config.client_configs.train_configs, "loss_fn"):
            kwargs = self.server_agent_config.client_configs.train_configs.get(
                "loss_fn_kwargs", {}
            )
            if hasattr(
                nn, self.server_agent_config.client_configs.train_configs.loss_fn
            ):
                self.loss_fn = getattr(
                    nn, self.server_agent_config.client_configs.train_configs.loss_fn
                )(**kwargs)
            else:
                self.loss_fn = None
        else:
            self.loss_fn = None

    def _load_metric(self) -> None:
        """
        Load metric function from a file.
        User can define their own way to load the metric function from other sources.
        """
        if not hasattr(self.server_agent_config, "client_configs") or not hasattr(
            self.server_agent_config.client_configs, "train_configs"
        ):
            self.metric = None
            return
        if hasattr(
            self.server_agent_config.client_configs.train_configs, "metric_path"
        ):
            self.metric = get_function_from_file(
                self.server_agent_config.client_configs.train_configs.metric_path,
                self.server_agent_config.client_configs.train_configs.metric_name
                if hasattr(
                    self.server_agent_config.client_configs.train_configs, "metric_name"
                )
                else None,
            )
            with open(
                self.server_agent_config.client_configs.train_configs.metric_path
            ) as f:
                self.server_agent_config.client_configs.train_configs.metric_source = (
                    f.read()
                )
            del self.server_agent_config.client_configs.train_configs.metric_path
        else:
            self.metric = None

    def _load_scheduler(self) -> None:
        """Obtain the scheduler."""
        if hasattr(self.server_agent_config.server_configs, "aggregator_path"):
            # Load the user-defined aggregator from the file
            self.aggregator = create_instance_from_file(
                self.server_agent_config.server_configs.aggregator_path,
                self.server_agent_config.server_configs.aggregator,
                aggregator_configs=OmegaConf.create(
                    self.server_agent_config.server_configs.aggregator_kwargs
                    if hasattr(
                        self.server_agent_config.server_configs, "aggregator_kwargs"
                    )
                    else {}
                ),
                logger=self.logger,
            )
        else:
            if hasattr(self.server_agent_config.server_configs, "aggregator"):
                self.aggregator: BaseAggregator = get_appfl_aggregator(
                    aggregator_name=self.server_agent_config.server_configs.aggregator,
                    model=self.model,
                    aggregator_config=OmegaConf.create(
                        self.server_agent_config.server_configs.aggregator_kwargs
                        if hasattr(
                            self.server_agent_config.server_configs, "aggregator_kwargs"
                        )
                        else {}
                    ),
                    logger=self.logger,
                )
            else:
                self.aggregator = None

        if hasattr(self.server_agent_config.server_configs, "scheduler_path"):
            # Load the user-defined scheduler from the file
            self.scheduler = create_instance_from_file(
                self.server_agent_config.server_configs.scheduler_path,
                self.server_agent_config.server_configs.scheduler,
                scheduler_configs=OmegaConf.create(
                    self.server_agent_config.server_configs.scheduler_kwargs
                    if hasattr(
                        self.server_agent_config.server_configs, "scheduler_kwargs"
                    )
                    else {}
                ),
                aggregator=self.aggregator,
                logger=self.logger,
            )
        else:
            if hasattr(self.server_agent_config.server_configs, "scheduler"):
                self.scheduler: BaseScheduler = get_appfl_scheduler(
                    scheduler_name=self.server_agent_config.server_configs.scheduler,
                    scheduler_config=OmegaConf.create(
                        self.server_agent_config.server_configs.scheduler_kwargs
                        if hasattr(
                            self.server_agent_config.server_configs, "scheduler_kwargs"
                        )
                        else {}
                    ),
                    aggregator=self.aggregator,
                    logger=self.logger,
                )
            else:
                self.scheduler = None

    def _load_trainer(self) -> None:
        """
        Process the trainer configurations if the trainer is provided locally as a user-defined class.
        """
        if not hasattr(self.server_agent_config, "client_configs") or not hasattr(
            self.server_agent_config.client_configs, "train_configs"
        ):
            self.loss_fn = None
            return
        if hasattr(
            self.server_agent_config.client_configs.train_configs, "trainer_path"
        ):
            with open(
                self.server_agent_config.client_configs.train_configs.trainer_path
            ) as f:
                self.server_agent_config.client_configs.train_configs.trainer_source = (
                    f.read()
                )
            del self.server_agent_config.client_configs.train_configs.trainer_path

    def _load_compressor(self) -> None:
        """Compression is disabled in appfl[sim]."""
        self.compressor = None
        self.enable_compression = False

    def _bytes_to_model(self, model_bytes: bytes) -> Union[Dict, OrderedDict]:
        """Deserialize the model from bytes (compression disabled)."""
        if self.optimize_memory:
            with io.BytesIO(model_bytes) as buffer:
                model = torch.load(buffer, map_location="cpu")
            gc.collect()
            return model
        return torch.load(io.BytesIO(model_bytes))

    def _load_val_data(self) -> None:
        if hasattr(self, "_val_dataset") and self._val_dataset is not None:
            self._val_dataloader = DataLoader(
                self._val_dataset,
                batch_size=int(
                    self.server_agent_config.server_configs.get("eval_batch_size", 128)
                ),
                shuffle=False,
                num_workers=int(
                    self.server_agent_config.server_configs.get("num_workers", 0)
                ),
            )
            return
        if hasattr(self.server_agent_config.server_configs, "val_data_configs"):
            self._val_dataset = run_function_from_file(
                self.server_agent_config.server_configs.val_data_configs.dataset_path,
                self.server_agent_config.server_configs.val_data_configs.dataset_name,
                **(
                    self.server_agent_config.server_configs.val_data_configs.dataset_kwargs
                    if hasattr(
                        self.server_agent_config.server_configs.val_data_configs,
                        "dataset_kwargs",
                    )
                    else {}
                ),
            )
            self._val_dataloader = DataLoader(
                self._val_dataset,
                batch_size=self.server_agent_config.server_configs.val_data_configs.get(
                    "batch_size", 1
                ),
                shuffle=self.server_agent_config.server_configs.val_data_configs.get(
                    "shuffle", False
                ),
                num_workers=self.server_agent_config.server_configs.val_data_configs.get(
                    "num_workers", 0
                ),
            )

    def _validate(self) -> Tuple[float, float]:
        """
        Validate the model
        :return: loss, accuracy
        """
        stats = self._evaluate_metrics()
        return float(stats["loss"]), float(stats["accuracy"])

    def _set_seed(self):
        """
        This function makes sure that all clients have the same initial model parameters.
        """
        seed_value = self.server_agent_config.client_configs.model_configs.get(
            "seed", 42
        )
        torch.manual_seed(seed_value)  # Set PyTorch seed
        torch.cuda.manual_seed_all(seed_value)  # Set seed for all GPUs
        torch.backends.cudnn.deterministic = True  # Use deterministic algorithms
        torch.backends.cudnn.benchmark = False  # Disable this to ensure reproducibility

    def _set_num_clients(self) -> None:
        """
        Set the number of clients.
        The recommended way is to set the number of clients in the server_configs.
        Give deprecation warnings if the number of clients is set in the scheduler_kwargs or aggregator_kwargs.
        """
        if not hasattr(self, "num_clients"):
            assert (
                hasattr(self.server_agent_config.server_configs, "num_clients")
                or (
                    hasattr(self.server_agent_config.server_configs, "scheduler_kwargs")
                    and hasattr(
                        self.server_agent_config.server_configs.scheduler_kwargs,
                        "num_clients",
                    )
                )
                or (
                    hasattr(
                        self.server_agent_config.server_configs, "aggregator_kwargs"
                    )
                    and hasattr(
                        self.server_agent_config.server_configs.aggregator_kwargs,
                        "num_clients",
                    )
                )
            ), "The number of clients should be set in the server configurations."
            self.num_clients = (
                self.server_agent_config.server_configs.num_clients
                if hasattr(self.server_agent_config.server_configs, "num_clients")
                else self.server_agent_config.server_configs.scheduler_kwargs.num_clients
                if (
                    hasattr(self.server_agent_config.server_configs, "scheduler_kwargs")
                    and hasattr(
                        self.server_agent_config.server_configs.scheduler_kwargs,
                        "num_clients",
                    )
                )
                else self.server_agent_config.server_configs.aggregator_kwargs.num_clients
            )
            # [Deprecation]: It is recommended to specify the number of clients once in server_configs
            if hasattr(
                self.server_agent_config.server_configs, "scheduler_kwargs"
            ) and hasattr(
                self.server_agent_config.server_configs.scheduler_kwargs,
                "num_clients",
            ):
                warnings.warn(
                    message="It is deprecated to specify the number of clients in the scheduler_kwargs. It is recommended to specify it in the server_configs.num_clients instead.",
                    category=DeprecationWarning,
                )
            if hasattr(
                self.server_agent_config.server_configs, "aggregator_kwargs"
            ) and hasattr(
                self.server_agent_config.server_configs.aggregator_kwargs,
                "num_clients",
            ):
                warnings.warn(
                    message="It is deprecated to specify the number of clients in the aggregator_kwargs. It is recommended to specify it in the server_configs.num_clients instead.",
                    category=DeprecationWarning,
                )
            # Set num_clients for aggregator and scheduler
            if hasattr(self.server_agent_config.server_configs, "scheduler_kwargs"):
                self.server_agent_config.server_configs.scheduler_kwargs.num_clients = (
                    self.num_clients
                )
            else:
                self.server_agent_config.server_configs.scheduler_kwargs = (
                    OmegaConf.create({"num_clients": self.num_clients})
                )
            if hasattr(self.server_agent_config.server_configs, "aggregator_kwargs"):
                self.server_agent_config.server_configs.aggregator_kwargs.num_clients = self.num_clients
            else:
                self.server_agent_config.server_configs.aggregator_kwargs = (
                    OmegaConf.create({"num_clients": self.num_clients})
                )
            # Set num_clients for server_configs
            self.server_agent_config.server_configs.num_clients = self.num_clients

    def _prepare_configs(self):
        """
        Prepare the configurations for the server agent.
        """
        if hasattr(
            self.server_agent_config.client_configs.train_configs, "send_gradient"
        ):
            if hasattr(self.server_agent_config.server_configs, "aggregator_kwargs"):
                if hasattr(
                    self.server_agent_config.server_configs.aggregator_kwargs,
                    "gradient_based",
                ):
                    warnings.warn(
                        message="There is no need to specify the gradient_based in the aggregator_kwargs. It is automatically set based on the send_gradient in the client_configs.train_configs.",
                        category=UserWarning,
                    )
                self.server_agent_config.server_configs.aggregator_kwargs.gradient_based = self.server_agent_config.client_configs.train_configs.send_gradient
            else:
                self.server_agent_config.server_configs.aggregator_kwargs = OmegaConf.create(
                    {
                        "gradient_based": self.server_agent_config.client_configs.train_configs.send_gradient
                    }
                )
        else:
            # Assert no gradient_based in the aggregator_kwargs
            assert not (
                hasattr(self.server_agent_config.server_configs, "aggregator_kwargs")
                and hasattr(
                    self.server_agent_config.server_configs.aggregator_kwargs,
                    "gradient_based",
                )
            ), (
                "The gradient_based should be set in the client_configs.train_configs.send_gradient."
            )

        if hasattr(
            self.server_agent_config.server_configs, "aggregator_kwargs"
        ) and hasattr(
            self.server_agent_config.server_configs.aggregator_kwargs, "use_secure_agg"
        ):
            if hasattr(self.server_agent_config.client_configs, "train_configs"):
                if hasattr(
                    self.server_agent_config.client_configs.train_configs,
                    "use_secure_agg",
                ):
                    warnings.warn(
                        message="There is no need to specify the use_secure_agg in client_configs.train_configs. It is automatically set based on the use_secure_agg in the server_configs.aggregator_kwargs.",
                        category=UserWarning,
                    )
                self.server_agent_config.client_configs.train_configs.use_secure_agg = self.server_agent_config.server_configs.aggregator_kwargs.use_secure_agg
            else:
                self.server_agent_config.client_configs.train_configs = OmegaConf.create(
                    {
                        "use_secure_agg": self.server_agent_config.server_configs.aggregator_kwargs.use_secure_agg
                    }
                )
        else:
            assert not (
                hasattr(self.server_agent_config.client_configs, "train_configs")
                and hasattr(
                    self.server_agent_config.client_configs.train_configs,
                    "use_secure_agg",
                )
            ), (
                "The use_secure_agg should be set in the server_configs.aggregator_kwargs.use_secure_agg."
            )

        if hasattr(
            self.server_agent_config.server_configs, "aggregator_kwargs"
        ) and hasattr(
            self.server_agent_config.server_configs.aggregator_kwargs,
            "secure_agg_client_weights_mode",
        ):
            if hasattr(self.server_agent_config.client_configs, "train_configs"):
                if hasattr(
                    self.server_agent_config.client_configs.train_configs,
                    "secure_agg_client_weights_mode",
                ):
                    warnings.warn(
                        message="There is no need to specify the secure_agg_client_weights_mode in client_configs.train_configs. It is automatically set based on the secure_agg_client_weights_mode in the server_configs.aggregator_kwargs.",
                        category=UserWarning,
                    )
                self.server_agent_config.client_configs.train_configs.secure_agg_client_weights_mode = self.server_agent_config.server_configs.aggregator_kwargs.secure_agg_client_weights_mode
            else:
                self.server_agent_config.client_configs.train_configs = OmegaConf.create(
                    {
                        "secure_agg_client_weights_mode": self.server_agent_config.server_configs.aggregator_kwargs.secure_agg_client_weights_mode
                    }
                )
        else:
            assert not (
                hasattr(self.server_agent_config.client_configs, "train_configs")
                and hasattr(
                    self.server_agent_config.client_configs.train_configs,
                    "secure_agg_client_weights_mode",
                )
            ), (
                "The secure_agg_client_weights_mode should be set in the server_configs.aggregator_kwargs.secure_agg_client_weights_mode."
            )
