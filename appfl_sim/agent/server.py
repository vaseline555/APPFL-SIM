import io
import gc
import torch
import threading
from appfl_sim.logger import ServerAgentFileLogger
from appfl_sim.algorithm.scheduler import BaseScheduler
from appfl_sim.algorithm.aggregator import BaseAggregator
from appfl_sim.metrics import MetricsManager, parse_metric_names
from appfl_sim.misc.runtime_utils import (
    _create_instance_from_file,
    _run_function_from_file,
    _create_aggregator_instance,
    _create_scheduler_instance,
)
from appfl_sim.misc.config_utils import build_loss_from_train_cfg
from concurrent.futures import Future
from torch.utils.data import DataLoader
from omegaconf import OmegaConf, DictConfig
from typing import Union, Dict, OrderedDict, Tuple, Optional, Any

try:
    from tqdm.auto import tqdm as _tqdm
except Exception:  # pragma: no cover
    _tqdm = None


class ServerAgent:
    """
    `ServerAgent` should act on behalf of the FL server to:
    - provide configurations that are shared among all clients to the clients (e.g. trainer, model, etc.) `ServerAgent.get_client_configs`
    - take the local model from a client, update the global model, and return it `ServerAgent.global_update`
    - provide the global model to the clients (no input and no aggregation) `ServerAgent.get_parameters`

    User can overwrite any class method to customize the behavior of the server agent.
    """

    def __init__(
        self, server_agent_config: Optional[DictConfig | Dict[str, Any]] = None
    ) -> None:
        if server_agent_config is None:
            self.server_agent_config = self._default_config()
        elif isinstance(server_agent_config, DictConfig):
            self.server_agent_config = server_agent_config
        else:
            self.server_agent_config = OmegaConf.create(server_agent_config)
        self.num_clients: Optional[int] = None
        self.model = None
        self.loss_fn = None
        self.aggregator = None
        self.scheduler = None
        self._val_dataset = None
        self._val_dataloader = None
        self._client_sample_size = {}
        self._client_sample_size_future = {}
        self._client_sample_size_lock = threading.Lock()
        self.closed_clients = set()
        self._close_connection_lock = threading.Lock()
        self.cleaned = False
        self._ensure_config_contract()
        self.optimize_memory = bool(
            self.server_agent_config.server_configs.get("optimize_memory", True)
        )
        self._set_num_clients()
        self._prepare_configs()
        self._create_logger()
        self._load_model()
        self._load_loss()
        self._load_scheduler()
        self._load_val_data()

    def _ensure_config_contract(self) -> None:
        if "server_configs" not in self.server_agent_config:
            raise ValueError("ServerAgentConfig is missing required section: server_configs")
        if "client_configs" not in self.server_agent_config:
            raise ValueError("ServerAgentConfig is missing required section: client_configs")
        client_cfg = self.server_agent_config.client_configs
        for name in ("train_configs", "model_configs"):
            if name not in client_cfg:
                raise ValueError(f"ServerAgentConfig.client_configs is missing required section: {name}")
            if client_cfg.get(name) is None:
                raise ValueError(f"ServerAgentConfig.client_configs.{name} must not be None.")
        if self.server_agent_config.server_configs is None:
            raise ValueError("ServerAgentConfig.server_configs must not be None.")
        for name in ("num_clients", "aggregator", "scheduler"):
            if name not in self.server_agent_config.server_configs:
                raise ValueError(f"ServerAgentConfig.server_configs.{name} is required.")

    def get_num_clients(self) -> int:
        """
        Get the number of clients.
        """
        if self.num_clients is None:
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
            global_model = self.scheduler.get_parameters()
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
        :param: `blocking`: The global model may not be immediately available.
            Setting `blocking` to `True` will block the client until the global model is available.
        """
        del kwargs
        global_model = self.scheduler.get_parameters()
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
        client_train_stats: Optional[Dict[Union[int, str], Dict[str, Any]]] = None,
    ) -> Dict[Union[int, str], float]:
        """
        Aggregate local client updates using the configured APPFL aggregator.
        Returns normalized aggregation weights for logging.
        """
        if not local_states:
            return {}
        if self.aggregator is None:
            raise RuntimeError("ServerAgent aggregator is not initialized.")

        if (
            hasattr(self.aggregator, "model")
            and getattr(self.aggregator, "model", None) is None
            and self.model is not None
        ):
            self.aggregator.model = self.model

        total = float(sum(int(sample_sizes.get(cid, 0)) for cid in local_states))
        if total <= 0.0:
            weights = {cid: 1.0 / len(local_states) for cid in local_states}
        else:
            weights = {
                cid: float(int(sample_sizes.get(cid, 0))) / total
                for cid in local_states
            }

        for cid, size in sample_sizes.items():
            self.aggregator.set_client_sample_size(cid, int(size))

        aggregated = self.aggregator.aggregate(
            local_states,
            client_train_stats=client_train_stats or {},
        )
        if isinstance(aggregated, tuple):
            aggregated = aggregated[0]
        if (
            isinstance(aggregated, dict)
            and self.model is not None
        ):
            self.model.load_state_dict(aggregated, strict=False)
        return weights

    @torch.no_grad()
    def evaluate(self, round_idx: Optional[int] = None) -> Dict[str, Any]:
        return self._evaluate_metrics(round_idx=round_idx)

    def _evaluate_metrics(self, round_idx: Optional[int] = None) -> Dict[str, Any]:
        if self._val_dataset is None:
            return {"loss": -1.0, "num_examples": 0, "metrics": {}}
        if len(self._val_dataset) == 0:
            return {"loss": -1.0, "num_examples": 0, "metrics": {}}
        if self.model is None:
            return {"loss": -1.0, "num_examples": 0, "metrics": {}}

        if self.loss_fn is None:
            self.loss_fn = torch.nn.CrossEntropyLoss()

        if self._val_dataloader is None:
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
                ),
            )
        )
        manager = MetricsManager(eval_metrics=eval_metric_names)
        was_training = self.model.training
        self.model.to(device)
        if hasattr(self.loss_fn, "to"):
            self.loss_fn = self.loss_fn.to(device)
        self.model.eval()

        total_examples = 0
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
                desc = f"appfl-sim: ✅[Server | Evaluation (Global)]"
            else:
                desc = f"appfl-sim: ✅[Server (Round {int(round_idx):04d}) Evaluation (Global)]"
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
        finally:
            if progress_bar is not None:
                progress_bar.close()

        result = manager.aggregate(total_len=total_examples)

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
        if self._val_dataset is None:
            self.logger.info("No validation dataset is provided.")
            return None
        else:
            stats = self._evaluate_metrics()
            metric_names = parse_metric_names(
                self.server_agent_config.server_configs.get("eval_metrics", None)
            )
            metric_name = (
                str(metric_names[0]).strip().lower() if metric_names else "acc1"
            )
            metric_value = -1.0
            nested = stats.get("metrics", {})
            if isinstance(nested, dict) and metric_name in nested:
                metric_value = float(nested[metric_name])
            elif f"metric_{metric_name}" in stats:
                metric_value = float(stats[f"metric_{metric_name}"])
            return float(stats["loss"]), metric_value

    def training_finished(self, **kwargs) -> bool:
        """Indicate whether the training is finished."""
        return (
            self.server_agent_config.server_configs.num_global_epochs
            <= self.scheduler.get_num_global_epochs()
        )

    def close_connection(self, client_id: Union[int, str]) -> None:
        """Record the client that has finished the communication with the server."""
        with self._close_connection_lock:
            self.closed_clients.add(client_id)

    def server_terminated(self):
        """Indicate whether the server can be terminated from listening to the clients."""
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
        if not self.cleaned:
            self.cleaned = True
            if hasattr(self.scheduler, "clean_up"):
                self.scheduler.clean_up()

    def _create_logger(self) -> None:
        kwargs = {}
        if self.server_agent_config.server_configs.get("logging_output_dirname", None) is not None:
            kwargs["file_dir"] = (
                self.server_agent_config.server_configs.logging_output_dirname
            )
        if self.server_agent_config.server_configs.get("logging_output_filename", None) is not None:
            kwargs["file_name"] = (
                self.server_agent_config.server_configs.logging_output_filename
            )
        self.logger = ServerAgentFileLogger(**kwargs)

    def _load_model(self) -> None:
        """
        Load model from the definition file, and read the source code of the model for sendind to the client.
        User can overwrite this method to load the model from other sources.
        """
        if self.model is not None:
            return
        self._set_seed()
        model_configs = self.server_agent_config.client_configs.model_configs
        if "model_path" not in model_configs:
            self.model = None
            return
        if "model_name" in model_configs:
            self.model = _create_instance_from_file(
                model_configs.model_path,
                model_configs.model_name,
                **model_configs.get("model_kwargs", {}),
            )
        else:
            self.model = _run_function_from_file(
                model_configs.model_path,
                None,
                **model_configs.get("model_kwargs", {}),
            )

    def _load_loss(self) -> None:
        """
        Load loss function from client train configuration.
        """
        train_cfg = self.server_agent_config.client_configs.train_configs
        self.loss_fn = build_loss_from_train_cfg(train_cfg)

    def _load_scheduler(self) -> None:
        """Obtain the scheduler."""
        server_cfg = self.server_agent_config.server_configs
        self.aggregator: BaseAggregator = _create_aggregator_instance(
            aggregator_name=server_cfg.aggregator,
            model=self.model,
            aggregator_config=OmegaConf.create(server_cfg.get("aggregator_kwargs", {})),
            logger=self.logger,
        )

        self.scheduler: BaseScheduler = _create_scheduler_instance(
            scheduler_name=server_cfg.scheduler,
            scheduler_config=OmegaConf.create(server_cfg.get("scheduler_kwargs", {})),
            aggregator=self.aggregator,
            logger=self.logger,
        )

    def _bytes_to_model(self, model_bytes: bytes) -> Union[Dict, OrderedDict]:
        """Deserialize the model from bytes (compression disabled)."""
        if self.optimize_memory:
            with io.BytesIO(model_bytes) as buffer:
                model = torch.load(buffer, map_location="cpu")
            gc.collect()
            return model
        return torch.load(io.BytesIO(model_bytes))

    def _load_val_data(self) -> None:
        if self._val_dataset is not None:
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
        if "val_data_configs" in self.server_agent_config.server_configs:
            self._val_dataset = _run_function_from_file(
                self.server_agent_config.server_configs.val_data_configs.dataset_path,
                self.server_agent_config.server_configs.val_data_configs.dataset_name,
                **self.server_agent_config.server_configs.val_data_configs.get(
                    "dataset_kwargs", {}
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
        The number of clients must be set in server_configs.
        """
        if self.num_clients is None:
            if "num_clients" not in self.server_agent_config.server_configs:
                raise ValueError("server_configs.num_clients is required.")
            self.num_clients = self.server_agent_config.server_configs.num_clients
            # Set num_clients for aggregator and scheduler
            if "scheduler_kwargs" not in self.server_agent_config.server_configs:
                self.server_agent_config.server_configs.scheduler_kwargs = OmegaConf.create({})
            if "aggregator_kwargs" not in self.server_agent_config.server_configs:
                self.server_agent_config.server_configs.aggregator_kwargs = OmegaConf.create({})
            self.server_agent_config.server_configs.scheduler_kwargs.num_clients = (
                self.num_clients
            )
            self.server_agent_config.server_configs.aggregator_kwargs.num_clients = (
                self.num_clients
            )
            # Set num_clients for server_configs
            self.server_agent_config.server_configs.num_clients = self.num_clients

    def _prepare_configs(self):
        """
        Prepare the configurations for the server agent.
        """
        train_cfg = self.server_agent_config.client_configs.train_configs
        agg_kwargs = self.server_agent_config.server_configs.get("aggregator_kwargs", {})
        if "send_gradient" in train_cfg:
            agg_kwargs["gradient_based"] = bool(train_cfg.send_gradient)
        if "use_secure_agg" in agg_kwargs:
            train_cfg.use_secure_agg = bool(agg_kwargs["use_secure_agg"])
        if "secure_agg_client_weights_mode" in agg_kwargs:
            train_cfg.secure_agg_client_weights_mode = str(
                agg_kwargs["secure_agg_client_weights_mode"]
            )

    @staticmethod
    def _default_config() -> DictConfig:
        return OmegaConf.create(
            {
                "client_configs": {
                    "train_configs": {
                        "trainer": "FedavgTrainer",
                        "device": "cpu",
                        "mode": "epoch",
                        "num_local_epochs": 1,
                        "batch_size": 32,
                        "eval_batch_size": 128,
                        "num_workers": 0,
                        "train_data_shuffle": True,
                        "train_pin_memory": False,
                        "eval_pin_memory": False,
                        "dataloader_persistent_workers": False,
                        "dataloader_prefetch_factor": 2,
                        "optimizer_name": "SGD",
                        "optimizer_backend": "auto",
                        "optimizer_path": "",
                        "optimizer_configs": {"weight_decay": 0.0},
                        "lr": 0.01,
                        "loss_name": "CrossEntropyLoss",
                        "loss_backend": "auto",
                        "loss_path": "",
                        "loss_configs": {},
                        "max_grad_norm": 0.0,
                        "client_logging_enabled": True,
                        "do_pre_evaluation": True,
                        "do_post_evaluation": True,
                        "eval_metrics": ["acc1"],
                    },
                    "model_configs": {},
                },
                "server_configs": {
                    "num_clients": 1,
                    "num_global_epochs": 1,
                    "num_sampled_clients": 1,
                    "device": "cpu",
                    "num_workers": 0,
                    "eval_batch_size": 128,
                    "eval_show_progress": True,
                    "eval_metrics": ["acc1"],
                    "aggregator": "FedavgAggregator",
                    "aggregator_kwargs": {},
                    "scheduler": "FedavgScheduler",
                    "scheduler_kwargs": {},
                },
            }
        )
