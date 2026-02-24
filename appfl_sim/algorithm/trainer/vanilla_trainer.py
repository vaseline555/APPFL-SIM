import gc
import copy
import time
import math
import torch
import numpy as np
from torch.nn import Module
from omegaconf import DictConfig
from typing import Tuple, Dict, Optional, Any, List
from torch.utils.data import Dataset, DataLoader
from appfl_sim.algorithm.trainer.base_trainer import BaseTrainer
from appfl_sim.metrics import MetricsManager, parse_metric_names
from appfl_sim.misc.config_utils import build_optimizer_from_train_cfg
from appfl_sim.misc.metrics_utils import _attach_prefixed_metrics
from appfl_sim.misc.logging_utils import (
    _build_trainer_log_row,
    _build_trainer_log_title,
)
from appfl_sim.misc.system_utils import (
    extract_model_state_optimized,
)

# Logging
try:
    import wandb
except Exception:  # pragma: no cover
    class _WandbStub:
        run = None

        @staticmethod
        def log(*args, **kwargs):
            return None

    wandb = _WandbStub()

def _make_dataloader(
    dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int,
):
    kwargs = {
        "batch_size": int(batch_size),
        "shuffle": bool(shuffle),
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
    }
    if int(num_workers) > 0:
        kwargs["persistent_workers"] = bool(persistent_workers)
        kwargs["prefetch_factor"] = max(2, int(prefetch_factor))
    return DataLoader(dataset, **kwargs)

class VanillaTrainer(BaseTrainer):
    """
    VanillaTrainer:
        Vanilla trainer for FL clients, which trains the model using `torch.optim`
        optimizers for a certain number of local epochs or local steps.
        Users need to specify which training model to use in the configuration,
        as well as the number of local epochs or steps.
    """

    def __init__(
        self,
        model: Optional[Module] = None,
        loss_fn: Optional[Module] = None,
        metric: Optional[Any] = None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        train_configs: Optional[DictConfig] = None,
        logger: Optional[Any] = None,
        **kwargs,
    ):
        if train_configs is None:
            train_configs = DictConfig({})
        super().__init__(
            model=model,
            loss_fn=loss_fn,
            metric=metric,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            train_configs=train_configs,
            logger=logger,
            **kwargs,
        )

        # Normalize local-update and clipping controls
        self.train_configs = self._normalize_train_configs(train_configs)
        self._validate_train_config()

        # Build local dataloaders 
        if self.train_dataset is None:
            raise ValueError("train_dataset must be provided for VanillaTrainer.")
        self.test_dataset = kwargs.get("test_dataset", None)
        (
            self.train_dataloader,
            self.val_dataloader,
            self.test_dataloader,
        ) = self._instantiate_dataloaders(
            self.train_dataset,
            self.val_dataset,
            self.test_dataset,
            self.train_configs,
        )

        # Initialize runtime state
        self.optimize_memory = bool(train_configs.get("optimize_memory", True))
        self.eval_metric_names = parse_metric_names(
            self.train_configs.get("eval_metrics", None)
        )
        self.model_state = None
        self.eval_results: Optional[Dict[str, Any]] = None

        # Initialize logging scheme
        if bool(self.train_configs.get("enable_wandb", False)):
            self.enabled_wandb = True
            self.wandb_logging_id = self.train_configs.wandb_logging_id
        else:
            self.enabled_wandb = False

        # Resolve device routing
        self.device = str(self.train_configs.get("device", "cpu")).strip().lower()
        if "," in self.device:
            raise ValueError(
                "Multi-device strings are unsupported in VanillaTrainer; use a single device (e.g., cpu, cuda, cuda:0)."
            )

    @staticmethod
    def _normalize_train_configs(train_configs: DictConfig) -> None:
        train_configs.mode = str(train_configs.get("mode", "epoch"))
        if train_configs.mode == "epoch":
            train_configs.num_local_epochs = int(train_configs.get("num_local_epochs", 1))
        else:
            train_configs.num_local_steps = int(train_configs.get("num_local_steps", 1))
        max_grad_norm = float(train_configs.get("max_grad_norm", 0.0))
        if max_grad_norm > 0.0 and not train_configs.get("clip_grad", False):
            train_configs.clip_grad = True
            train_configs.clip_value = float(max_grad_norm)
            train_configs.clip_norm = float(train_configs.get("clip_norm", 2.0))
        return train_configs
            
    @staticmethod
    def _instantiate_dataloaders(
        train_dataset: Dataset,
        val_dataset: Optional[Dataset],
        test_dataset: Optional[Dataset],
        train_configs: DictConfig,
    ) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
        num_workers = int(train_configs.get("num_workers", 0))
        train_pin_memory = bool(train_configs.get("train_pin_memory", False))
        eval_pin_memory = bool(train_configs.get("eval_pin_memory", train_pin_memory))
        persistent_workers = bool(train_configs.get("dataloader_persistent_workers", False))
        prefetch_factor = int(train_configs.get("dataloader_prefetch_factor", 2))
        train_loader = _make_dataloader(
            train_dataset,
            batch_size=train_configs.get("batch_size", 32),
            shuffle=train_configs.get("train_data_shuffle", True),
            num_workers=num_workers,
            pin_memory=train_pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        )
        val_loader = (
            _make_dataloader(
                val_dataset,
                batch_size=train_configs.get("eval_batch_size", 32),
                shuffle=False,
                num_workers=num_workers,
                pin_memory=eval_pin_memory,
                persistent_workers=persistent_workers,
                prefetch_factor=prefetch_factor,
            )
            if val_dataset is not None
            else None
        )
        test_loader = (
            _make_dataloader(
                test_dataset,
                batch_size=train_configs.get("eval_batch_size", 32),
                shuffle=False,
                num_workers=num_workers,
                pin_memory=eval_pin_memory,
                persistent_workers=persistent_workers,
                prefetch_factor=prefetch_factor,
            )
            if test_dataset is not None
            else None
        )
        return train_loader, val_loader, test_loader

    def _validate_train_config(self):
        """
        Check if the configurations are valid.
        """
        if "mode" not in self.train_configs:
            raise ValueError("Training mode must be specified.")
        if self.train_configs.mode not in {"epoch", "step"}:
            raise ValueError("Training mode must be either 'epoch' or 'step'.")
        if self.train_configs.mode == "epoch":
            if "num_local_epochs" not in self.train_configs:
                raise ValueError("Number of local epochs must be specified.")
        else:
            if "num_local_steps" not in self.train_configs:
                raise ValueError("Number of local steps must be specified.")
            
    def _new_metrics_manager(self) -> MetricsManager:
        return MetricsManager(eval_metrics=self.eval_metric_names)

    def _primary_metric_name(self) -> str:
        if self.eval_metric_names:
            return str(self.eval_metric_names[0]).strip().lower()
        return "acc1"

    def _metric_from_stats(self, stats: Optional[Dict[str, Any]]) -> float:
        if not isinstance(stats, dict):
            return -1.0
        metric_name = self._primary_metric_name()
        nested = stats.get("metrics", {})
        if isinstance(nested, dict):
            value = nested.get(metric_name, None)
            if isinstance(value, (int, float)):
                return float(value)
        for key in (f"metric_{metric_name}", metric_name):
            value = stats.get(key, None)
            if isinstance(value, (int, float)):
                return float(value)
        return -1.0

    def _resolve_eval_split(
        self,
        split: Optional[str] = None,
        val: bool = False,
        test: bool = False,
    ) -> str:
        if test:
            return "test"
        if val:
            return "val"
        name = str(split or "").strip().lower()
        if name in {"val", "validation"}:
            return "val"
        if name in {"test", "testing"}:
            return "test"
        # Fallback preference: val first, then test.
        if self.val_dataloader is not None:
            return "val"
        return "test"

    def _offload_model_to_cpu(self) -> None:
        """Move model/loss to CPU to avoid VRAM accumulation across many clients."""
        self.model = self.model.to("cpu")
        loss_to = getattr(self.loss_fn, "to", None)
        if callable(loss_to):
            self.loss_fn = loss_to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _apply_round_lr_decay(self, optimizer: torch.optim.Optimizer) -> None:
        enabled = bool(self.train_configs.get("lr_decay_enable", False))
        decay_type = str(self.train_configs.get("lr_decay_type", "none")).strip().lower()
        if (not enabled) or decay_type in {"", "none", "off", "false"}:
            self._apply_lr_floor(optimizer)
            return None

        base_lr = float(self.train_configs.get("lr", 0.01))
        round_idx = max(1, int(self.round))
        elapsed_rounds = max(0, round_idx - 1)

        if decay_type in {"exp", "exponential"}:
            gamma = float(self.train_configs.get("lr_decay_gamma", 0.99))
            lr_value = float(base_lr * (gamma**elapsed_rounds))
        elif decay_type in {"cos", "cosine"}:
            t_max = int(self.train_configs.get("lr_decay_t_max", 0))
            if t_max <= 0:
                t_max = int(self.train_configs.get("num_rounds", 1))
            t_max = max(1, t_max)
            eta_min = float(self.train_configs.get("lr_decay_eta_min", 0.0))
            progress = min(elapsed_rounds, t_max)
            cosine = 0.5 * (1.0 + math.cos(math.pi * float(progress) / float(t_max)))
            lr_value = float(eta_min + (base_lr - eta_min) * cosine)
        else:
            raise ValueError(
                f"Unsupported optimizer.lr_decay.type={decay_type}. "
                "Supported: none, exponential, cosine."
            )

        for param_group in optimizer.param_groups:
            param_group["lr"] = float(lr_value)
        self._apply_lr_floor(optimizer)

    def _apply_lr_floor(self, optimizer: torch.optim.Optimizer) -> None:
        min_lr = float(self.train_configs.get("lr_decay_min_lr", 0.0))
        if min_lr <= 0.0:
            return
        for param_group in optimizer.param_groups:
            current = float(param_group.get("lr", 0.0))
            if current < min_lr:
                param_group["lr"] = float(min_lr)

    def train(self, **kwargs):
        """
        Train the model for a certain number of local epochs or steps and store model state.
        """
        # Retrieve current round
        if "round" in kwargs:
            self.round = kwargs["round"]
        set_round_label = getattr(self.logger, "set_round_label", None)
        if callable(set_round_label):
            set_round_label(f"Round {int(self.round):04d}")
        self.eval_results = {"round": self.round + 1}

        # Check metrcis
        metric_names_for_log = []
        for metric_name in self.eval_metric_names:
            text = str(metric_name).strip().lower()
            if text and text not in metric_names_for_log:
                metric_names_for_log.append(text)
        if not metric_names_for_log:
            fallback = self._primary_metric_name()
            if fallback:
                metric_names_for_log.append(str(fallback).strip().lower())

        # Check dataset
        has_val_split = self.val_dataloader is not None
        has_test_split = self.test_dataloader is not None
        has_any_eval_split = has_val_split or has_test_split

        # Initiate model
        self.model = self.model.to(self.device)

        # Set up logging title
        title: List[str] = _build_trainer_log_title(
            mode=self.train_configs.mode,
            has_any_eval_split=has_any_eval_split,
            has_val_split=has_val_split,
            has_test_split=has_test_split,
            metric_names_for_log=metric_names_for_log,
        )
        self.logger.log_title(title)
        self.logger.set_title(title)

        # Evaluation scheme
        do_validation = bool(self.train_configs.get("do_post_evaluation", True)) and (
            has_any_eval_split
        )
        do_pre_evaluation = bool(self.train_configs.get("do_pre_evaluation", True)) and (
            has_any_eval_split
        )

        ## Pre-evaluation
        if do_pre_evaluation:
            pre_train_stats = self._evaluate_split_metrics(
                split="train", offload_after=False
            )
            self.eval_results["pre_train_loss"] = float(pre_train_stats["loss"])
            self.eval_results["pre_train_metric_value"] = float(
                self._metric_from_stats(pre_train_stats)
            )
            val_stats = (
                self._evaluate_split_metrics(split="val") if has_val_split else None
            )
            test_stats = (
                self._evaluate_split_metrics(split="test") if has_test_split else None
            )
            if val_stats is not None:
                self.eval_results["pre_val_loss"] = float(val_stats["loss"])
                self.eval_results["pre_val_metric_value"] = float(
                    self._metric_from_stats(val_stats)
                )
                self.eval_results["pre_val_metrics"] = dict(val_stats.get("metrics", {}))
            if test_stats is not None:
                self.eval_results["pre_test_loss"] = float(test_stats["loss"])
                self.eval_results["pre_test_metric_value"] = float(
                    self._metric_from_stats(test_stats)
                )
                self.eval_results["pre_test_metrics"] = dict(
                    test_stats.get("metrics", {})
                )

                self.logger.log_content(
                    _build_trainer_log_row(
                        mode=self.train_configs.mode,
                        has_any_eval_split=has_any_eval_split,
                        has_val_split=has_val_split,
                        has_test_split=has_test_split,
                        metric_names_for_log=metric_names_for_log,
                        epoch_idx=0 if self.train_configs.mode == "epoch" else None,
                        pre_eval_flag="Y",
                        elapsed="-",
                        train_stats_obj=None,
                        val_stats_obj=val_stats,
                        test_stats_obj=test_stats,
                    )
                )
                if self.enabled_wandb:
                    payload = {}
                    if val_stats is not None:
                        payload[f"{self.wandb_logging_id}/val-loss (before train)"] = float(
                            val_stats["loss"]
                        )
                        for key, value in val_stats.get("metrics", {}).items():
                            payload[
                                f"{self.wandb_logging_id}/val-{key} (before train)"
                            ] = float(value)
                    if test_stats is not None:
                        payload[
                            f"{self.wandb_logging_id}/test-loss (before train)"
                        ] = float(test_stats["loss"])
                        for key, value in test_stats.get("metrics", {}).items():
                            payload[
                                f"{self.wandb_logging_id}/test-{key} (before train)"
                            ] = float(value)
                    wandb.log(payload)

        # Define optimizer
        optimizer = build_optimizer_from_train_cfg(
            self.train_configs,
            self.model.parameters(),
        )
        self._apply_round_lr_decay(optimizer)

        # Identify local budget (epoch vs iteration)
        round_local_budget = kwargs.get("local_steps", None)
        if round_local_budget is not None:
            try:
                round_local_budget = max(1, int(round_local_budget))
            except Exception:
                round_local_budget = None

        # Start training
        total_examples = 0
        train_metrics_manager = self._new_metrics_manager()

        ## Identify update base
        is_epoch_mode = self.train_configs.mode == "epoch"
        effective_local_budget = (
            int(self.train_configs.num_local_epochs)
            if is_epoch_mode
            else int(self.train_configs.num_local_steps)
        )
        if round_local_budget is not None:
            effective_local_budget = int(round_local_budget)
        self.eval_results["current_local_steps"] = effective_local_budget

        total_units = effective_local_budget if is_epoch_mode else 1
        data_iter = iter(self.train_dataloader) if not is_epoch_mode else None
        for unit_idx in range(total_units):
            start_time = time.time()
            unit_examples = 0
            unit_metrics_manager = self._new_metrics_manager()

            if is_epoch_mode:
                for data, target in self.train_dataloader:
                    loss, pred, label = self._train_batch(optimizer, data, target)
                    batch_size = len(label)
                    unit_examples += batch_size
                    total_examples += batch_size
                    unit_metrics_manager.track(loss, pred, label)
                    train_metrics_manager.track(loss, pred, label)
            else:
                for _ in range(effective_local_budget):
                    try:
                        data, target = next(data_iter)
                    except StopIteration:
                        data_iter = iter(self.train_dataloader)
                        data, target = next(data_iter)
                    loss, pred, label = self._train_batch(optimizer, data, target)
                    batch_size = len(label)
                    unit_examples += batch_size
                    total_examples += batch_size
                    unit_metrics_manager.track(loss, pred, label)
                    train_metrics_manager.track(loss, pred, label)

            aggregate_kwargs = {"total_len": unit_examples}
            if is_epoch_mode:
                aggregate_kwargs["curr_step"] = unit_idx + 1
            train_stats = unit_metrics_manager.aggregate(**aggregate_kwargs)
            train_loss = float(train_stats["loss"])

            val_stats = None
            test_stats = None
            if do_validation:
                if has_val_split:
                    val_stats = self._evaluate_split_metrics(split="val")
                    if is_epoch_mode:
                        if "val_loss" not in self.eval_results:
                            self.eval_results["val_loss"] = []
                            self.eval_results["val_metric_value"] = []
                            self.eval_results["val_metrics"] = []
                        self.eval_results["val_loss"].append(float(val_stats["loss"]))
                        self.eval_results["val_metric_value"].append(
                            float(self._metric_from_stats(val_stats))
                        )
                        self.eval_results["val_metrics"].append(
                            dict(val_stats.get("metrics", {}))
                        )
                    else:
                        self.eval_results["val_loss"] = float(val_stats["loss"])
                        self.eval_results["val_metric_value"] = float(
                            self._metric_from_stats(val_stats)
                        )
                        self.eval_results["val_metrics"] = dict(val_stats.get("metrics", {}))
                if has_test_split:
                    test_stats = self._evaluate_split_metrics(split="test")
                    if is_epoch_mode:
                        if "test_loss" not in self.eval_results:
                            self.eval_results["test_loss"] = []
                            self.eval_results["test_metric_value"] = []
                            self.eval_results["test_metrics"] = []
                        self.eval_results["test_loss"].append(float(test_stats["loss"]))
                        self.eval_results["test_metric_value"].append(
                            float(self._metric_from_stats(test_stats))
                        )
                        self.eval_results["test_metrics"].append(
                            dict(test_stats.get("metrics", {}))
                        )
                    else:
                        self.eval_results["test_loss"] = float(test_stats["loss"])
                        self.eval_results["test_metric_value"] = float(
                            self._metric_from_stats(test_stats)
                        )
                        self.eval_results["test_metrics"] = dict(test_stats.get("metrics", {}))

            elapsed = time.time() - start_time
            if self.enabled_wandb:
                payload = {
                    f"{self.wandb_logging_id}/train-loss (during train)": train_loss,
                }
                for key, value in train_stats.get("metrics", {}).items():
                    payload[f"{self.wandb_logging_id}/train-{key} (during train)"] = float(
                        value
                    )
                if val_stats is not None:
                    payload[f"{self.wandb_logging_id}/val-loss (during train)"] = float(
                        val_stats["loss"]
                    )
                    for key, value in val_stats.get("metrics", {}).items():
                        payload[f"{self.wandb_logging_id}/val-{key} (during train)"] = float(
                            value
                        )
                if test_stats is not None:
                    payload[f"{self.wandb_logging_id}/test-loss (during train)"] = float(
                        test_stats["loss"]
                    )
                    for key, value in test_stats.get("metrics", {}).items():
                        payload[
                            f"{self.wandb_logging_id}/test-{key} (during train)"
                        ] = float(value)
                wandb.log(payload)
            self.logger.log_content(
                _build_trainer_log_row(
                    mode=self.train_configs.mode,
                    has_any_eval_split=has_any_eval_split,
                    has_val_split=has_val_split,
                    has_test_split=has_test_split,
                    metric_names_for_log=metric_names_for_log,
                    epoch_idx=unit_idx if is_epoch_mode else None,
                    pre_eval_flag="N",
                    elapsed=elapsed,
                    train_stats_obj=train_stats,
                    val_stats_obj=val_stats if do_validation else None,
                    test_stats_obj=test_stats if do_validation else None,
                )
            )

        self.round += 1
        if self.optimize_memory:
            self.model_state = extract_model_state_optimized(
                self.model, include_buffers=True, cpu_transfer=False
            )
        else:
            self.model_state = copy.deepcopy(self.model.state_dict())

        # Move model state to CPU for communication.
        if "cuda" in str(self.train_configs.get("device", "cpu")):
            if self.optimize_memory:
                for k in self.model_state:
                    if self.model_state[k].device.type != "cpu":
                        self.model_state[k] = self.model_state[k].cpu()
                gc.collect()
            else:
                for k in self.model_state:
                    self.model_state[k] = self.model_state[k].cpu()

        self._offload_model_to_cpu()

        result = train_metrics_manager.aggregate(total_len=total_examples)
        if "pre_val_loss" in self.eval_results:
            result["pre_val_loss"] = float(self.eval_results["pre_val_loss"])
            if "pre_val_metric_value" in self.eval_results:
                result["pre_val_metric_value"] = float(
                    self.eval_results["pre_val_metric_value"]
                )
            _attach_prefixed_metrics(
                result,
                self.eval_results.get("pre_val_metrics", {}),
                prefix="pre_val",
            )
        if "pre_train_loss" in self.eval_results:
            result["pre_train_loss"] = float(self.eval_results["pre_train_loss"])
            if "pre_train_metric_value" in self.eval_results:
                result["pre_train_metric_value"] = float(
                    self.eval_results["pre_train_metric_value"]
                )
        if "pre_test_loss" in self.eval_results:
            result["pre_test_loss"] = float(self.eval_results["pre_test_loss"])
            if "pre_test_metric_value" in self.eval_results:
                result["pre_test_metric_value"] = float(
                    self.eval_results["pre_test_metric_value"]
                )
            _attach_prefixed_metrics(
                result,
                self.eval_results.get("pre_test_metrics", {}),
                prefix="pre_test",
            )
        if "val_loss" in self.eval_results:
            if isinstance(self.eval_results["val_loss"], list):
                result["post_val_loss"] = float(self.eval_results["val_loss"][-1])
                val_metric_list = self.eval_results.get("val_metric_value", [])
                if isinstance(val_metric_list, list) and val_metric_list:
                    result["post_val_metric_value"] = float(val_metric_list[-1])
                val_metrics_list = self.eval_results.get("val_metrics", [])
                if isinstance(val_metrics_list, list) and val_metrics_list:
                    _attach_prefixed_metrics(
                        result,
                        val_metrics_list[-1],
                        prefix="post_val",
                    )
            else:
                result["post_val_loss"] = float(self.eval_results["val_loss"])
                if "val_metric_value" in self.eval_results:
                    result["post_val_metric_value"] = float(
                        self.eval_results["val_metric_value"]
                    )
                _attach_prefixed_metrics(
                    result,
                    self.eval_results.get("val_metrics", {}),
                    prefix="post_val",
                )
        if "test_loss" in self.eval_results:
            if isinstance(self.eval_results["test_loss"], list):
                result["post_test_loss"] = float(self.eval_results["test_loss"][-1])
                test_metric_list = self.eval_results.get("test_metric_value", [])
                if isinstance(test_metric_list, list) and test_metric_list:
                    result["post_test_metric_value"] = float(test_metric_list[-1])
                test_metrics_list = self.eval_results.get("test_metrics", [])
                if isinstance(test_metrics_list, list) and test_metrics_list:
                    _attach_prefixed_metrics(
                        result,
                        test_metrics_list[-1],
                        prefix="post_test",
                    )
            else:
                result["post_test_loss"] = float(self.eval_results["test_loss"])
                if "test_metric_value" in self.eval_results:
                    result["post_test_metric_value"] = float(
                        self.eval_results["test_metric_value"]
                    )
                _attach_prefixed_metrics(
                    result,
                    self.eval_results.get("test_metrics", {}),
                    prefix="post_test",
                )
        if "pre_val_loss" in result and "pre_train_loss" in result:
            result["local_gen_error"] = float(
                result["pre_val_loss"] - result["pre_train_loss"]
            )
        return result

    def get_parameters(self) -> Dict:
        if self.model_state is None:
            if self.optimize_memory:
                self.model_state = extract_model_state_optimized(
                    self.model, include_buffers=True, cpu_transfer=True
                )
            else:
                self.model_state = {
                    k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()
                }
        return (
            (self.model_state, self.eval_results)
            if self.eval_results is not None
            else self.model_state
        )

    @torch.no_grad()
    def evaluate(
        self,
        split: str = "test",
        val: bool = False,
        test: bool = False,
        offload_after: Optional[bool] = None,
    ) -> Dict[str, Any]:
        chosen = self._resolve_eval_split(split=split, val=val, test=test)
        if chosen == "val" and self.val_dataloader is None:
            return {"loss": -1.0, "num_examples": 0, "metrics": {}}
        if chosen == "test" and self.test_dataloader is None:
            if self.val_dataloader is None:
                return {"loss": -1.0, "num_examples": 0, "metrics": {}}
            chosen = "val"
        if chosen == "val" and self.val_dataloader is None:
            return {"loss": -1.0, "num_examples": 0, "metrics": {}}
        offload_flag = True if offload_after is None else bool(offload_after)
        return self._validate_metrics(
            split=chosen,
            offload_after=offload_flag,
        )

    def _validate(self) -> Tuple[float, float]:
        """
        Validate the model
        :return: loss, primary metric value
        """
        stats = self._validate_metrics(split="val", offload_after=False)
        return float(stats["loss"]), float(self._metric_from_stats(stats))

    def _evaluate_train_metrics(self, offload_after: bool = False) -> Dict[str, Any]:
        return self._evaluate_metrics_on_loader(
            self.train_dataloader, offload_after=offload_after
        )

    def _evaluate_split_metrics(
        self, split: str = "val", offload_after: bool = False
    ) -> Dict[str, Any]:
        split_name = str(split).strip().lower()
        if split_name in {"train", "training"}:
            return self._evaluate_metrics_on_loader(
                self.train_dataloader, offload_after=offload_after
            )
        return self._validate_metrics(split=split_name, offload_after=offload_after)

    def _validate_metrics(self, split: str = "val", offload_after: bool = False) -> Dict[str, Any]:
        chosen = self._resolve_eval_split(split=split)
        dataloader = self.val_dataloader if chosen == "val" else self.test_dataloader
        if dataloader is None:
            fallback = self.test_dataloader if chosen == "val" else self.val_dataloader
            dataloader = fallback
        return self._evaluate_metrics_on_loader(
            dataloader, offload_after=offload_after
        )

    def _evaluate_metrics_on_loader(
        self,
        dataloader: Optional[DataLoader],
        offload_after: bool = False,
    ) -> Dict[str, Any]:
        if dataloader is None:
            return {"loss": -1.0, "num_examples": 0, "metrics": {}}
        device = self.device
        was_training = self.model.training
        self.model = self.model.to(device)
        loss_to = getattr(self.loss_fn, "to", None)
        if callable(loss_to):
            self.loss_fn = loss_to(device)
        self.model.eval()
        manager = self._new_metrics_manager()
        total_examples = 0
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(device), target.to(device)
                output = self.model(data)
                loss = self.loss_fn(output, target)
                pred_cpu = output.detach().cpu()
                true_cpu = target.detach().cpu()
                manager.track(float(loss.item()), pred_cpu, true_cpu)
                total_examples += int(true_cpu.shape[0]) if true_cpu.ndim > 0 else 1
        stats = manager.aggregate(total_len=total_examples)
        if offload_after:
            self._offload_model_to_cpu()
        if was_training:
            self.model.train()
        return stats

    def _train_batch(
        self, optimizer: torch.optim.Optimizer, data, target
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Train the model for one batch of data
        :param optimizer: torch optimizer
        :param data: input data
        :param target: target label
        :return: loss, prediction, label
        """
        device = self.device
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = self.model(data)
        loss = self.loss_fn(output, target)
        loss.backward()
        if bool(self.train_configs.get("clip_grad", False)):
            if "clip_value" not in self.train_configs:
                raise ValueError("Gradient clipping value must be specified.")
            if "clip_norm" not in self.train_configs:
                raise ValueError("Gradient clipping norm must be specified.")
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.train_configs.clip_value,
                norm_type=self.train_configs.clip_norm,
            )
        optimizer.step()
        return loss.item(), output.detach().cpu().numpy(), target.detach().cpu().numpy()
