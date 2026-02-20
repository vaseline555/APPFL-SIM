import copy
import time
import torch
import importlib
import numpy as np
from torch.nn import Module
from omegaconf import DictConfig
from typing import Tuple, Dict, Optional, Any, List
from torch.utils.data import Dataset, DataLoader
from appfl_sim.privacy import (
    SecureAggregator,
    laplace_mechanism_output_perturb,
    gaussian_mechanism_output_perturb,
    make_private_with_opacus,
)
from appfl_sim.algorithm.trainer.base_trainer import BaseTrainer
from appfl_sim.metrics import MetricsManager, parse_metric_names
from appfl_sim.misc.system_utils import parse_device_str, apply_model_device
from appfl_sim.misc.system_utils import (
    extract_model_state_optimized,
    safe_inplace_operation,
    optimize_memory_cleanup,
)
import logging

try:
    import wandb
except Exception:  # pragma: no cover
    class _WandbStub:
        run = None

        @staticmethod
        def log(*args, **kwargs):
            return None

    wandb = _WandbStub()

try:
    from opacus import PrivacyEngine
    from opacus.utils.batch_memory_manager import BatchMemoryManager
except Exception:  # pragma: no cover
    PrivacyEngine = None
    BatchMemoryManager = None

logging.getLogger().handlers.clear()
logging.getLogger().setLevel(logging.WARNING)


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


def _default_classification_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fallback accuracy metric when no metric is configured."""
    if y_true.size == 0 or y_pred.size == 0:
        return 0.0
    if y_pred.ndim <= 1:
        return 0.0
    pred_labels = np.argmax(y_pred, axis=1)
    true_labels = y_true.reshape(-1)
    return float(np.mean(pred_labels == true_labels))


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
        train_configs: DictConfig = DictConfig({}),
        logger: Optional[Any] = None,
        **kwargs,
    ):
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
        if not hasattr(train_configs, "mode"):
            train_configs.mode = "epoch"
        if not hasattr(train_configs, "optim"):
            train_configs.optim = str(
                train_configs.get("optimizer", train_configs.get("optim", "SGD"))
            )
        if not hasattr(train_configs, "lr"):
            train_configs.lr = float(train_configs.get("lr", 0.01))
        if not hasattr(train_configs, "weight_decay"):
            train_configs.weight_decay = float(train_configs.get("weight_decay", 0.0))
        if not hasattr(train_configs, "batch_size"):
            train_configs.batch_size = int(train_configs.get("batch_size", 32))
        if not hasattr(train_configs, "eval_batch_size"):
            train_configs.eval_batch_size = int(
                train_configs.get("eval_batch_size", train_configs.get("batch_size", 32))
            )
        if (
            float(train_configs.get("max_grad_norm", 0.0)) > 0.0
            and not train_configs.get("clip_grad", False)
        ):
            train_configs.clip_grad = True
            train_configs.clip_value = float(train_configs.max_grad_norm)
            train_configs.clip_norm = float(train_configs.get("clip_norm", 2.0))

        # Check for optimize_memory in train_configs, default to True
        self.optimize_memory = getattr(train_configs, "optimize_memory", True)
        if self.metric is None:
            self.metric = _default_classification_accuracy
        self.eval_metric_names = parse_metric_names(
            self.train_configs.get("eval_metrics", None)
        )

        self.privacy_engine = None
        if not hasattr(self.train_configs, "device"):
            self.train_configs.device = "cpu"
        num_workers = int(self.train_configs.get("num_workers", 0))
        train_pin_memory = bool(self.train_configs.get("train_pin_memory", False))
        eval_pin_memory = bool(
            self.train_configs.get("eval_pin_memory", train_pin_memory)
        )
        persistent_workers = bool(
            self.train_configs.get("dataloader_persistent_workers", False)
        )
        prefetch_factor = int(self.train_configs.get("dataloader_prefetch_factor", 2))

        self.train_dataloader = _make_dataloader(
            self.train_dataset,
            batch_size=self.train_configs.get("batch_size", 32),
            shuffle=self.train_configs.get("train_data_shuffle", True),
            num_workers=num_workers,
            pin_memory=train_pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        )
        self.val_dataloader = (
            _make_dataloader(
                self.val_dataset,
                batch_size=self.train_configs.get("eval_batch_size", 32),
                shuffle=False,
                num_workers=num_workers,
                pin_memory=eval_pin_memory,
                persistent_workers=persistent_workers,
                prefetch_factor=prefetch_factor,
            )
            if self.val_dataset is not None
            else None
        )
        self.test_dataset = getattr(self, "test_dataset", None)
        self.test_dataloader = (
            _make_dataloader(
                self.test_dataset,
                batch_size=self.train_configs.get("eval_batch_size", 32),
                shuffle=False,
                num_workers=num_workers,
                pin_memory=eval_pin_memory,
                persistent_workers=persistent_workers,
                prefetch_factor=prefetch_factor,
            )
            if self.test_dataset is not None
            else None
        )
        if (
            hasattr(self.train_configs, "enable_wandb")
            and self.train_configs.enable_wandb
        ):
            self.enabled_wandb = True
            self.wandb_logging_id = self.train_configs.wandb_logging_id
        else:
            self.enabled_wandb = False
        self._sanity_check()

        # Extract train device, and configurations for possible DataParallel
        self.device_config, self.device = parse_device_str(self.train_configs.device)

        # Differential privacy through Opacus
        if self.train_configs.get("use_dp", False) and (
            self.train_configs.get("dp_mechanism", "laplace") == "opacus"
        ):
            if PrivacyEngine is None:
                raise ImportError(
                    "Opacus is required for dp_mechanism='opacus'. Install opacus or disable this mode."
                )
            self.privacy_engine = PrivacyEngine()

    def _new_metrics_manager(self) -> MetricsManager:
        return MetricsManager(eval_metrics=self.eval_metric_names)

    def _primary_metric_name(self) -> str:
        if self.eval_metric_names:
            return str(self.eval_metric_names[0]).strip().lower()
        return "accuracy"

    def _primary_metric_display_name(self) -> str:
        return self._primary_metric_name()

    def _metric_from_stats(self, stats: Optional[Dict[str, Any]]) -> float:
        if not isinstance(stats, dict):
            return -1.0
        metric_name = self._primary_metric_name()
        nested = stats.get("metrics", {})
        if isinstance(nested, dict):
            value = nested.get(metric_name, None)
            if isinstance(value, (int, float)):
                return float(value)
        for key in (f"metric_{metric_name}", metric_name, "accuracy"):
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

    def _fill_accuracy_from_fallback(
        self,
        stats: Dict[str, Any],
        target_true: List[np.ndarray],
        target_pred: List[np.ndarray],
    ) -> None:
        if float(stats.get("accuracy", -1.0)) >= 0.0:
            return
        if self.metric is None or not target_true or not target_pred:
            return
        try:
            y_true = np.concatenate(target_true)
            y_pred = np.concatenate(target_pred)
            stats["accuracy"] = float(self.metric(y_true, y_pred))
        except Exception:
            pass

    @staticmethod
    def _attach_prefixed_metrics(
        output: Dict[str, Any],
        metrics: Optional[Dict[str, Any]],
        prefix: str,
    ) -> None:
        if not isinstance(metrics, dict) or not metrics:
            return
        output[f"{prefix}_metrics"] = {k: float(v) for k, v in metrics.items()}
        for key, value in metrics.items():
            output[f"{prefix}_metric_{key}"] = float(value)

    def _should_offload_after_local_job(self) -> bool:
        """Whether to move model/loss back to CPU after local train/eval."""
        return True

    def _offload_model_to_cpu(self) -> None:
        """Move model/loss to CPU to avoid VRAM accumulation across many clients."""
        try:
            self.model = self.model.to("cpu")
        except Exception:
            pass
        try:
            if hasattr(self.loss_fn, "to"):
                self.loss_fn = self.loss_fn.to("cpu")
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def train(self, **kwargs):
        """
        Train the model for a certain number of local epochs or steps and store the mode state
        (probably with perturbation for differential privacy) in `self.model_state`.
        """
        if "round" in kwargs:
            self.round = kwargs["round"]
        override_local_steps = kwargs.get("local_steps", None)
        if override_local_steps is not None:
            try:
                override_local_steps = max(1, int(override_local_steps))
            except Exception:
                override_local_steps = None
        if hasattr(self.logger, "set_round_label"):
            self.logger.set_round_label(f"Round {int(self.round):04d}")
        self.val_results = {"round": self.round + 1}

        # Store the previous model state for gradient computation
        send_gradient = self.train_configs.get("send_gradient", False)
        use_secure_agg = self.train_configs.get("use_secure_agg", False)
        if send_gradient or use_secure_agg:
            if self.optimize_memory:
                self.model_prev = extract_model_state_optimized(
                    self.model, include_buffers=True, cpu_transfer=False
                )
            else:
                self.model_prev = copy.deepcopy(self.model.state_dict())

        # Configure model for possible DataParallel
        self.model = apply_model_device(self.model, self.device_config, self.device)

        has_val_split = self.val_dataloader is not None
        has_test_split = self.test_dataloader is not None
        has_any_eval_split = has_val_split or has_test_split
        do_validation = bool(self.train_configs.get("do_validation", True)) and (
            has_any_eval_split
        )
        do_pre_validation = bool(self.train_configs.get("do_pre_validation", True)) and (
            has_any_eval_split
        )
        metric_names_for_log = []
        for metric_name in self.eval_metric_names:
            text = str(metric_name).strip().lower()
            if text and text not in metric_names_for_log:
                metric_names_for_log.append(text)
        if not metric_names_for_log:
            fallback = self._primary_metric_name()
            if fallback:
                metric_names_for_log.append(str(fallback).strip().lower())

        def _metric_title(metric_name: str) -> str:
            text = str(metric_name).strip()
            if text == "":
                return "Metric"
            return text[:1].upper() + text[1:]

        def _metric_value(stats_obj: Optional[Dict[str, Any]], metric_name: str) -> float:
            if not isinstance(stats_obj, dict):
                return -1.0
            nested = stats_obj.get("metrics", {})
            if isinstance(nested, dict):
                value = nested.get(metric_name, None)
                if isinstance(value, (int, float)):
                    return float(value)
            for key in (f"metric_{metric_name}", metric_name):
                value = stats_obj.get(key, None)
                if isinstance(value, (int, float)):
                    return float(value)
            if metric_name in {"acc", "acc1", "accuracy"}:
                value = stats_obj.get("accuracy", None)
                if isinstance(value, (int, float)):
                    return float(value)
            return -1.0

        def _build_log_row(
            epoch_idx: Optional[int],
            pre_eval_flag: str,
            elapsed,
            train_stats_obj: Optional[Dict[str, Any]],
            val_stats_obj: Optional[Dict[str, Any]],
            test_stats_obj: Optional[Dict[str, Any]],
        ) -> List[Any]:
            row: List[Any] = []
            if self.train_configs.mode == "epoch":
                row.append(epoch_idx if epoch_idx is not None else "-")
            if has_any_eval_split:
                row.append(pre_eval_flag)
            row.append(elapsed)
            row.append(
                float(train_stats_obj["loss"])
                if isinstance(train_stats_obj, dict) and "loss" in train_stats_obj
                else "-"
            )
            for metric_name in metric_names_for_log:
                row.append(_metric_value(train_stats_obj, metric_name))
            if has_val_split:
                row.append(
                    float(val_stats_obj["loss"])
                    if isinstance(val_stats_obj, dict) and "loss" in val_stats_obj
                    else -1.0
                )
                for metric_name in metric_names_for_log:
                    row.append(_metric_value(val_stats_obj, metric_name))
            if has_test_split:
                row.append(
                    float(test_stats_obj["loss"])
                    if isinstance(test_stats_obj, dict) and "loss" in test_stats_obj
                    else -1.0
                )
                for metric_name in metric_names_for_log:
                    row.append(_metric_value(test_stats_obj, metric_name))
            return row

        # Set up logging title
        title: List[str] = []
        if self.train_configs.mode == "epoch":
            title.append("Epoch")
        if has_any_eval_split:
            title.append("Pre Eval?")
        title.extend(["Time", "Train. Loss"])
        for metric_name in metric_names_for_log:
            title.append(f"Train. {_metric_title(metric_name)}")
        if has_val_split:
            title.append("Val. Loss")
            for metric_name in metric_names_for_log:
                title.append(f"Val. {_metric_title(metric_name)}")
        if has_test_split:
            title.append("Test Loss")
            for metric_name in metric_names_for_log:
                title.append(f"Test {_metric_title(metric_name)}")

        self.logger.log_title(title)
        self.logger.set_title(title)

        if do_pre_validation:
            pre_train_stats = self._evaluate_train_metrics(offload_after=False)
            self.val_results["pre_train_loss"] = float(pre_train_stats["loss"])
            self.val_results["pre_train_accuracy"] = float(pre_train_stats["accuracy"])
            val_stats = self._validate_metrics(split="val") if has_val_split else None
            test_stats = (
                self._validate_metrics(split="test") if has_test_split else None
            )
            if val_stats is not None:
                self.val_results["pre_val_loss"] = float(val_stats["loss"])
                self.val_results["pre_val_accuracy"] = float(val_stats["accuracy"])
                self.val_results["pre_val_metric_value"] = float(
                    self._metric_from_stats(val_stats)
                )
                self.val_results["pre_val_metrics"] = dict(val_stats.get("metrics", {}))
            if test_stats is not None:
                self.val_results["pre_test_loss"] = float(test_stats["loss"])
                self.val_results["pre_test_accuracy"] = float(test_stats["accuracy"])
                self.val_results["pre_test_metric_value"] = float(
                    self._metric_from_stats(test_stats)
                )
                self.val_results["pre_test_metrics"] = dict(
                    test_stats.get("metrics", {})
                )

                self.logger.log_content(
                    _build_log_row(
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
                        payload[
                            f"{self.wandb_logging_id}/val-accuracy (before train)"
                        ] = float(val_stats["accuracy"])
                        for key, value in val_stats.get("metrics", {}).items():
                            payload[
                                f"{self.wandb_logging_id}/val-{key} (before train)"
                            ] = float(value)
                    if test_stats is not None:
                        payload[
                            f"{self.wandb_logging_id}/test-loss (before train)"
                        ] = float(test_stats["loss"])
                        payload[
                            f"{self.wandb_logging_id}/test-accuracy (before train)"
                        ] = float(test_stats["accuracy"])
                        for key, value in test_stats.get("metrics", {}).items():
                            payload[
                                f"{self.wandb_logging_id}/test-{key} (before train)"
                            ] = float(value)
                    wandb.log(payload)

        # Start training
        optim_module = importlib.import_module("torch.optim")
        optim_name = self.train_configs.get("optim", self.train_configs.get("optimizer", "SGD"))
        assert hasattr(optim_module, optim_name), (
            f"Optimizer {optim_name} not found in torch.optim"
        )
        optimizer = getattr(optim_module, optim_name)(
            self.model.parameters(),
            lr=float(self.train_configs.get("lr", 0.01)),
            weight_decay=float(self.train_configs.get("weight_decay", 0.0)),
        )
        total_examples = 0
        total_correct = 0
        total_has_logits = False
        train_metrics_manager = self._new_metrics_manager()

        if self.train_configs.get("use_dp", False) and (
            self.train_configs.get("dp_mechanism", "laplace") == "opacus"
        ):
            dp_cfg = self.train_configs.get("dp_config", {})
            noise_multiplier = dp_cfg.get("noise_multiplier", 1.0)
            max_grad_norm = dp_cfg.get("max_grad_norm", 1.0)

            self.model, optimizer, self.train_dataloader = make_private_with_opacus(
                self.privacy_engine,
                self.model,
                optimizer,
                self.train_dataloader,
                noise_multiplier=noise_multiplier,
                max_grad_norm=max_grad_norm,
                device=self.train_configs.device,
            )

        if self.train_configs.mode == "epoch":
            effective_local_epochs = int(self.train_configs.num_local_epochs)
            if override_local_steps is not None:
                effective_local_epochs = int(override_local_steps)
            self.val_results["current_local_steps"] = effective_local_epochs
            for epoch in range(effective_local_epochs):
                start_time = time.time()
                target_true, target_pred = [], []
                epoch_examples = 0
                epoch_correct = 0
                epoch_has_logits = False
                epoch_metrics_manager = self._new_metrics_manager()
                for data, target in self.train_dataloader:
                    loss, pred, label = self._train_batch(optimizer, data, target)
                    target_true.append(label)
                    target_pred.append(pred)
                    batch_size = len(label)
                    epoch_examples += batch_size
                    total_examples += batch_size
                    epoch_metrics_manager.track(loss, pred, label)
                    train_metrics_manager.track(loss, pred, label)
                    if pred.ndim > 1:
                        epoch_has_logits = True
                        total_has_logits = True
                        correct = int(np.sum(np.argmax(pred, axis=1) == label.reshape(-1)))
                        epoch_correct += correct
                        total_correct += correct
                train_stats = epoch_metrics_manager.aggregate(
                    total_len=epoch_examples,
                    curr_step=epoch + 1,
                )
                if float(train_stats.get("accuracy", -1.0)) < 0.0 and epoch_has_logits:
                    train_stats["accuracy"] = float(epoch_correct / max(epoch_examples, 1))
                self._fill_accuracy_from_fallback(train_stats, target_true, target_pred)
                train_loss = float(train_stats["loss"])
                train_accuracy = float(train_stats["accuracy"])
                val_stats = None
                test_stats = None
                if do_validation:
                    if has_val_split:
                        val_stats = self._validate_metrics(split="val")
                        if "val_loss" not in self.val_results:
                            self.val_results["val_loss"] = []
                            self.val_results["val_accuracy"] = []
                            self.val_results["val_metric_value"] = []
                            self.val_results["val_metrics"] = []
                        self.val_results["val_loss"].append(float(val_stats["loss"]))
                        self.val_results["val_accuracy"].append(
                            float(val_stats["accuracy"])
                        )
                        self.val_results["val_metric_value"].append(
                            float(self._metric_from_stats(val_stats))
                        )
                        self.val_results["val_metrics"].append(
                            dict(val_stats.get("metrics", {}))
                        )
                    if has_test_split:
                        test_stats = self._validate_metrics(split="test")
                        if "test_loss" not in self.val_results:
                            self.val_results["test_loss"] = []
                            self.val_results["test_accuracy"] = []
                            self.val_results["test_metric_value"] = []
                            self.val_results["test_metrics"] = []
                        self.val_results["test_loss"].append(float(test_stats["loss"]))
                        self.val_results["test_accuracy"].append(
                            float(test_stats["accuracy"])
                        )
                        self.val_results["test_metric_value"].append(
                            float(self._metric_from_stats(test_stats))
                        )
                        self.val_results["test_metrics"].append(
                            dict(test_stats.get("metrics", {}))
                        )
                per_epoch_time = time.time() - start_time
                if self.enabled_wandb:
                    payload = {
                        f"{self.wandb_logging_id}/train-loss (during train)": train_loss,
                        f"{self.wandb_logging_id}/train-accuracy (during train)": train_accuracy,
                    }
                    for key, value in train_stats.get("metrics", {}).items():
                        payload[
                            f"{self.wandb_logging_id}/train-{key} (during train)"
                        ] = float(value)
                    if val_stats is not None:
                        payload[
                            f"{self.wandb_logging_id}/val-loss (during train)"
                        ] = float(val_stats["loss"])
                        payload[
                            f"{self.wandb_logging_id}/val-accuracy (during train)"
                        ] = float(val_stats["accuracy"])
                        for key, value in val_stats.get("metrics", {}).items():
                            payload[
                                f"{self.wandb_logging_id}/val-{key} (during train)"
                            ] = float(value)
                    if test_stats is not None:
                        payload[
                            f"{self.wandb_logging_id}/test-loss (during train)"
                        ] = float(test_stats["loss"])
                        payload[
                            f"{self.wandb_logging_id}/test-accuracy (during train)"
                        ] = float(test_stats["accuracy"])
                        for key, value in test_stats.get("metrics", {}).items():
                            payload[
                                f"{self.wandb_logging_id}/test-{key} (during train)"
                            ] = float(value)
                    wandb.log(payload)
                self.logger.log_content(
                    _build_log_row(
                        epoch_idx=epoch,
                        pre_eval_flag="N",
                        elapsed=per_epoch_time,
                        train_stats_obj=train_stats,
                        val_stats_obj=val_stats if do_validation else None,
                        test_stats_obj=test_stats if do_validation else None,
                    )
                )
        else:
            effective_local_steps = int(self.train_configs.num_local_steps)
            if override_local_steps is not None:
                effective_local_steps = int(override_local_steps)
            self.val_results["current_local_steps"] = effective_local_steps
            start_time = time.time()
            target_true, target_pred = [], []
            step_examples = 0
            step_correct = 0
            step_has_logits = False
            step_metrics_manager = self._new_metrics_manager()
            if (
                self.train_configs.get("use_dp", False)
                and self.train_configs.get("dp_mechanism", "laplace") == "opacus"
            ):
                with BatchMemoryManager(
                    data_loader=self.train_dataloader,
                    max_physical_batch_size=self.train_configs.get(
                        "batch_size", 32
                    ),
                    optimizer=optimizer,
                ) as memory_safe_data_loader:
                    step_count = 0
                    for data, target in memory_safe_data_loader:
                        loss, pred, label = self._train_batch(optimizer, data, target)
                        target_true.append(label)
                        target_pred.append(pred)
                        batch_size = len(label)
                        step_examples += batch_size
                        total_examples += batch_size
                        step_metrics_manager.track(loss, pred, label)
                        train_metrics_manager.track(loss, pred, label)
                        if pred.ndim > 1:
                            step_has_logits = True
                            total_has_logits = True
                            correct = int(np.sum(np.argmax(pred, axis=1) == label.reshape(-1)))
                            step_correct += correct
                            total_correct += correct
                        step_count += 1
                        if step_count >= effective_local_steps:
                            break
            else:
                data_iter = iter(self.train_dataloader)
                for _ in range(effective_local_steps):
                    try:
                        data, target = next(data_iter)
                    except:  # noqa E722
                        data_iter = iter(self.train_dataloader)
                        data, target = next(data_iter)
                    loss, pred, label = self._train_batch(optimizer, data, target)
                    target_true.append(label)
                    target_pred.append(pred)
                    batch_size = len(label)
                    step_examples += batch_size
                    total_examples += batch_size
                    step_metrics_manager.track(loss, pred, label)
                    train_metrics_manager.track(loss, pred, label)
                    if pred.ndim > 1:
                        step_has_logits = True
                        total_has_logits = True
                        correct = int(np.sum(np.argmax(pred, axis=1) == label.reshape(-1)))
                        step_correct += correct
                        total_correct += correct
            train_stats = step_metrics_manager.aggregate(total_len=step_examples)
            if float(train_stats.get("accuracy", -1.0)) < 0.0 and step_has_logits:
                train_stats["accuracy"] = float(step_correct / max(step_examples, 1))
            self._fill_accuracy_from_fallback(train_stats, target_true, target_pred)
            train_loss = float(train_stats["loss"])
            train_accuracy = float(train_stats["accuracy"])
            val_stats = None
            test_stats = None
            if do_validation:
                if has_val_split:
                    val_stats = self._validate_metrics(split="val")
                    self.val_results["val_loss"] = float(val_stats["loss"])
                    self.val_results["val_accuracy"] = float(val_stats["accuracy"])
                    self.val_results["val_metric_value"] = float(
                        self._metric_from_stats(val_stats)
                    )
                    self.val_results["val_metrics"] = dict(val_stats.get("metrics", {}))
                if has_test_split:
                    test_stats = self._validate_metrics(split="test")
                    self.val_results["test_loss"] = float(test_stats["loss"])
                    self.val_results["test_accuracy"] = float(test_stats["accuracy"])
                    self.val_results["test_metric_value"] = float(
                        self._metric_from_stats(test_stats)
                    )
                    self.val_results["test_metrics"] = dict(test_stats.get("metrics", {}))
            per_step_time = time.time() - start_time
            if self.enabled_wandb:
                payload = {
                    f"{self.wandb_logging_id}/train-loss (during train)": train_loss,
                    f"{self.wandb_logging_id}/train-accuracy (during train)": train_accuracy,
                }
                for key, value in train_stats.get("metrics", {}).items():
                    payload[
                        f"{self.wandb_logging_id}/train-{key} (during train)"
                    ] = float(value)
                if val_stats is not None:
                    payload[f"{self.wandb_logging_id}/val-loss (during train)"] = float(
                        val_stats["loss"]
                    )
                    payload[
                        f"{self.wandb_logging_id}/val-accuracy (during train)"
                    ] = float(val_stats["accuracy"])
                    for key, value in val_stats.get("metrics", {}).items():
                        payload[
                            f"{self.wandb_logging_id}/val-{key} (during train)"
                        ] = float(value)
                if test_stats is not None:
                    payload[f"{self.wandb_logging_id}/test-loss (during train)"] = float(
                        test_stats["loss"]
                    )
                    payload[
                        f"{self.wandb_logging_id}/test-accuracy (during train)"
                    ] = float(test_stats["accuracy"])
                    for key, value in test_stats.get("metrics", {}).items():
                        payload[
                            f"{self.wandb_logging_id}/test-{key} (during train)"
                        ] = float(value)
                wandb.log(payload)
            self.logger.log_content(
                _build_log_row(
                    epoch_idx=None,
                    pre_eval_flag="N",
                    elapsed=per_step_time,
                    train_stats_obj=train_stats,
                    val_stats_obj=val_stats if do_validation else None,
                    test_stats_obj=test_stats if do_validation else None,
                )
            )

        # --- Log DP budget ---
        if (
            self.train_configs.get("use_dp", False)
            and self.train_configs.get("dp_mechanism", "laplace") == "opacus"
        ):
            epsilon = self.privacy_engine.get_epsilon(delta=1e-5)
            self.logger.info(
                f"[DP] Training completed with (ε = {epsilon:.2f}, δ = 1e-5)"
            )

        # If model was wrapped in DataParallel, unload it
        if self.device_config["device_type"] == "gpu-multi":
            self.model = self.model.module.to(self.device)

        self.round += 1

        # Differential privacy
        if self.train_configs.get("use_dp", False) and (
            self.train_configs.get("dp_mechanism", "laplace") == "gaussian"
        ):
            assert hasattr(self.train_configs, "clip_value"), (
                "Using laplace differential privacy, and gradient clipping value must be specified"
            )
            assert hasattr(self.train_configs, "epsilon"), (
                "Using laplace differential privacy, and privacy budget (epsilon) must be specified"
            )
            sensitivity = (
                2.0 * self.train_configs.clip_value * float(self.train_configs.get("lr", 0.01))
            )
            self.model_state = gaussian_mechanism_output_perturb(
                self.model,
                sensitivity,
                self.train_configs.epsilon,
            )
        elif self.train_configs.get("use_dp", False) and (
            self.train_configs.get("dp_mechanism", "laplace") == "laplace"
        ):
            assert hasattr(self.train_configs, "clip_value"), (
                "Using laplace differential privacy, and gradient clipping value must be specified"
            )
            assert hasattr(self.train_configs, "epsilon"), (
                "Using laplace differential privacy, and privacy budget (epsilon) must be specified"
            )
            sensitivity = (
                2.0 * self.train_configs.clip_value * float(self.train_configs.get("lr", 0.01))
            )
            self.model_state = laplace_mechanism_output_perturb(
                self.model,
                sensitivity,
                self.train_configs.epsilon,
            )
        else:
            if self.optimize_memory:
                self.model_state = extract_model_state_optimized(
                    self.model, include_buffers=True, cpu_transfer=False
                )
            else:
                self.model_state = copy.deepcopy(self.model.state_dict())

        if self.train_configs.get("use_secure_agg", False):
            local_state = {
                k: v.detach().clone() for k, v in self.model.state_dict().items()
            }

            # compute delta = local - global
            delta_state = {}
            for k in local_state:
                # ensure devices match
                g = self.model_prev[k].to(local_state[k].device)
                delta_state[k] = (local_state[k] - g).detach().clone()

            # optional weighting (e.g., sample_ratio). If you want weighted aggregation,
            # pre-scale delta_state here by client weight w_i.
            secure_agg_client_weights_mode = str(
                self.train_configs.get("secure_agg_client_weights_mode", "uniform")
            ).strip().lower()
            if secure_agg_client_weights_mode not in {"uniform", "sample_ratio"}:
                secure_agg_client_weights_mode = "uniform"
            if secure_agg_client_weights_mode == "sample_ratio":
                # requires runtime_context["global_num_examples_sum"]
                if "global_num_examples_sum" not in self.runtime_context:
                    raise RuntimeError(
                        "global_num_examples_sum required in runtime_context for sample_ratio weighting"
                    )
                local_n = float(
                    len(self.train_dataset)
                )
                total_n = float(self.runtime_context["global_num_examples_sum"])
                w = local_n / total_n
                for k in delta_state:
                    delta_state[k] = delta_state[k] * w

            # prepare secure aggregator and mask

            client_id = self.client_id
            all_client_ids = self.runtime_context["all_client_ids"]
            round_id = int(self.runtime_context["round_id"])
            secret = self.runtime_context["secure_agg_secret"]  # bytes
            device = next(self.model.parameters()).device

            sa = SecureAggregator(
                client_id=client_id,
                all_client_ids=all_client_ids,
                secret=secret,
                device=device,
            )
            masked_flat, shapes = sa.mask_update(delta_state, round_id)

            # model_state now becomes a masked payload
            self.model_state = {
                "type": "masked_update_flat",
                "flat": masked_flat.cpu(),  # send CPU tensors to aggregator/orchestrator
                "shapes": shapes,
                "num_examples": int(
                    len(self.train_dataset)
                ),
            }
            if hasattr(self, "model_prev"):
                del self.model_prev
            del local_state, delta_state
            optimize_memory_cleanup(force_gc=True)
        else:
            # Move to CPU for communication
            if "cuda" in self.train_configs.device:
                if self.optimize_memory:
                    for k in self.model_state:
                        if self.model_state[k].device.type != "cpu":
                            self.model_state[k] = self.model_state[k].cpu()
                    optimize_memory_cleanup(force_gc=True)
                else:
                    for k in self.model_state:
                        self.model_state[k] = self.model_state[k].cpu()

            # Compute the gradient if needed
            if send_gradient:
                self._compute_gradient()
                if hasattr(self, "model_prev"):
                    del self.model_prev
                optimize_memory_cleanup(force_gc=True)

        if self._should_offload_after_local_job():
            self._offload_model_to_cpu()

        result = train_metrics_manager.aggregate(total_len=total_examples)
        if float(result.get("accuracy", -1.0)) < 0.0 and total_has_logits:
            result["accuracy"] = float(total_correct / max(total_examples, 1))
        if "pre_val_loss" in self.val_results and "pre_val_accuracy" in self.val_results:
            result["pre_val_loss"] = float(self.val_results["pre_val_loss"])
            result["pre_val_accuracy"] = float(self.val_results["pre_val_accuracy"])
            self._attach_prefixed_metrics(
                result,
                self.val_results.get("pre_val_metrics", {}),
                prefix="pre_val",
            )
        if (
            "pre_train_loss" in self.val_results
            and "pre_train_accuracy" in self.val_results
        ):
            result["pre_train_loss"] = float(self.val_results["pre_train_loss"])
            result["pre_train_accuracy"] = float(self.val_results["pre_train_accuracy"])
        if "pre_test_loss" in self.val_results and "pre_test_accuracy" in self.val_results:
            result["pre_test_loss"] = float(self.val_results["pre_test_loss"])
            result["pre_test_accuracy"] = float(self.val_results["pre_test_accuracy"])
            self._attach_prefixed_metrics(
                result,
                self.val_results.get("pre_test_metrics", {}),
                prefix="pre_test",
            )
        if "val_loss" in self.val_results and "val_accuracy" in self.val_results:
            if isinstance(self.val_results["val_loss"], list):
                result["post_val_loss"] = float(self.val_results["val_loss"][-1])
                result["post_val_accuracy"] = float(self.val_results["val_accuracy"][-1])
                val_metrics_list = self.val_results.get("val_metrics", [])
                if isinstance(val_metrics_list, list) and val_metrics_list:
                    self._attach_prefixed_metrics(
                        result,
                        val_metrics_list[-1],
                        prefix="post_val",
                    )
            else:
                result["post_val_loss"] = float(self.val_results["val_loss"])
                result["post_val_accuracy"] = float(self.val_results["val_accuracy"])
                self._attach_prefixed_metrics(
                    result,
                    self.val_results.get("val_metrics", {}),
                    prefix="post_val",
                )
        if "test_loss" in self.val_results and "test_accuracy" in self.val_results:
            if isinstance(self.val_results["test_loss"], list):
                result["post_test_loss"] = float(self.val_results["test_loss"][-1])
                result["post_test_accuracy"] = float(self.val_results["test_accuracy"][-1])
                test_metrics_list = self.val_results.get("test_metrics", [])
                if isinstance(test_metrics_list, list) and test_metrics_list:
                    self._attach_prefixed_metrics(
                        result,
                        test_metrics_list[-1],
                        prefix="post_test",
                    )
            else:
                result["post_test_loss"] = float(self.val_results["test_loss"])
                result["post_test_accuracy"] = float(self.val_results["test_accuracy"])
                self._attach_prefixed_metrics(
                    result,
                    self.val_results.get("test_metrics", {}),
                    prefix="post_test",
                )
        if "pre_val_loss" in result and "pre_train_loss" in result:
            result["local_gen_error"] = float(
                result["pre_val_loss"] - result["pre_train_loss"]
            )
        return result

    def get_parameters(self) -> Dict:
        if not hasattr(self, "model_state"):
            if self.optimize_memory:
                self.model_state = extract_model_state_optimized(
                    self.model, include_buffers=True, cpu_transfer=True
                )
            else:
                self.model_state = {
                    k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()
                }
        return (
            (self.model_state, self.val_results)
            if hasattr(self, "val_results")
            else self.model_state
        )

    def _sanity_check(self):
        """
        Check if the configurations are valid.
        """
        assert hasattr(self.train_configs, "mode"), "Training mode must be specified"
        assert self.train_configs.mode in [
            "epoch",
            "step",
        ], "Training mode must be either 'epoch' or 'step'"
        if self.train_configs.mode == "epoch":
            assert hasattr(self.train_configs, "num_local_epochs"), (
                "Number of local epochs must be specified"
            )
        else:
            assert hasattr(self.train_configs, "num_local_steps"), (
                "Number of local steps must be specified"
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
            return {"loss": -1.0, "accuracy": -1.0, "num_examples": 0, "metrics": {}}
        if chosen == "test" and self.test_dataloader is None:
            if self.val_dataloader is None:
                return {"loss": -1.0, "accuracy": -1.0, "num_examples": 0, "metrics": {}}
            chosen = "val"
        if chosen == "val" and self.val_dataloader is None:
            return {"loss": -1.0, "accuracy": -1.0, "num_examples": 0, "metrics": {}}
        offload_flag = (
            self._should_offload_after_local_job()
            if offload_after is None
            else bool(offload_after)
        )
        return self._validate_metrics(
            split=chosen,
            offload_after=offload_flag,
        )

    def _validate(self) -> Tuple[float, float]:
        """
        Validate the model
        :return: loss, accuracy
        """
        stats = self._validate_metrics(split="val", offload_after=False)
        return float(stats["loss"]), float(stats["accuracy"])

    def _evaluate_train_metrics(self, offload_after: bool = False) -> Dict[str, Any]:
        return self._evaluate_metrics_on_loader(
            self.train_dataloader, offload_after=offload_after
        )

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
            return {"loss": -1.0, "accuracy": -1.0, "num_examples": 0, "metrics": {}}
        device = self.device
        was_training = self.model.training
        self.model = apply_model_device(self.model, self.device_config, device)
        if hasattr(self.loss_fn, "to"):
            self.loss_fn = self.loss_fn.to(device)
        self.model.eval()
        manager = self._new_metrics_manager()
        total_examples = 0
        target_pred, target_true = [], []
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(device), target.to(device)
                output = self.model(data)
                loss = self.loss_fn(output, target)
                pred_cpu = output.detach().cpu()
                true_cpu = target.detach().cpu()
                manager.track(float(loss.item()), pred_cpu, true_cpu)
                total_examples += int(true_cpu.shape[0]) if true_cpu.ndim > 0 else 1
                target_true.append(true_cpu.numpy())
                target_pred.append(pred_cpu.numpy())
        stats = manager.aggregate(total_len=total_examples)
        self._fill_accuracy_from_fallback(stats, target_true, target_pred)
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
        if getattr(self.train_configs, "clip_grad", False):
            assert hasattr(self.train_configs, "clip_value"), (
                "Gradient clipping value must be specified"
            )
            assert hasattr(self.train_configs, "clip_norm"), (
                "Gradient clipping norm must be specified"
            )
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.train_configs.clip_value,
                norm_type=self.train_configs.clip_norm,
            )
        optimizer.step()
        return loss.item(), output.detach().cpu().numpy(), target.detach().cpu().numpy()

    def _compute_gradient(self) -> None:
        """
        Compute the gradient of the model and store in `self.model_state`,
        where gradient = prev_model - new_model
        """
        if not hasattr(self, "named_parameters"):
            self.named_parameters = set()
            for name, _ in self.model.named_parameters():
                self.named_parameters.add(name)

        if self.optimize_memory:
            with torch.no_grad():
                for name in self.model_state:
                    if name in self.named_parameters:
                        prev_param = (
                            self.model_prev[name].cpu()
                            if self.model_prev[name].device.type != "cpu"
                            else self.model_prev[name]
                        )
                        self.model_state[name] = safe_inplace_operation(
                            prev_param, "sub", self.model_state[name], alpha=1
                        )
            optimize_memory_cleanup(self.model_prev, force_gc=True)
            del self.model_prev
        else:
            for name in self.model_state:
                if name in self.named_parameters:
                    self.model_state[name] = (
                        self.model_prev[name].cpu() - self.model_state[name]
                    )
