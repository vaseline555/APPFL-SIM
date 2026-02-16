import copy
import time
import torch
import importlib
import numpy as np
from torch.nn import Module
from omegaconf import DictConfig
from typing import Tuple, Dict, Optional, Any
from torch.utils.data import Dataset, DataLoader
from appfl_sim.privacy import (
    SecureAggregator,
    laplace_mechanism_output_perturb,
    gaussian_mechanism_output_perturb,
    make_private_with_opacus,
)
from appfl_sim.algorithm.trainer.base_trainer import BaseTrainer
from appfl_sim.misc.utils import parse_device_str, apply_model_device
from appfl_sim.misc.memory_utils import (
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
        # Backward compatibility with sim-style train config keys.
        if not hasattr(train_configs, "mode"):
            train_configs.mode = "epoch"
        if (
            train_configs.mode == "epoch"
            and not hasattr(train_configs, "num_local_epochs")
            and hasattr(train_configs, "local_epochs")
        ):
            train_configs.num_local_epochs = int(train_configs.local_epochs)
        if not hasattr(train_configs, "optim"):
            train_configs.optim = str(
                train_configs.get("optimizer", train_configs.get("optim", "SGD"))
            )
        if not hasattr(train_configs, "optim_args"):
            train_configs.optim_args = {
                "lr": float(train_configs.get("lr", 0.01)),
                "weight_decay": float(train_configs.get("weight_decay", 0.0)),
            }
        if not hasattr(train_configs, "train_batch_size"):
            train_configs.train_batch_size = int(
                train_configs.get("batch_size", train_configs.get("train_batch_size", 32))
            )
        if not hasattr(train_configs, "val_batch_size"):
            train_configs.val_batch_size = int(
                train_configs.get("val_batch_size", train_configs.get("batch_size", 32))
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

        self.privacy_engine = None
        if not hasattr(self.train_configs, "device"):
            self.train_configs.device = "cpu"
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.train_configs.get("train_batch_size", 32),
            shuffle=self.train_configs.get("train_data_shuffle", True),
            num_workers=self.train_configs.get("num_workers", 0),
        )
        self.val_dataloader = (
            DataLoader(
                self.val_dataset,
                batch_size=self.train_configs.get("val_batch_size", 32),
                shuffle=self.train_configs.get("val_data_shuffle", False),
                num_workers=self.train_configs.get("num_workers", 0),
            )
            if self.val_dataset is not None
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

    def train(self, **kwargs):
        """
        Train the model for a certain number of local epochs or steps and store the mode state
        (probably with perturbation for differential privacy) in `self.model_state`.
        """
        if "round" in kwargs:
            self.round = kwargs["round"]
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

        do_validation = (
            self.train_configs.get("do_validation", False)
            and self.val_dataloader is not None
        )
        do_pre_validation = (
            self.train_configs.get("do_pre_validation", False)
            and self.val_dataloader is not None
        )

        # Set up logging title
        title = (
            ["Time", "Train Loss", "Train Accuracy"]
            if (not do_validation) and (not do_pre_validation)
            else (
                [
                    "Pre Val?",
                    "Time",
                    "Train Loss",
                    "Train Accuracy",
                    "Val Loss",
                    "Val Accuracy",
                ]
                if do_pre_validation
                else [
                    "Time",
                    "Train Loss",
                    "Train Accuracy",
                    "Val Loss",
                    "Val Accuracy",
                ]
            )
        )
        if self.train_configs.mode == "epoch":
            title.insert(0, "Epoch")

        if self.train_configs.get("client_log_title_each_round", True):
            self.logger.log_title(title)
        elif not hasattr(self.logger, "titles"):
            self.logger.log_title(title)
        self.logger.set_title(title)

        if do_pre_validation:
            if (
                self.train_configs.get("use_secure_agg", False) and False
            ):  # TODO: Check why skip evaluation in secure aggregation mode
                # Skip evaluation in secure aggregation mode
                val_loss, val_acc = None, None  # noqa F841
                self.logger.info(
                    f"Round {self.round} Pre-Validation skipped (secure aggregation enabled)"
                )
            else:
                val_loss, val_accuracy = self._validate()
                self.val_results["pre_val_loss"] = val_loss
                self.val_results["pre_val_accuracy"] = val_accuracy
                content = ["Y", "-", "-", "-", val_loss, val_accuracy]
                if self.train_configs.mode == "epoch":
                    content.insert(0, 0)
                self.logger.log_content(content)
                if self.enabled_wandb:
                    wandb.log(
                        {
                            f"{self.wandb_logging_id}/val-loss (before train)": val_loss,
                            f"{self.wandb_logging_id}/val-accuracy (before train)": val_accuracy,
                        }
                    )

        # Start training
        optim_module = importlib.import_module("torch.optim")
        optim_name = self.train_configs.get("optim", self.train_configs.get("optimizer", "SGD"))
        assert hasattr(optim_module, optim_name), (
            f"Optimizer {optim_name} not found in torch.optim"
        )
        optimizer = getattr(optim_module, optim_name)(
            self.model.parameters(),
            **self.train_configs.get(
                "optim_args",
                {
                    "lr": float(self.train_configs.get("lr", 0.01)),
                    "weight_decay": float(self.train_configs.get("weight_decay", 0.0)),
                },
            ),
        )
        total_examples = 0
        total_loss_sum = 0.0
        total_correct = 0

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
            for epoch in range(self.train_configs.num_local_epochs):
                start_time = time.time()
                train_loss, target_true, target_pred = 0, [], []
                for data, target in self.train_dataloader:
                    loss, pred, label = self._train_batch(optimizer, data, target)
                    train_loss += loss
                    target_true.append(label)
                    target_pred.append(pred)
                    batch_size = len(label)
                    total_examples += batch_size
                    total_loss_sum += float(loss) * batch_size
                    if pred.ndim > 1:
                        total_correct += int(
                            np.sum(np.argmax(pred, axis=1) == label.reshape(-1))
                        )
                train_loss /= len(self.train_dataloader)
                target_true, target_pred = (
                    np.concatenate(target_true),
                    np.concatenate(target_pred),
                )
                train_accuracy = float(self.metric(target_true, target_pred))
                if do_validation:
                    val_loss, val_accuracy = self._validate()
                    if "val_loss" not in self.val_results:
                        self.val_results["val_loss"] = []
                        self.val_results["val_accuracy"] = []
                    self.val_results["val_loss"].append(val_loss)
                    self.val_results["val_accuracy"].append(val_accuracy)
                per_epoch_time = time.time() - start_time
                if self.enabled_wandb:
                    wandb.log(
                        {
                            f"{self.wandb_logging_id}/train-loss (during train)": train_loss,
                            f"{self.wandb_logging_id}/train-accuracy (during train)": train_accuracy,
                            f"{self.wandb_logging_id}/val-loss (during train)": val_loss,
                            f"{self.wandb_logging_id}/val-accuracy (during train)": val_accuracy,
                        }
                    )
                self.logger.log_content(
                    [epoch, per_epoch_time, train_loss, train_accuracy]
                    if (not do_validation) and (not do_pre_validation)
                    else (
                        [
                            epoch,
                            per_epoch_time,
                            train_loss,
                            train_accuracy,
                            val_loss,
                            val_accuracy,
                        ]
                        if not do_pre_validation
                        else [
                            epoch,
                            "N",
                            per_epoch_time,
                            train_loss,
                            train_accuracy,
                            val_loss,
                            val_accuracy,
                        ]
                    )
                )
        else:
            self.val_results["current_local_steps"] = self.train_configs.num_local_steps
            start_time = time.time()
            train_loss, target_true, target_pred = 0, [], []
            if (
                self.train_configs.get("use_dp", False)
                and self.train_configs.get("dp_mechanism", "laplace") == "opacus"
            ):
                with BatchMemoryManager(
                    data_loader=self.train_dataloader,
                    max_physical_batch_size=self.train_configs.get(
                        "train_batch_size", 32
                    ),
                    optimizer=optimizer,
                ) as memory_safe_data_loader:
                    step_count = 0
                    for data, target in memory_safe_data_loader:
                        loss, pred, label = self._train_batch(optimizer, data, target)
                        train_loss += loss
                        target_true.append(label)
                        target_pred.append(pred)
                        batch_size = len(label)
                        total_examples += batch_size
                        total_loss_sum += float(loss) * batch_size
                        if pred.ndim > 1:
                            total_correct += int(
                                np.sum(np.argmax(pred, axis=1) == label.reshape(-1))
                            )
                        step_count += 1
                        if step_count >= self.train_configs.num_local_steps:
                            break
            else:
                data_iter = iter(self.train_dataloader)
                for _ in range(self.train_configs.num_local_steps):
                    try:
                        data, target = next(data_iter)
                    except:  # noqa E722
                        data_iter = iter(self.train_dataloader)
                        data, target = next(data_iter)
                    loss, pred, label = self._train_batch(optimizer, data, target)
                    train_loss += loss
                    target_true.append(label)
                    target_pred.append(pred)
                    batch_size = len(label)
                    total_examples += batch_size
                    total_loss_sum += float(loss) * batch_size
                    if pred.ndim > 1:
                        total_correct += int(
                            np.sum(np.argmax(pred, axis=1) == label.reshape(-1))
                        )
            train_loss /= len(self.train_dataloader)
            target_true, target_pred = (
                np.concatenate(target_true),
                np.concatenate(target_pred),
            )
            train_accuracy = float(self.metric(target_true, target_pred))
            if do_validation:
                val_loss, val_accuracy = self._validate()
                self.val_results["val_loss"] = val_loss
                self.val_results["val_accuracy"] = val_accuracy
            per_step_time = time.time() - start_time
            if self.enabled_wandb:
                wandb.log(
                    {
                        f"{self.wandb_logging_id}/train-loss (during train)": train_loss,
                        f"{self.wandb_logging_id}/train-accuracy (during train)": train_accuracy,
                        f"{self.wandb_logging_id}/val-loss (during train)": val_loss,
                        f"{self.wandb_logging_id}/val-accuracy (during train)": val_accuracy,
                    }
                )
            self.logger.log_content(
                [per_step_time, train_loss, train_accuracy]
                if (not do_validation) and (not do_pre_validation)
                else (
                    [
                        per_step_time,
                        train_loss,
                        train_accuracy,
                        val_loss,
                        val_accuracy,
                    ]
                    if not do_pre_validation
                    else [
                        "N",
                        per_step_time,
                        train_loss,
                        train_accuracy,
                        val_loss,
                        val_accuracy,
                    ]
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
                2.0 * self.train_configs.clip_value * self.train_configs.optim_args.lr
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
                2.0 * self.train_configs.clip_value * self.train_configs.optim_args.lr
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

            # optional weighting (e.g., sample_size). If you want weighted aggregation,
            # pre-scale delta_state here by client weight w_i.
            secure_agg_client_weights_mode = self.train_configs.get(
                "secure_agg_client_weights_mode", "equal"
            )  # "equal" or "num_examples"
            if secure_agg_client_weights_mode == "num_examples":
                # requires runtime_context["global_num_examples_sum"]
                if "global_num_examples_sum" not in self.runtime_context:
                    raise RuntimeError(
                        "global_num_examples_sum required in runtime_context for num_examples weighting"
                    )
                local_n = float(
                    self.runtime_context.get(
                        "local_num_examples", len(self.train_dataset)
                    )
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
                    self.runtime_context.get(
                        "local_num_examples", len(self.train_dataset)
                    )
                ),
            }
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

        result = {
            "loss": float(total_loss_sum / max(total_examples, 1)),
            "accuracy": float(total_correct / max(total_examples, 1)),
            "num_examples": int(total_examples),
        }
        if "pre_val_loss" in self.val_results and "pre_val_accuracy" in self.val_results:
            result["pre_val_loss"] = float(self.val_results["pre_val_loss"])
            result["pre_val_accuracy"] = float(self.val_results["pre_val_accuracy"])
        if "val_loss" in self.val_results and "val_accuracy" in self.val_results:
            if isinstance(self.val_results["val_loss"], list):
                result["post_val_loss"] = float(self.val_results["val_loss"][-1])
                result["post_val_accuracy"] = float(self.val_results["val_accuracy"][-1])
            else:
                result["post_val_loss"] = float(self.val_results["val_loss"])
                result["post_val_accuracy"] = float(self.val_results["val_accuracy"])
        return result

    def get_parameters(self) -> Dict:
        if not hasattr(self, "model_state"):
            if self.optimize_memory:
                self.model_state = extract_model_state_optimized(
                    self.model, include_buffers=True, cpu_transfer=False
                )
            else:
                self.model_state = copy.deepcopy(self.model.state_dict())
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
    def evaluate(self) -> Dict[str, float]:
        if self.val_dataloader is None:
            return {"loss": -1.0, "accuracy": -1.0, "num_examples": 0}
        loss, accuracy = self._validate()
        num_examples = len(self.val_dataset) if self.val_dataset is not None else 0
        return {
            "loss": float(loss),
            "accuracy": float(accuracy),
            "num_examples": int(num_examples),
        }

    def _validate(self) -> Tuple[float, float]:
        """
        Validate the model
        :return: loss, accuracy
        """
        device = self.device
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            target_pred, target_true = [], []
            for data, target in self.val_dataloader:
                data, target = data.to(device), target.to(device)
                output = self.model(data)
                val_loss += self.loss_fn(output, target).item()
                target_true.append(target.detach().cpu().numpy())
                target_pred.append(output.detach().cpu().numpy())
        val_loss /= len(self.val_dataloader)
        val_accuracy = float(
            self.metric(np.concatenate(target_true), np.concatenate(target_pred))
        )
        self.model.train()
        return val_loss, val_accuracy

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
