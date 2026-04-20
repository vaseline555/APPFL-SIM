from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from omegaconf import DictConfig
from torch.nn import Module
from torch.utils.data import Dataset

from appfl_sim.algorithm.trainer.fedavg_trainer import FedavgTrainer


class ScaffoldTrainer(FedavgTrainer):
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
        self._completed_optimizer_steps = 0
        self._global_reference_state: Optional[Dict[str, torch.Tensor]] = None
        self._server_control_cpu: Dict[str, torch.Tensor] = {}
        self._client_control_cpu: Dict[str, torch.Tensor] = {}
        self._server_control_device: Optional[Dict[str, torch.Tensor]] = None
        self._client_control_device: Optional[Dict[str, torch.Tensor]] = None
        self._control_device_name: Optional[str] = None

    def load_parameters(self, params):
        context = {}
        model_state = params
        if isinstance(params, tuple):
            model_state = params[0]
            if len(params) > 1 and isinstance(params[1], dict):
                context = params[1]
        super().load_parameters(model_state)
        self._global_reference_state = self._parameter_state(model_state)
        self._server_control_cpu = self._sanitize_control_state(
            context.get("scaffold_server_control", {})
        )
        self._client_control_cpu = self._sanitize_control_state(
            context.get("scaffold_client_control", {})
        )
        self._server_control_device = None
        self._client_control_device = None
        self._control_device_name = None

    def _sanitize_control_state(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        sanitized: Dict[str, torch.Tensor] = {}
        for name, ref_tensor in (self._global_reference_state or {}).items():
            if isinstance(state, dict) and name in state and torch.is_tensor(state[name]):
                sanitized[name] = state[name].detach().cpu().clone()
            else:
                sanitized[name] = torch.zeros_like(ref_tensor, device="cpu")
        return sanitized

    def _control_state_for_device(
        self,
        source: Dict[str, torch.Tensor],
        *,
        cache_name: str,
        device: str,
    ) -> Dict[str, torch.Tensor]:
        device_name = str(device)
        if self._control_device_name != device_name:
            self._server_control_device = None
            self._client_control_device = None
            self._control_device_name = device_name
        cache = self._server_control_device if cache_name == "server" else self._client_control_device
        if cache is None:
            cache = {
                name: tensor.to(device_name)
                for name, tensor in source.items()
            }
            if cache_name == "server":
                self._server_control_device = cache
            else:
                self._client_control_device = cache
        return cache

    def _train_batch(self, optimizer: torch.optim.Optimizer, data, target):
        device = self.device
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = self.model(data)
        loss = self.loss_fn(output, target)
        loss.backward()

        server_control = self._control_state_for_device(
            self._server_control_cpu,
            cache_name="server",
            device=device,
        )
        client_control = self._control_state_for_device(
            self._client_control_cpu,
            cache_name="client",
            device=device,
        )
        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
            server_term = server_control.get(name, None)
            client_term = client_control.get(name, None)
            if server_term is not None:
                param.grad.add_(server_term)
            if client_term is not None:
                param.grad.sub_(client_term)

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
        self._completed_optimizer_steps += 1
        return loss.item(), output.detach().cpu().numpy(), target.detach().cpu().numpy()

    def _resolve_control_update_lr(self, result: Dict[str, Any]) -> float:
        value = result.get("current_lr", None)
        if isinstance(value, (int, float)) and float(value) > 0.0:
            return float(value)
        optimizer_cfg = self.train_configs.get("optimizer", {})
        value = optimizer_cfg.get("lr", None)
        if isinstance(value, (int, float)) and float(value) > 0.0:
            return float(value)
        return 0.0

    def _compute_updated_client_control(self, result: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        local_steps = int(self._completed_optimizer_steps)
        learning_rate = self._resolve_control_update_lr(result)
        if local_steps <= 0 or learning_rate <= 0.0:
            return {
                name: tensor.detach().cpu().clone()
                for name, tensor in self._client_control_cpu.items()
            }

        denom = float(local_steps) * float(learning_rate)
        current_state = self.model.state_dict()
        updated: Dict[str, torch.Tensor] = {}
        for name, global_tensor in (self._global_reference_state or {}).items():
            local_tensor = current_state.get(name, None)
            if local_tensor is None:
                continue
            global_cpu = global_tensor.detach().cpu()
            client_control = self._client_control_cpu.get(name, torch.zeros_like(global_cpu))
            server_control = self._server_control_cpu.get(name, torch.zeros_like(global_cpu))
            updated[name] = client_control - server_control + (
                (global_cpu - local_tensor.detach().cpu()) / denom
            )
        return updated

    @staticmethod
    def _control_l2_norm(control_state: Dict[str, torch.Tensor]) -> float:
        total = 0.0
        for tensor in control_state.values():
            total += float(torch.sum(tensor.float() * tensor.float()).item())
        return float(total ** 0.5)

    def train(self, *args, **kwargs):
        self._completed_optimizer_steps = 0
        result = super().train(*args, **kwargs)
        updated_control = self._compute_updated_client_control(result)
        self._client_control_cpu = {
            name: tensor.detach().cpu().clone()
            for name, tensor in updated_control.items()
        }
        result["completed_local_steps"] = int(self._completed_optimizer_steps)
        result["scaffold_client_control"] = self._client_control_cpu
        result["scaffold_control_norm"] = self._control_l2_norm(self._client_control_cpu)
        return result
