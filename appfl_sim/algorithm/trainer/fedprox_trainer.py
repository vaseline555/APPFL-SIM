from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from omegaconf import DictConfig
from torch.nn import Module
from torch.utils.data import Dataset

from appfl_sim.algorithm.trainer.fedavg_trainer import FedavgTrainer


class FedproxTrainer(FedavgTrainer):
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
        self.mu = float(self.train_configs.get("mu", 0.001))
        self._global_reference_cpu: Dict[str, torch.Tensor] = {}
        self._global_reference_device: Optional[Dict[str, torch.Tensor]] = None
        self._global_reference_device_name: Optional[str] = None

    def load_parameters(self, params):
        model_state = params[0] if isinstance(params, tuple) else params
        super().load_parameters(model_state)
        reference: Dict[str, torch.Tensor] = {}
        for name, _ in self.model.named_parameters():
            tensor = model_state.get(name, None)
            if tensor is None:
                continue
            reference[name] = tensor.detach().cpu().clone()
        self._global_reference_cpu = reference
        self._global_reference_device = None
        self._global_reference_device_name = None

    def _reference_state_for_device(self, device: str) -> Dict[str, torch.Tensor]:
        device_name = str(device)
        if (
            self._global_reference_device is None
            or self._global_reference_device_name != device_name
        ):
            self._global_reference_device = {
                name: tensor.to(device_name)
                for name, tensor in self._global_reference_cpu.items()
            }
            self._global_reference_device_name = device_name
        return self._global_reference_device

    def _train_batch(
        self, optimizer: torch.optim.Optimizer, data, target
    ):
        if self.mu <= 0.0 or not self._global_reference_cpu:
            return super()._train_batch(optimizer, data, target)

        device = self.device
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = self.model(data)
        loss = self.loss_fn(output, target)

        reference_state = self._reference_state_for_device(device)
        prox_term = None
        for name, param in self.model.named_parameters():
            ref_param = reference_state.get(name, None)
            if ref_param is None:
                continue
            diff = param - ref_param
            term = torch.sum(diff * diff)
            prox_term = term if prox_term is None else prox_term + term
        if prox_term is not None:
            loss = loss + 0.5 * float(self.mu) * prox_term

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

