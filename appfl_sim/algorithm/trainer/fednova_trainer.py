from __future__ import annotations

from typing import Any, Optional

from omegaconf import DictConfig
from torch.nn import Module
from torch.utils.data import Dataset

from appfl_sim.algorithm.trainer.fedavg_trainer import FedavgTrainer


class FednovaTrainer(FedavgTrainer):
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

    def _train_batch(self, optimizer, data, target):
        result = super()._train_batch(optimizer, data, target)
        self._completed_optimizer_steps += 1
        return result

    def train(self, *args, **kwargs):
        self._completed_optimizer_steps = 0
        result = super().train(*args, **kwargs)
        result["completed_local_steps"] = int(self._completed_optimizer_steps)
        result["fednova_tau"] = int(self._completed_optimizer_steps)
        return result

