from collections import OrderedDict
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader, Dataset

from appfl_sim.algorithm.trainer.base_trainer import BaseTrainer
from appfl_sim.metrics.metricszoo import accuracy_from_logits
from appfl_sim.misc.utils import clone_state_dict


class VanillaTrainer(BaseTrainer):
    """Simple local trainer for simulation clients."""

    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn: Optional[torch.nn.Module],
        metric: Optional[Any],
        train_dataset: Dataset,
        val_dataset: Optional[Dataset],
        train_configs: Optional[Dict] = None,
        client_id: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            model=model,
            loss_fn=loss_fn,
            metric=metric,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            train_configs=train_configs,
            client_id=client_id,
            **kwargs,
        )
        cfg = self.train_configs
        self.device = cfg.get("device", "cpu")
        self.batch_size = int(cfg.get("batch_size", 32))
        self.num_workers = int(cfg.get("num_workers", 0))
        self.local_epochs = int(cfg.get("local_epochs", 1))
        self.max_grad_norm = float(cfg.get("max_grad_norm", 0.0))
        self.optim_name = cfg.get("optimizer", "SGD")
        self.optim_args = cfg.get("optim_args", {"lr": 0.01})

        if self.loss_fn is None:
            self.loss_fn = torch.nn.CrossEntropyLoss()

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        self.val_loader = (
            DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )
            if self.val_dataset is not None and len(self.val_dataset) > 0
            else None
        )

    def _build_optimizer(self) -> torch.optim.Optimizer:
        if self.optim_name not in torch.optim.__dict__:
            raise ValueError(f"Unsupported optimizer: {self.optim_name}")
        optim_cls = torch.optim.__dict__[self.optim_name]
        return optim_cls(self.model.parameters(), **self.optim_args)

    def get_parameters(self) -> OrderedDict:
        state = self.model.state_dict()
        return clone_state_dict(OrderedDict((k, v.cpu()) for k, v in state.items()))

    def train(self, **kwargs) -> Dict[str, float]:
        if "round" in kwargs:
            self.round = int(kwargs["round"])

        self.model.to(self.device)
        self.model.train()
        optimizer = self._build_optimizer()
        printed_shape = False

        total_loss = 0.0
        total_correct = 0
        total_examples = 0

        for _ in range(self.local_epochs):
            for inputs, targets in self.train_loader:
                if not printed_shape:
                    msg = f"[client {self.client_id}] local batch x.shape={tuple(inputs.shape)}"
                    if self.logger is not None:
                        self.logger.info(msg)
                    else:
                        print(msg)
                    printed_shape = True
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                logits = self.model(inputs)
                loss = self.loss_fn(logits, targets)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if self.max_grad_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                optimizer.step()

                batch_size = targets.size(0)
                total_examples += batch_size
                total_loss += loss.item() * batch_size
                total_correct += accuracy_from_logits(logits.detach(), targets.detach(), as_count=True)

        self.model.to("cpu")
        avg_loss = total_loss / max(total_examples, 1)
        avg_acc = float(total_correct / max(total_examples, 1))
        return {
            "loss": float(avg_loss),
            "accuracy": float(avg_acc),
            "num_examples": int(total_examples),
        }

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        if self.val_loader is None:
            return {"loss": -1.0, "accuracy": -1.0, "num_examples": 0}

        self.model.to(self.device)
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_examples = 0

        for inputs, targets in self.val_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            logits = self.model(inputs)
            loss = self.loss_fn(logits, targets)
            batch_size = targets.size(0)
            total_examples += batch_size
            total_loss += loss.item() * batch_size
            total_correct += accuracy_from_logits(logits, targets, as_count=True)

        self.model.to("cpu")
        return {
            "loss": float(total_loss / max(total_examples, 1)),
            "accuracy": float(total_correct / max(total_examples, 1)),
            "num_examples": int(total_examples),
        }
