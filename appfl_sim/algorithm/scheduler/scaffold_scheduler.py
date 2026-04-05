from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from omegaconf import DictConfig

from appfl_sim.algorithm.aggregator import BaseAggregator
from appfl_sim.algorithm.scheduler.fedavg_scheduler import FedavgScheduler


class ScaffoldScheduler(FedavgScheduler):
    def __init__(
        self,
        scheduler_configs: DictConfig,
        aggregator: BaseAggregator,
        logger: Any,
    ):
        super().__init__(scheduler_configs=scheduler_configs, aggregator=aggregator, logger=logger)
        self._control_template: Optional[Dict[str, torch.Tensor]] = None
        self._server_control_variate: Optional[Dict[str, torch.Tensor]] = None
        self._client_control_variates: Dict[int, Dict[str, torch.Tensor]] = {}

    def _build_control_template(self) -> Dict[str, torch.Tensor]:
        if self._control_template is not None:
            return self._control_template

        template: Dict[str, torch.Tensor] = {}
        model = getattr(self.aggregator, "model", None)
        if model is not None:
            for name, param in model.named_parameters():
                template[name] = torch.zeros_like(param.detach(), device="cpu")
        else:
            state = self.aggregator.get_parameters()
            if isinstance(state, tuple):
                state = state[0]
            for name, tensor in state.items():
                if torch.is_tensor(tensor) and tensor.is_floating_point():
                    template[name] = torch.zeros_like(tensor.detach(), device="cpu")
        self._control_template = template
        return template

    def _clone_control_state(
        self,
        source: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        template = self._build_control_template()
        if source is None:
            return {name: tensor.clone() for name, tensor in template.items()}
        cloned: Dict[str, torch.Tensor] = {}
        for name, template_tensor in template.items():
            if name in source and torch.is_tensor(source[name]):
                cloned[name] = source[name].detach().cpu().clone()
            else:
                cloned[name] = template_tensor.clone()
        return cloned

    def _ensure_control_state_initialized(self) -> None:
        if self._server_control_variate is None:
            self._server_control_variate = self._clone_control_state()

    def _ensure_client_control(self, client_id: int) -> Dict[str, torch.Tensor]:
        self._ensure_control_state_initialized()
        if client_id not in self._client_control_variates:
            self._client_control_variates[client_id] = self._clone_control_state()
        return self._client_control_variates[client_id]

    def _recompute_server_control(self) -> None:
        self._ensure_control_state_initialized()
        server_control = self._clone_control_state()
        num_clients = max(1, int(self.num_clients))
        for client_control in self._client_control_variates.values():
            for name in server_control:
                server_control[name].add_(client_control[name], alpha=1.0 / float(num_clients))
        self._server_control_variate = server_control

    def get_client_contexts(
        self,
        *,
        selected_ids=None,
        round_idx: int | None = None,
    ) -> Dict[int, Dict[str, Any]] | None:
        del round_idx
        if not selected_ids:
            return None
        self._ensure_control_state_initialized()
        contexts: Dict[int, Dict[str, Any]] = {}
        for raw_client_id in selected_ids:
            client_id = int(raw_client_id)
            contexts[client_id] = {
                "scaffold_server_control": self._clone_control_state(
                    self._server_control_variate
                ),
                "scaffold_client_control": self._clone_control_state(
                    self._ensure_client_control(client_id)
                ),
            }
        return contexts

    def update_round_feedback(
        self,
        *,
        round_metrics: Dict[str, Any],
        client_train_stats=None,
        sample_sizes=None,
    ):
        feedback = super().update_round_feedback(
            round_metrics=round_metrics,
            client_train_stats=client_train_stats,
            sample_sizes=sample_sizes,
        )
        if not isinstance(client_train_stats, dict):
            return feedback

        updated = False
        for raw_client_id, stats in client_train_stats.items():
            if not isinstance(stats, dict):
                continue
            control_state = stats.get("scaffold_client_control", None)
            if not isinstance(control_state, dict):
                continue
            client_id = int(raw_client_id)
            self._client_control_variates[client_id] = self._clone_control_state(control_state)
            updated = True

        if updated:
            self._recompute_server_control()
        return feedback

