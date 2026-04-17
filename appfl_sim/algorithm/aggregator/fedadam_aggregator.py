from __future__ import annotations

from typing import Dict, Optional, OrderedDict, Union

import torch
from omegaconf import DictConfig

from appfl_sim.algorithm.aggregator.fedavg_aggregator import FedavgAggregator
from appfl_sim.misc.system_utils import clone_state_dict_optimized


class FedadamAggregator(FedavgAggregator):
    def __init__(
        self,
        model: Optional[torch.nn.Module] = None,
        aggregator_configs: Optional[DictConfig] = None,
        logger=None,
    ):
        super().__init__(
            model=model,
            aggregator_configs=aggregator_configs,
            logger=logger,
        )
        cfg = aggregator_configs if aggregator_configs is not None else DictConfig({})
        self.server_learning_rate = float(cfg.get("server_learning_rate", 1.0))
        self.beta1 = float(cfg.get("beta1", 0.9))
        self.beta2 = float(cfg.get("beta2", 0.99))
        self.tau = float(cfg.get("tau", cfg.get("epsilon", 1e-3)))
        self._first_moment: Dict[str, torch.Tensor] = {}
        self._second_moment: Dict[str, torch.Tensor] = {}

    def aggregate(
        self,
        local_models: Dict[Union[str, int], Union[Dict, OrderedDict]],
        **kwargs,
    ) -> Dict:
        del kwargs
        local_models = self._normalize_local_models(local_models)
        if not local_models:
            return self.get_parameters()

        self._initialize_global_state(local_models)
        total_clients = len(local_models)

        with torch.no_grad():
            for name, global_tensor in self.global_state.items():
                if (
                    self.named_parameters is None
                    or name not in self.named_parameters
                    or global_tensor.dtype in {torch.int32, torch.int64}
                ):
                    if global_tensor.dtype in {torch.int32, torch.int64}:
                        averaged = torch.zeros_like(global_tensor)
                        for _, model in local_models.items():
                            averaged.add_(model[name])
                        self.global_state[name] = torch.div(
                            averaged, len(local_models)
                        ).type(global_tensor.dtype)
                    else:
                        averaged = torch.zeros_like(global_tensor)
                        for client_id, model in local_models.items():
                            weight = self._client_weight(client_id, total_clients)
                            averaged.add_(model[name], alpha=float(weight))
                        self.global_state[name] = averaged
                    continue

                pseudo_grad = torch.zeros_like(global_tensor)
                for client_id, model in local_models.items():
                    weight = self._client_weight(client_id, total_clients)
                    pseudo_grad.add_(global_tensor - model[name], alpha=float(weight))

                if name not in self._first_moment:
                    self._first_moment[name] = torch.zeros_like(global_tensor)
                if name not in self._second_moment:
                    self._second_moment[name] = torch.zeros_like(global_tensor)

                self._first_moment[name].mul_(self.beta1).add_(
                    pseudo_grad, alpha=(1.0 - self.beta1)
                )
                self._second_moment[name].mul_(self.beta2).addcmul_(
                    pseudo_grad,
                    pseudo_grad,
                    value=(1.0 - self.beta2),
                )
                denom = self._second_moment[name].sqrt().add(self.tau)
                self.global_state[name] = global_tensor - (
                    float(self.server_learning_rate)
                    * self._first_moment[name]
                    / denom
                )

        if self.model is not None:
            self.model.load_state_dict(self.global_state, strict=False)

        if self.optimize_memory:
            return clone_state_dict_optimized(self.global_state)
        return {k: v.clone() for k, v in self.global_state.items()}
