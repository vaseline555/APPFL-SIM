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
        local_models = self._normalize_local_models(local_models)
        if not local_models:
            return self.get_parameters()

        self._initialize_global_state(local_models)
        sample_sizes = kwargs.get("sample_sizes", {})
        round_weights = self._round_client_weights(
            local_models,
            sample_sizes=sample_sizes,
        )

        with torch.no_grad():
            for name, global_tensor in self.global_state.items():
                is_parameter = (
                    self.named_parameters is None or name in self.named_parameters
                )
                if not is_parameter or not global_tensor.is_floating_point():
                    self.global_state[name] = self._weighted_tensor_average(
                        local_models=local_models,
                        round_weights=round_weights,
                        name=name,
                        reference_tensor=global_tensor,
                    )
                    continue

                pseudo_grad = torch.zeros_like(global_tensor)
                for client_id, model in local_models.items():
                    pseudo_grad.add_(
                        global_tensor - model[name],
                        alpha=float(round_weights.get(client_id, 0.0)),
                    )

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
