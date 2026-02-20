import torch
from typing import Dict, OrderedDict, Union, Any, Optional
from omegaconf import DictConfig
from appfl_sim.algorithm.aggregator.fedavg_aggregator import FedAvgAggregator
from appfl_sim.misc.system_utils import safe_inplace_operation, optimize_memory_cleanup


class FedNovaAggregator(FedAvgAggregator):
    """
    FedNova-style normalized aggregation.
    For each client i, aggregate normalized delta:
        (w_i - w_t) / tau_i
    where tau_i is the local step count used by client i in that round.
    """

    def __init__(
        self,
        model: Optional[torch.nn.Module] = None,
        aggregator_configs: DictConfig = DictConfig({}),
        logger: Optional[Any] = None,
    ):
        super().__init__(
            model=model,
            aggregator_configs=aggregator_configs,
            logger=logger,
        )
        self._client_local_steps: Dict[Union[str, int], int] = {}

    @staticmethod
    def _safe_local_steps(value: Any) -> int:
        try:
            parsed = int(value)
        except Exception:
            return 1
        return max(1, parsed)

    def aggregate(
        self, local_models: Dict[Union[str, int], Union[Dict, OrderedDict]], **kwargs
    ) -> Dict:
        client_train_stats = kwargs.get("client_train_stats", {}) or {}
        local_steps_map: Dict[Union[str, int], int] = {}
        for client_id in local_models:
            stats = client_train_stats.get(client_id, {})
            steps = 1
            if isinstance(stats, dict) and "current_local_steps" in stats:
                steps = self._safe_local_steps(stats.get("current_local_steps"))
            local_steps_map[client_id] = steps
        self._client_local_steps = local_steps_map
        return super().aggregate(local_models, **kwargs)

    def compute_steps(
        self, local_models: Dict[Union[str, int], Union[Dict, OrderedDict]]
    ):
        if self.optimize_memory:
            with torch.no_grad():
                for name in self.global_state:
                    if (
                        self.named_parameters is not None
                        and name not in self.named_parameters
                    ) or (
                        self.global_state[name].dtype == torch.int64
                        or self.global_state[name].dtype == torch.int32
                    ):
                        continue
                    self.step[name] = torch.zeros_like(self.global_state[name])

                for client_id, model in local_models.items():
                    weight = self._client_weight(client_id, len(local_models))
                    local_steps = float(
                        self._safe_local_steps(self._client_local_steps.get(client_id, 1))
                    )
                    for name in model:
                        if name in self.step:
                            diff = model[name] - self.global_state[name]
                            normalized_diff = safe_inplace_operation(diff, "div", local_steps)
                            weighted_diff = normalized_diff * weight
                            self.step[name] = safe_inplace_operation(
                                self.step[name], "add", weighted_diff
                            )
                            optimize_memory_cleanup(
                                diff, normalized_diff, weighted_diff, force_gc=False
                            )
        else:
            for name in self.global_state:
                if (
                    self.named_parameters is not None
                    and name not in self.named_parameters
                ) or (
                    self.global_state[name].dtype == torch.int64
                    or self.global_state[name].dtype == torch.int32
                ):
                    continue
                self.step[name] = torch.zeros_like(self.global_state[name])

            for client_id, model in local_models.items():
                weight = self._client_weight(client_id, len(local_models))
                local_steps = float(
                    self._safe_local_steps(self._client_local_steps.get(client_id, 1))
                )
                for name in model:
                    if name in self.step:
                        self.step[name] += weight * (
                            (model[name] - self.global_state[name]) / local_steps
                        )
