from __future__ import annotations

import gc
from typing import Any, Dict, Optional, OrderedDict, Union

import torch

from appfl_sim.algorithm.aggregator.fedavg_aggregator import FedavgAggregator
from appfl_sim.misc.system_utils import clone_state_dict_optimized, safe_inplace_operation


class FednovaAggregator(FedavgAggregator):
    @staticmethod
    def _client_stats_for_id(
        client_train_stats: Dict[Union[str, int], Dict[str, Any]],
        client_id: Union[str, int],
    ) -> Dict[str, Any]:
        if not isinstance(client_train_stats, dict):
            return {}
        stats = client_train_stats.get(client_id, None)
        if isinstance(stats, dict):
            return stats
        stats = client_train_stats.get(str(client_id), None)
        return stats if isinstance(stats, dict) else {}

    def _resolve_client_normalizer(
        self,
        client_id: Union[str, int],
        client_train_stats: Dict[Union[str, int], Dict[str, Any]],
    ) -> float:
        stats = self._client_stats_for_id(client_train_stats, client_id)
        for key in (
            "fednova_a_i",
            "fednova_tau",
            "completed_local_steps",
            "current_local_steps",
        ):
            value = stats.get(key, None)
            if isinstance(value, (int, float)) and float(value) > 0.0:
                return float(value)
        return 1.0

    def aggregate(
        self,
        local_models: Dict[Union[str, int], Union[Dict, OrderedDict]],
        **kwargs,
    ) -> Dict:
        local_models = self._normalize_local_models(local_models)
        if not local_models:
            return self.get_parameters()

        client_train_stats = kwargs.get("client_train_stats", {})
        self._initialize_global_state(local_models)

        normalizers = {
            client_id: self._resolve_client_normalizer(client_id, client_train_stats)
            for client_id in local_models
        }
        tau_eff = 0.0
        total_clients = len(local_models)
        for client_id in local_models:
            tau_eff += self._client_weight(client_id, total_clients) * normalizers[client_id]
        if tau_eff <= 0.0:
            tau_eff = 1.0

        self.step = {}
        with torch.no_grad():
            for name in self.global_state:
                if (
                    self.named_parameters is not None
                    and name not in self.named_parameters
                ) or self.global_state[name].dtype in {
                    torch.int32,
                    torch.int64,
                }:
                    continue
                self.step[name] = torch.zeros_like(self.global_state[name])

            for client_id, model in local_models.items():
                weight = self._client_weight(client_id, total_clients)
                normalizer = max(float(normalizers[client_id]), 1e-12)
                for name in model:
                    if name not in self.step:
                        continue
                    normalized_diff = (model[name] - self.global_state[name]) / normalizer
                    self.step[name] = safe_inplace_operation(
                        self.step[name],
                        "add",
                        normalized_diff * weight,
                    )

            for name in list(self.step.keys()):
                self.step[name] = safe_inplace_operation(
                    self.step[name], "mul", float(tau_eff)
                )

            for name in self.global_state:
                if name in self.step:
                    self.global_state[name] = safe_inplace_operation(
                        self.global_state[name],
                        "add",
                        self.step[name],
                    )
                else:
                    param_sum = torch.zeros_like(self.global_state[name])
                    for _, model in local_models.items():
                        param_sum = safe_inplace_operation(param_sum, "add", model[name])
                    self.global_state[name] = safe_inplace_operation(
                        param_sum, "div", len(local_models)
                    )
                    del param_sum

        gc.collect()
        self.step.clear()

        if self.model is not None:
            self.model.load_state_dict(self.global_state, strict=False)

        if self.optimize_memory:
            return clone_state_dict_optimized(self.global_state)
        return {k: v.clone() for k, v in self.global_state.items()}

