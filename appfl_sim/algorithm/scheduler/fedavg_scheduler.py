import gc
import threading
import numbers
from collections.abc import Sequence
from typing import Any, Union, Dict, OrderedDict
from concurrent.futures import Future

from omegaconf import DictConfig

from appfl_sim.algorithm.scheduler.base_scheduler import BaseScheduler
from appfl_sim.algorithm.aggregator import BaseAggregator


class FedavgScheduler(BaseScheduler):
    def __init__(
        self, scheduler_configs: DictConfig, aggregator: BaseAggregator, logger: Any
    ):
        super().__init__(scheduler_configs, aggregator, logger)
        self.local_models = {}
        self.aggregation_kwargs = {}
        self.future = {}
        self.num_clients = self.scheduler_configs.num_clients
        self._access_lock = threading.Lock()

        self.optimize_memory = bool(scheduler_configs.get("optimize_memory", True))
        self._prev_global_gen_error = None
        self._cumulative_gen_reward = 0.0

    def schedule(
        self,
        client_id: Union[int, str],
        local_model: Union[Dict, OrderedDict],
        **kwargs,
    ) -> Future:
        with self._access_lock:
            future = Future()

            self.local_models[client_id] = local_model

            for key, value in kwargs.items():
                if key not in self.aggregation_kwargs:
                    self.aggregation_kwargs[key] = {}
                self.aggregation_kwargs[key][client_id] = value
            self.future[client_id] = future

            if len(self.local_models) == self.num_clients:
                if self.optimize_memory:
                    aggregated_model = self.aggregator.aggregate(
                        self.local_models, **self.aggregation_kwargs
                    )
                    temp_futures = dict(self.future)
                    self.local_models.clear()
                    self.aggregation_kwargs.clear()

                    while temp_futures:
                        cid, client_future = temp_futures.popitem()
                        client_future.set_result(
                            self._parse_aggregated_model(aggregated_model, cid)
                        )
                    self.future.clear()
                    gc.collect()
                else:
                    aggregated_model = self.aggregator.aggregate(
                        self.local_models, **self.aggregation_kwargs
                    )
                    while self.future:
                        cid, client_future = self.future.popitem()
                        client_future.set_result(
                            self._parse_aggregated_model(aggregated_model, cid)
                        )
                    self.local_models.clear()

            return future

    def _parse_aggregated_model(
        self, aggregated_model: Dict, client_id: Union[int, str]
    ) -> Dict:
        if isinstance(aggregated_model, tuple):
            if client_id in aggregated_model[0]:
                return (aggregated_model[0][client_id], aggregated_model[1])
            return aggregated_model
        if client_id in aggregated_model:
            return aggregated_model[client_id]
        return aggregated_model

    @staticmethod
    def _coerce_nonnegative_int(value: Any):
        try:
            if isinstance(value, numbers.Real):
                return max(0, int(value))
        except Exception:
            return None
        return None

    @classmethod
    def _summarize_round_local_steps(cls, round_local_steps: Any) -> Dict[str, float | int]:
        scalar = cls._coerce_nonnegative_int(round_local_steps)
        if scalar is not None:
            return {"tau_t": int(scalar)}

        values: list[int] = []
        if isinstance(round_local_steps, dict):
            source = round_local_steps.values()
        elif isinstance(round_local_steps, Sequence) and not isinstance(
            round_local_steps, (str, bytes, dict)
        ):
            source = round_local_steps
        else:
            source = []

        for value in source:
            parsed = cls._coerce_nonnegative_int(value)
            if parsed is not None:
                values.append(int(parsed))
        if not values:
            return {}

        return {
            "tau_t_clients": int(len(values)),
            "tau_t_mean": float(sum(values) / float(len(values))),
            "tau_t_min": int(min(values)),
            "tau_t_max": int(max(values)),
        }

    def get_round_metrics(self, *, round_local_steps: Any) -> Dict[str, Any]:
        policy_metrics = self._summarize_round_local_steps(round_local_steps)
        if not policy_metrics:
            return {}
        return {"policy": policy_metrics}

    def update_round_feedback(
        self,
        *,
        round_metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not bool(self.scheduler_configs.get("track_gen_rewards", False)):
            return {"logging": {}}

        current = round_metrics.get("global_gen_error", None)
        round_reward = None
        if isinstance(current, (int, float)):
            current_value = float(current)
            if isinstance(self._prev_global_gen_error, (int, float)):
                round_reward = float(self._prev_global_gen_error - current_value)
                self._cumulative_gen_reward = float(self._cumulative_gen_reward) + float(
                    round_reward
                )
            self._prev_global_gen_error = float(current_value)

        return {
            "logging": {
                "gen_reward": {
                    "round": float(round_reward)
                    if isinstance(round_reward, (int, float))
                    else None,
                    "cumulative": float(self._cumulative_gen_reward),
                }
            }
        }
