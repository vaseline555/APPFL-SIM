import gc
import math
import threading
import numbers
from collections.abc import Sequence
from typing import Any, Optional, Union, Dict, OrderedDict
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
        self._prev_pre_val_error = None
        self._cumulative_gen_reward = 0.0
        self._fixed_tau_t = self._resolve_fixed_tau_t()
        self._latest_client_post_update_param_norms: Dict[int, float] = {}
        self._latest_client_context_weights: Dict[int, float] = {}

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

    def _resolve_fixed_tau_t(self) -> int | None:
        candidates = (
            self.scheduler_configs.get("fixed_local_steps", None),
            self.scheduler_configs.get("local_steps", None),
            self.scheduler_configs.get("num_local_steps", None),
            self.scheduler_configs.get("num_local_epochs", None),
        )
        for value in candidates:
            parsed = self._coerce_nonnegative_int(value)
            if parsed is not None:
                return int(parsed)
        return None

    def _resolve_round_learning_rate(self, round_idx: int) -> float:
        base_lr = float(self.scheduler_configs.get("base_lr", 0.01))
        decay_cfg = self.scheduler_configs.get("lr_decay", {})
        if not isinstance(decay_cfg, dict):
            decay_cfg = {}
        enabled = bool(decay_cfg.get("enable", False))
        decay_type = str(decay_cfg.get("type", "none")).strip().lower()
        min_lr = float(decay_cfg.get("min_lr", 0.0))
        if (not enabled) or decay_type in {"", "none", "off", "false"}:
            return float(max(base_lr, min_lr))

        round_value = max(1, int(round_idx))
        elapsed_rounds = max(0, round_value - 1)

        if decay_type in {"exp", "exponential"}:
            gamma = float(decay_cfg.get("gamma", 0.99))
            lr_value = float(base_lr * (gamma**elapsed_rounds))
        elif decay_type in {"step", "steplr"}:
            gamma = float(decay_cfg.get("gamma", 0.99))
            step_size = max(1, int(decay_cfg.get("step_size", 1)))
            decay_steps = elapsed_rounds // step_size
            lr_value = float(base_lr * (gamma**decay_steps))
        elif decay_type in {"multistep", "multi_step", "multisteplr"}:
            raw_milestones = decay_cfg.get("milestones", [])
            if isinstance(raw_milestones, str):
                raw_milestones = [
                    token.strip()
                    for token in raw_milestones.split(",")
                    if token.strip()
                ]
            if not isinstance(raw_milestones, (list, tuple)):
                raw_milestones = []
            milestones = sorted({int(m) for m in raw_milestones if int(m) > 0})
            gamma = float(decay_cfg.get("gamma", 0.99))
            decay_steps = sum(1 for m in milestones if elapsed_rounds >= int(m))
            lr_value = float(base_lr * (gamma**decay_steps))
        elif decay_type in {"cos", "cosine"}:
            t_max = int(decay_cfg.get("t_max", 0))
            if t_max <= 0:
                t_max = int(self.scheduler_configs.get("num_rounds", 1))
            t_max = max(1, t_max)
            eta_min = float(decay_cfg.get("eta_min", 0.0))
            progress = min(elapsed_rounds, t_max)
            cosine = 0.5 * (1.0 + math.cos(math.pi * float(progress) / float(t_max)))
            lr_value = float(eta_min + (base_lr - eta_min) * cosine)
        else:
            lr_value = float(base_lr)

        if min_lr > 0.0 and lr_value < min_lr:
            lr_value = float(min_lr)
        return float(lr_value)

    @staticmethod
    def _extract_context_features(client_stats: Dict[str, Any]) -> Optional[tuple[float, float]]:
        if not isinstance(client_stats, dict):
            return None
        lr_value = client_stats.get("current_lr", None)
        param_norm = client_stats.get("post_update_param_norm", None)
        if not isinstance(lr_value, (int, float)):
            return None
        if not isinstance(param_norm, (int, float)):
            return None
        return float(lr_value), float(param_norm)

    def _store_client_context_feedback(
        self,
        *,
        client_train_stats: Optional[Dict[Union[str, int], Dict[str, Any]]] = None,
        sample_sizes: Optional[Dict[Union[str, int], int]] = None,
    ) -> None:
        if not isinstance(client_train_stats, dict):
            return
        contexts: Dict[int, float] = {}
        weights: Dict[int, float] = {}
        for cid, client_stats in client_train_stats.items():
            try:
                client_id = int(cid)
            except Exception:
                continue
            features = self._extract_context_features(client_stats)
            if features is None:
                continue
            _, post_update_param_norm = features
            contexts[client_id] = float(post_update_param_norm)
            if isinstance(sample_sizes, dict):
                weight = sample_sizes.get(cid, sample_sizes.get(client_id, 1))
            else:
                weight = 1
            if not isinstance(weight, (int, float)) or float(weight) <= 0.0:
                weight = 1.0
            weights[client_id] = float(weight)
        if contexts:
            self._latest_client_post_update_param_norms = contexts
        if weights:
            self._latest_client_context_weights = weights

    def get_pull_kwargs(
        self,
        *,
        selected_ids: Optional[Sequence[int]] = None,
        round_idx: Optional[int] = None,
    ) -> Dict[str, Any]:
        del selected_ids, round_idx
        return {}

    def get_round_metrics(self, *, round_local_steps: Any) -> Dict[str, Any]:
        if round_local_steps is None and self._fixed_tau_t is not None:
            round_local_steps = int(self._fixed_tau_t)
        policy_metrics = self._summarize_round_local_steps(round_local_steps)
        if not policy_metrics:
            return {}
        return {"policy": policy_metrics}

    def update_round_feedback(
        self,
        *,
        round_metrics: Dict[str, Any],
        client_train_stats: Optional[Dict[Union[str, int], Dict[str, Any]]] = None,
        sample_sizes: Optional[Dict[Union[str, int], int]] = None,
    ) -> Dict[str, Any]:
        self._store_client_context_feedback(
            client_train_stats=client_train_stats,
            sample_sizes=sample_sizes,
        )
        if not bool(self.scheduler_configs.get("track_gen_rewards", False)):
            return {"logging": {}}

        current = round_metrics.get("pre_val_loss", None)
        round_reward = None
        if isinstance(current, (int, float)):
            current_value = float(current)
            if isinstance(self._prev_pre_val_error, (int, float)):
                round_reward = float(self._prev_pre_val_error - current_value)
                self._cumulative_gen_reward = float(self._cumulative_gen_reward) + float(
                    round_reward
                )
            self._prev_pre_val_error = float(current_value)

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
