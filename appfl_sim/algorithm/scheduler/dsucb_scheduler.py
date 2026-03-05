import math
import numbers
from typing import Any, Dict, List, Optional, Sequence

from omegaconf import DictConfig

from appfl_sim.algorithm.aggregator import BaseAggregator
from appfl_sim.algorithm.scheduler.fedavg_scheduler import FedavgScheduler


class _AdaptiveLocalStepSupport:
    @classmethod
    def required_data_fields(cls) -> set[str]:
        return {"eval.configs.dataset_ratio"}

    @staticmethod
    def _coerce_positive_int(value: Any) -> Optional[int]:
        try:
            if isinstance(value, numbers.Real):
                parsed = int(value)
                return max(1, parsed)
        except Exception:
            return None
        return None

    @classmethod
    def resolve_client_local_steps(
        cls,
        round_local_steps: Any,
        client_id: int,
    ) -> Optional[int]:
        if round_local_steps is None:
            return None

        scalar = cls._coerce_positive_int(round_local_steps)
        if scalar is not None:
            return scalar

        if isinstance(round_local_steps, dict):
            value = None
            if int(client_id) in round_local_steps:
                value = round_local_steps.get(int(client_id))
            elif str(int(client_id)) in round_local_steps:
                value = round_local_steps.get(str(int(client_id)))
            return cls._coerce_positive_int(value)

        if isinstance(round_local_steps, Sequence) and not isinstance(
            round_local_steps, (str, bytes, dict)
        ):
            idx = int(client_id)
            if idx < 0 or idx >= len(round_local_steps):
                return None
            return cls._coerce_positive_int(round_local_steps[idx])

        return None

    @classmethod
    def summarize_round_local_steps(cls, round_local_steps: Any) -> Dict[str, float | int]:
        scalar = cls._coerce_positive_int(round_local_steps)
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
            parsed = cls._coerce_positive_int(value)
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
        policy_metrics = self.summarize_round_local_steps(round_local_steps)
        if not policy_metrics:
            return {}
        return {"policy": policy_metrics}


class DsucbScheduler(_AdaptiveLocalStepSupport, FedavgScheduler):
    """
    Discounted UCB for non-contextual local-step adaptation.
    """

    def __init__(
        self, scheduler_configs: DictConfig, aggregator: BaseAggregator, logger: Any
    ):
        super().__init__(scheduler_configs=scheduler_configs, aggregator=aggregator, logger=logger)
        self.action_space: List[int] = sorted(
            {
                int(x)
                for x in scheduler_configs.get("action_space", [1, 2, 4, 8])
                if int(x) > 0
            }
        )
        if not self.action_space:
            self.action_space = [1]

        self.discount_gamma = float(scheduler_configs.get("discount_gamma", 0.99))
        self.discount_gamma = min(max(self.discount_gamma, 1e-8), 1.0)
        self.exploration_alpha = float(scheduler_configs.get("exploration_alpha", 0.1))

        self.N: Dict[int, float] = {a: 0.0 for a in self.action_space}
        self.S: Dict[int, float] = {a: 0.0 for a in self.action_space}
        self.C: Optional[int] = None

        self.prev_pre_val_error: Optional[float] = None
        self.current_round: int = 0
        self.last_selected_action: int = int(self.action_space[0])
        self.last_reward: Optional[float] = None


    def pull(self, round_idx: int) -> int:
        self.current_round = max(1, int(round_idx))

        chosen: Optional[int] = None
        for action in self.action_space:
            if float(self.N[action]) == 0.0:
                chosen = int(action)
                self.N[chosen] = 1.0
                break

        if chosen is None:
            best_score = float("-inf")
            for action in self.action_space:
                pulls = max(1e-12, float(self.N[action]))
                mean_reward = float(self.S[action]) / float(pulls)
                bonus = self.exploration_alpha * math.sqrt(
                    max(0.0, math.log(float(self.current_round))) / float(pulls)
                )
                score = float(mean_reward + bonus)
                if score > best_score:
                    best_score = score
                    chosen = int(action)

        if chosen is None:
            chosen = int(self.action_space[0])

        self.C = int(chosen)
        self.last_selected_action = int(chosen)
        return int(chosen)

    def adapt(
        self,
        pre_val_error: Optional[float] = None,
        reward: Optional[float] = None,
    ) -> Optional[float]:
        if reward is None:
            if pre_val_error is None:
                return None
            current = float(pre_val_error)
            if self.prev_pre_val_error is None:
                self.prev_pre_val_error = current
                self.last_reward = None
                return None
            reward_value = float(self.prev_pre_val_error - current)
            self.prev_pre_val_error = current
        else:
            reward_value = float(reward)

        self.last_reward = float(reward_value)

        if self.C is None:
            return float(reward_value)

        chosen = int(self.C)
        self.C = None

        for action in self.action_space:
            self.N[action] = self.discount_gamma * float(self.N[action])
            self.S[action] = self.discount_gamma * float(self.S[action])

        self.N[chosen] = float(self.N[chosen]) + 1.0
        self.S[chosen] = float(self.S[chosen]) + float(reward_value)
        return float(reward_value)

    def get_bandit_state(self) -> Dict[str, Any]:
        return {
            "name": "dsucb",
            "round": int(self.current_round),
            "last_action": int(self.last_selected_action),
            "last_reward": self.last_reward,
            "discount_gamma": float(self.discount_gamma),
        }
