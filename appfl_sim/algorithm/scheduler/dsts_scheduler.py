from typing import Any, Dict, List, Optional

import numpy as np
from omegaconf import DictConfig

from appfl_sim.algorithm.aggregator import BaseAggregator
from appfl_sim.algorithm.scheduler.dsucb_scheduler import _AdaptiveLocalStepSupport
from appfl_sim.algorithm.scheduler.fedavg_scheduler import FedavgScheduler


class DstsScheduler(_AdaptiveLocalStepSupport, FedavgScheduler):
    """
    Discounted Thompson Sampling for non-contextual local-step adaptation.
    """

    def __init__(
        self, scheduler_configs: DictConfig, aggregator: BaseAggregator, logger: Any
    ):
        super().__init__(scheduler_configs=scheduler_configs, aggregator=aggregator, logger=logger)
        self.action_space: List[int] = sorted(
            {
                int(x)
                for x in scheduler_configs.get("action_space", [1, 2, 4, 8])
                if int(x) >= 0
            }
        )
        if not self.action_space:
            self.action_space = [1]

        self.discount_gamma = float(scheduler_configs.get("discount_gamma", 0.99))
        self.discount_gamma = min(max(self.discount_gamma, 1e-8), 1.0)

        self.alpha = max(1e-12, float(scheduler_configs.get("likelihood_variance", 1.0)))
        self.beta = max(1e-12, float(scheduler_configs.get("prior_variance", 1.0)))

        self.N: Dict[int, float] = {a: 0.0 for a in self.action_space}
        self.S: Dict[int, float] = {a: 0.0 for a in self.action_space}
        self.C: Optional[int] = None

        self.prev_pre_val_error: Optional[float] = None
        self.current_round: int = 0
        self.last_selected_action: int = int(self.action_space[0])
        self.last_reward: Optional[float] = None

        seed = scheduler_configs.get("seed", None)
        self._rng = np.random.default_rng(None if seed is None else int(seed))


    def pull(self, round_idx: int) -> int:
        self.current_round = max(1, int(round_idx))

        sampled_scores: Dict[int, float] = {}
        for action in self.action_space:
            pulls = float(self.N[action])
            reward_sum = float(self.S[action])

            v_tau = 1.0 / ((1.0 / self.beta) + ((1.0 / self.alpha) * pulls))
            m_tau = (1.0 / self.alpha) * v_tau * reward_sum
            sampled_scores[action] = float(
                self._rng.normal(loc=m_tau, scale=float(np.sqrt(v_tau)))
            )

        chosen = int(max(self.action_space, key=lambda a: sampled_scores[int(a)]))
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

        chosen = int(self.C) if self.C is not None else None
        reward_value = self._apply_cost_reward(reward_value, chosen)

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
            "name": "dsts",
            "round": int(self.current_round),
            "last_action": int(self.last_selected_action),
            "last_reward": self.last_reward,
            "discount_gamma": float(self.discount_gamma),
        }
