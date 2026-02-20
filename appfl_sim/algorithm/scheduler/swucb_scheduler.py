import math
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

from omegaconf import DictConfig

from appfl_sim.algorithm.aggregator import BaseAggregator
from appfl_sim.algorithm.scheduler.sync_scheduler import SyncScheduler


class SwucbScheduler(SyncScheduler):
    """
    Sliding-window UCB for non-contextual local-step adaptation.
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
        self.window_size = max(1, int(scheduler_configs.get("window_size", 50)))
        self.exploration_alpha = float(scheduler_configs.get("exploration_alpha", 0.5))
        self.history: Deque[Tuple[int, float]] = deque(maxlen=self.window_size)
        self.pending_actions: Deque[int] = deque()
        self.prev_global_gen_error: Optional[float] = None
        self.current_round: int = 0
        self.last_selected_action: int = int(self.action_space[0])
        self.last_reward: Optional[float] = None

    def select_local_steps(self, round_idx: int) -> int:
        t = max(1, int(round_idx))
        self.current_round = t
        counts: Dict[int, int] = {a: 0 for a in self.action_space}
        sums: Dict[int, float] = {a: 0.0 for a in self.action_space}
        for action, reward in self.history:
            counts[action] += 1
            sums[action] += float(reward)

        missing = [a for a in self.action_space if counts[a] == 0]
        if missing:
            chosen = int(missing[0])
        else:
            best_action = int(self.action_space[0])
            best_score = float("-inf")
            for action in self.action_space:
                n = max(1, counts[action])
                mean_reward = sums[action] / n
                bonus = self.exploration_alpha * math.sqrt(max(0.0, math.log(float(t))) / n)
                score = mean_reward + bonus
                if score > best_score:
                    best_score = score
                    best_action = int(action)
            chosen = best_action

        self.pending_actions.append(chosen)
        self.last_selected_action = int(chosen)
        return int(chosen)

    def observe_global_gen_error(
        self, global_gen_error: float, round_idx: Optional[int] = None
    ) -> Optional[float]:
        _ = round_idx
        current = float(global_gen_error)
        if self.prev_global_gen_error is None:
            self.prev_global_gen_error = current
            self.last_reward = None
            return None

        reward = -(current - self.prev_global_gen_error)
        self.prev_global_gen_error = current
        self.last_reward = float(reward)

        if self.pending_actions:
            action = int(self.pending_actions.popleft())
            self.history.append((action, float(reward)))
        return float(reward)

    def get_bandit_state(self) -> Dict[str, Any]:
        return {
            "name": "swucb",
            "round": int(self.current_round),
            "last_action": int(self.last_selected_action),
            "last_reward": self.last_reward,
            "window_size": int(self.window_size),
            "history_size": int(len(self.history)),
            "pending_actions": int(len(self.pending_actions)),
        }
