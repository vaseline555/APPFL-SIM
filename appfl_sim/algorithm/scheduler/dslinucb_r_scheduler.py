from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from omegaconf import DictConfig

from appfl_sim.algorithm.aggregator import BaseAggregator
from appfl_sim.algorithm.scheduler.dsucb_scheduler import _AdaptiveLocalStepSupport
from appfl_sim.algorithm.scheduler.fedavg_scheduler import FedavgScheduler


class DslinucbRScheduler(_AdaptiveLocalStepSupport, FedavgScheduler):
    """
    Discounted LinUCB (contextual, round-wise decision).
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

        self.num_clients = max(1, int(scheduler_configs.get("num_clients", 1)))
        self.context_dim = max(1, int(len(self.context_subjects)))
        self.feature_dim = int(self.context_dim + 1)

        self.discount_gamma = float(scheduler_configs.get("discount_gamma", 0.99))
        self.discount_gamma = min(max(self.discount_gamma, 1e-8), 1.0)
        self.alpha = max(1e-12, float(scheduler_configs.get("ridge_alpha", 1.0)))
        self.beta = float(scheduler_configs.get("exploration_beta", 0.1))

        self.A = self.alpha * np.eye(self.feature_dim, dtype=float)
        self.b = np.zeros(self.feature_dim, dtype=float)
        self.C: Optional[np.ndarray] = None

        self.prev_pre_val_error: Optional[float] = None
        self.current_round: int = 0
        self.last_selected_action: int = int(self.action_space[0])
        self.last_reward: Optional[float] = None


    def _coerce_context_vector(self, x: Sequence[float]) -> np.ndarray:
        arr = np.asarray(list(x), dtype=float).reshape(-1)
        if arr.size == self.context_dim:
            return arr
        if arr.size > self.context_dim:
            return arr[: self.context_dim]
        padded = np.zeros(self.context_dim, dtype=float)
        padded[: arr.size] = arr
        return padded

    def _resolve_contexts_and_weights(
        self,
        client_contexts: Optional[Sequence[Sequence[float]]] = None,
        client_weights: Optional[Sequence[float]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if client_contexts is None:
            contexts = np.zeros((self.num_clients, self.context_dim), dtype=float)
        else:
            raw = np.asarray(client_contexts, dtype=float)
            if raw.ndim == 1:
                raw = raw.reshape(1, -1)
            contexts = np.zeros((raw.shape[0], self.context_dim), dtype=float)
            for i in range(raw.shape[0]):
                contexts[i] = self._coerce_context_vector(raw[i])

        k = int(contexts.shape[0])
        if client_weights is None:
            weights = np.ones(k, dtype=float)
        else:
            weights = np.asarray(client_weights, dtype=float).reshape(-1)
            if weights.size != k:
                resized = np.ones(k, dtype=float)
                copied = min(k, int(weights.size))
                if copied > 0:
                    resized[:copied] = weights[:copied]
                weights = resized

        weight_sum = float(np.sum(weights))
        if weight_sum <= 0.0:
            weights = np.ones(k, dtype=float) / float(max(1, k))
        else:
            weights = weights / weight_sum
        return contexts, weights

    def _phi(self, x: np.ndarray, tau: int) -> np.ndarray:
        return np.concatenate([x.astype(float), np.array([float(tau)], dtype=float)])

    def pull(
        self,
        round_idx: int,
        client_contexts: Optional[Sequence[Sequence[float]]] = None,
        client_weights: Optional[Sequence[float]] = None,
    ) -> int:
        self.current_round = max(1, int(round_idx))
        contexts, weights = self._resolve_contexts_and_weights(
            client_contexts=client_contexts,
            client_weights=client_weights,
        )
        x_bar = np.sum(contexts * weights[:, None], axis=0)

        A_inv = np.linalg.pinv(self.A)
        w_hat = A_inv @ self.b

        chosen = int(self.action_space[0])
        best_score = float("-inf")
        chosen_phi = self._phi(x_bar, chosen)
        for action in self.action_space:
            psi = self._phi(x_bar, int(action))
            exploit = float(psi @ w_hat)
            uncertainty = float(np.sqrt(max(0.0, psi @ A_inv @ psi)))
            score = float(exploit + self.beta * uncertainty)
            if score > best_score:
                best_score = score
                chosen = int(action)
                chosen_phi = psi

        self.C = np.asarray(chosen_phi, dtype=float)
        self.last_selected_action = int(chosen)
        return int(chosen)

    def get_pull_kwargs(
        self,
        *,
        selected_ids: Optional[Sequence[int]] = None,
        round_idx: Optional[int] = None,
    ) -> Dict[str, Any]:
        client_ids = (
            [int(cid) for cid in selected_ids]
            if selected_ids is not None
            else list(range(self.num_clients))
        )
        current_round = 1 if round_idx is None else int(round_idx)
        client_contexts = []
        client_weights = []
        for cid in client_ids:
            client_contexts.append(
                self._coerce_context_vector(
                    self._build_client_context_vector(
                        client_id=int(cid),
                        round_idx=current_round,
                    )
                ).tolist()
            )
            client_weights.append(
                float(self._latest_client_context_weights.get(int(cid), 1.0))
            )
        return {
            "client_contexts": client_contexts,
            "client_weights": client_weights,
        }

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
            reward_value = float(
                self._scale_reward(float(self.prev_pre_val_error - current))
            )
            self.prev_pre_val_error = current
        else:
            reward_value = float(self._scale_reward(float(reward)))

        self.last_reward = float(reward_value)

        if self.C is None:
            return float(reward_value)

        psi = np.asarray(self.C, dtype=float).reshape(-1)
        self.C = None

        self.A = (
            self.discount_gamma * self.A
            + np.outer(psi, psi)
            + (1.0 - self.discount_gamma) * self.alpha * np.eye(self.feature_dim, dtype=float)
        )
        self.b = self.discount_gamma * self.b + float(reward_value) * psi
        return float(reward_value)

    def get_bandit_state(self) -> Dict[str, Any]:
        return {
            "name": "dslinucb_r",
            "round": int(self.current_round),
            "contexts": list(self.context_subjects),
            "last_action": int(self.last_selected_action),
            "last_reward": self.last_reward,
            "discount_gamma": float(self.discount_gamma),
        }
