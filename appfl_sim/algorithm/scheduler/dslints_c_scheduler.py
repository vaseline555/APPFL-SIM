from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from omegaconf import DictConfig

from appfl_sim.algorithm.aggregator import BaseAggregator
from appfl_sim.algorithm.scheduler.dsucb_scheduler import _AdaptiveLocalStepSupport
from appfl_sim.algorithm.scheduler.fedavg_scheduler import FedavgScheduler


class DslintsCScheduler(_AdaptiveLocalStepSupport, FedavgScheduler):
    """
    Discounted LinTS (contextual, client-wise decision).
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
        self.context_dim = max(2, int(scheduler_configs.get("context_dim", 2)))
        self.feature_dim = int(self.context_dim + 1)

        self.discount_gamma = float(scheduler_configs.get("discount_gamma", 0.99))
        self.discount_gamma = min(max(self.discount_gamma, 1e-8), 1.0)
        self.alpha = max(1e-12, float(scheduler_configs.get("ridge_alpha", 1.0)))
        self.beta = max(1e-12, float(scheduler_configs.get("noise_beta", 1.0)))

        self.A = self.alpha * np.eye(self.feature_dim, dtype=float)
        self.b = np.zeros(self.feature_dim, dtype=float)
        self.C: Optional[np.ndarray] = None

        self.prev_pre_val_error: Optional[float] = None
        self.current_round: int = 0
        self.last_selected_actions: Dict[int, int] = {}
        self.last_reward: Optional[float] = None

        seed = scheduler_configs.get("seed", None)
        self._rng = np.random.default_rng(None if seed is None else int(seed))


    def _coerce_context_vector(self, x: Sequence[float]) -> np.ndarray:
        arr = np.asarray(list(x), dtype=float).reshape(-1)
        if arr.size == self.context_dim:
            return arr
        if arr.size > self.context_dim:
            return arr[: self.context_dim]
        padded = np.zeros(self.context_dim, dtype=float)
        padded[: arr.size] = arr
        return padded

    def _resolve_client_contexts(
        self,
        client_contexts: Optional[Dict[int, Sequence[float]] | Sequence[Sequence[float]]] = None,
    ) -> Tuple[List[int], np.ndarray]:
        if client_contexts is None:
            client_ids = list(range(self.num_clients))
            contexts = np.zeros((self.num_clients, self.context_dim), dtype=float)
            return client_ids, contexts

        if isinstance(client_contexts, dict):
            client_ids = sorted(int(cid) for cid in client_contexts.keys())
            contexts = np.zeros((len(client_ids), self.context_dim), dtype=float)
            for idx, cid in enumerate(client_ids):
                contexts[idx] = self._coerce_context_vector(client_contexts[int(cid)])
            return client_ids, contexts

        raw = np.asarray(client_contexts, dtype=float)
        if raw.ndim == 1:
            raw = raw.reshape(1, -1)
        contexts = np.zeros((raw.shape[0], self.context_dim), dtype=float)
        for i in range(raw.shape[0]):
            contexts[i] = self._coerce_context_vector(raw[i])
        client_ids = list(range(int(raw.shape[0])))
        return client_ids, contexts

    def _phi(self, x: np.ndarray, tau: int) -> np.ndarray:
        return np.concatenate([x.astype(float), np.array([float(tau)], dtype=float)])

    def pull(
        self,
        round_idx: int,
        client_contexts: Optional[Dict[int, Sequence[float]] | Sequence[Sequence[float]]] = None,
    ) -> Dict[int, int]:
        self.current_round = max(1, int(round_idx))
        client_ids, contexts = self._resolve_client_contexts(client_contexts=client_contexts)

        A_inv = np.linalg.pinv(self.A)
        w_hat = A_inv @ self.b
        cov = 0.5 * (A_inv + A_inv.T)
        w_tilde = self._rng.multivariate_normal(mean=w_hat, cov=cov, check_valid="ignore")

        decisions: Dict[int, int] = {}
        psi_sum = np.zeros(self.feature_dim, dtype=float)

        for idx, cid in enumerate(client_ids):
            x_i = contexts[idx]
            chosen = int(self.action_space[0])
            best_score = float("-inf")
            chosen_phi = self._phi(x_i, chosen)
            for action in self.action_space:
                psi = self._phi(x_i, int(action))
                score = float(psi @ w_tilde)
                if score > best_score:
                    best_score = score
                    chosen = int(action)
                    chosen_phi = psi
            decisions[int(cid)] = int(chosen)
            psi_sum += chosen_phi

        self.C = np.asarray(psi_sum, dtype=float)
        self.last_selected_actions = dict(decisions)
        return decisions

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
        current_lr = self._resolve_round_learning_rate(1 if round_idx is None else int(round_idx))
        client_contexts: Dict[int, List[float]] = {}
        for cid in client_ids:
            post_norm = float(self._latest_client_post_update_param_norms.get(int(cid), 0.0))
            client_contexts[int(cid)] = self._coerce_context_vector(
                [float(current_lr), post_norm]
            ).tolist()
        return {"client_contexts": client_contexts}

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

        psi = np.asarray(self.C, dtype=float).reshape(-1)
        self.C = None

        beta_inv = 1.0 / self.beta
        self.A = (
            self.discount_gamma * self.A
            + beta_inv * np.outer(psi, psi)
            + (1.0 - self.discount_gamma) * self.alpha * np.eye(self.feature_dim, dtype=float)
        )
        self.b = self.discount_gamma * self.b + beta_inv * float(reward_value) * psi
        return float(reward_value)

    def get_bandit_state(self) -> Dict[str, Any]:
        return {
            "name": "dslints_c",
            "round": int(self.current_round),
            "last_action_count": int(len(self.last_selected_actions)),
            "last_reward": self.last_reward,
            "discount_gamma": float(self.discount_gamma),
        }
