from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from omegaconf import DictConfig, OmegaConf


@dataclass
class TrackerConfig:
    backend: str
    project_name: str
    experiment_name: str
    log_dir: str
    wandb_entity: str = ""
    wandb_mode: str = "online"


def _extract_tracker_config(config: DictConfig | dict) -> TrackerConfig:
    if isinstance(config, DictConfig):
        cfg = OmegaConf.to_container(config, resolve=True)
    else:
        cfg = dict(config)

    backend = str(cfg.get("logging_backend", cfg.get("logger_backend", "file"))).lower()
    project_name = str(cfg.get("project_name", cfg.get("exp_name", "appfl-sim")))
    experiment_name = str(
        cfg.get("experiment_name", cfg.get("exp_name", project_name))
    )
    log_dir = str(cfg.get("log_dir", "./logs"))
    wandb_entity = str(cfg.get("wandb_entity", ""))
    wandb_mode = str(cfg.get("wandb_mode", "online")).lower()
    return TrackerConfig(
        backend=backend,
        project_name=project_name,
        experiment_name=experiment_name,
        log_dir=log_dir,
        wandb_entity=wandb_entity,
        wandb_mode=wandb_mode,
    )


def _has_wandb_api_credential() -> bool:
    if os.getenv("WANDB_API_KEY"):
        return True

    try:
        import netrc

        auth = netrc.netrc().authenticators("api.wandb.ai")
        return auth is not None and bool(auth[2])
    except Exception:
        return False


class ExperimentTracker:
    """Track experiment metrics for file-only, TensorBoard, or Weights & Biases backends."""

    def __init__(self, config: DictConfig | dict):
        cfg = _extract_tracker_config(config)
        self.backend = cfg.backend
        self._writer = None
        self._wandb = None
        self._run = None

        if self.backend in {"none", "file", "console"}:
            return

        if self.backend == "tensorboard":
            try:
                from torch.utils.tensorboard import SummaryWriter
            except Exception as e:  # pragma: no cover
                raise RuntimeError(
                    "TensorBoard logging requested but tensorboard is not installed. "
                    "Install with: pip install tensorboard"
                ) from e

            run_dir = Path(cfg.log_dir) / cfg.project_name
            run_dir.mkdir(parents=True, exist_ok=True)
            self._writer = SummaryWriter(log_dir=str(run_dir))
            return

        if self.backend == "wandb":
            try:
                import wandb
            except Exception as e:  # pragma: no cover
                raise RuntimeError(
                    "WandB logging requested but wandb is not installed. "
                    "Install with: pip install wandb"
                ) from e

            if cfg.wandb_mode != "offline" and not _has_wandb_api_credential():
                raise RuntimeError(
                    "WandB logging requires CLI authentication. "
                    "Run `wandb login` (or set WANDB_API_KEY) before starting simulation."
                )

            init_kwargs = {
                "project": cfg.project_name,
                "name": cfg.experiment_name,
                "dir": cfg.log_dir,
                "mode": cfg.wandb_mode,
                "reinit": True,
            }
            if cfg.wandb_entity:
                init_kwargs["entity"] = cfg.wandb_entity

            self._wandb = wandb
            self._run = wandb.init(**init_kwargs)
            return

        raise ValueError(
            "logging_backend must be one of: file, tensorboard, wandb, none"
        )

    def log_metrics(self, step: int, metrics: Dict[str, float]) -> None:
        if not metrics:
            return
        if self.backend in {"none", "file", "console"}:
            return
        if self.backend == "tensorboard" and self._writer is not None:
            for key, val in metrics.items():
                self._writer.add_scalar(key, float(val), global_step=int(step))
            self._writer.flush()
            return
        if self.backend == "wandb" and self._wandb is not None:
            payload = dict(metrics)
            payload["round"] = int(step)
            self._wandb.log(payload, step=int(step))

    def close(self) -> None:
        if self.backend == "tensorboard" and self._writer is not None:
            self._writer.close()
            self._writer = None
        if self.backend == "wandb" and self._run is not None:
            self._run.finish()
            self._run = None


def create_experiment_tracker(config: DictConfig | dict) -> Optional[ExperimentTracker]:
    tracker = ExperimentTracker(config)
    if tracker.backend in {"none", "file", "console"}:
        return None
    return tracker
