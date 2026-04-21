from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from omegaconf import DictConfig, OmegaConf
from appfl_sim.misc.logging_utils import _remap_server_wandb_payload


@dataclass
class TrackerConfig:
    backend: str
    project_name: str
    run_name: str
    log_dir: str
    experiment_seed: str
    wandb_entity: str = ""
    wandb_mode: str = "online"
    wandb_group: str = ""
    wandb_tags: list[str] | None = None
    wandb_notes: str = ""


def _cfg_get(payload: dict, path: str, default: Any = None) -> Any:
    parts = [p for p in str(path).split(".") if p]
    cur: Any = payload
    for part in parts:
        if isinstance(cur, dict):
            if part not in cur:
                return default
            cur = cur[part]
            continue
        return default
    return default if cur is None else cur


def _extract_tracker_config(config: DictConfig | dict) -> TrackerConfig:
    if isinstance(config, DictConfig):
        cfg = OmegaConf.to_container(config, resolve=True)
    else:
        cfg = dict(config)

    backend = str(_cfg_get(cfg, "logging.backend", "file")).lower()
    experiment_name = str(_cfg_get(cfg, "experiment.name", "appfl-sim"))
    run_name = str(_cfg_get(cfg, "logging.name", experiment_name))
    project_name = experiment_name
    log_dir = str(_cfg_get(cfg, "logging.path", "./logs"))
    experiment_seed = str(_cfg_get(cfg, "experiment.seed", 0))
    wandb_entity = str(_cfg_get(cfg, "logging.configs.wandb_entity", ""))
    wandb_mode = str(_cfg_get(cfg, "logging.configs.wandb_mode", "online")).lower()
    wandb_group = str(_cfg_get(cfg, "logging.configs.wandb_group", "")).strip()
    wandb_notes = str(_cfg_get(cfg, "logging.configs.wandb_notes", "")).strip()
    wandb_tags_cfg = _cfg_get(cfg, "logging.configs.wandb_tags", None)
    wandb_tags: list[str] | None
    if isinstance(wandb_tags_cfg, list):
        wandb_tags = [str(tag).strip() for tag in wandb_tags_cfg if str(tag).strip()]
    elif isinstance(wandb_tags_cfg, str):
        wandb_tags = [
            token.strip()
            for token in wandb_tags_cfg.split(",")
            if token.strip()
        ]
    else:
        wandb_tags = None
    return TrackerConfig(
        backend=backend,
        project_name=project_name,
        run_name=run_name,
        log_dir=log_dir,
        experiment_seed=experiment_seed,
        wandb_entity=wandb_entity,
        wandb_mode=wandb_mode,
        wandb_group=wandb_group,
        wandb_tags=wandb_tags,
        wandb_notes=wandb_notes,
    )


def _resolve_config_payload(config: DictConfig | dict) -> dict[str, Any]:
    if isinstance(config, DictConfig):
        payload = OmegaConf.to_container(config, resolve=True)
        return payload if isinstance(payload, dict) else {}
    return dict(config) if isinstance(config, dict) else {}


def _json_compatible_config_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(k): _json_compatible_config_value(v)
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_compatible_config_value(v) for v in value]
    if isinstance(value, (str, bool)) or value is None:
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    if hasattr(value, "item"):
        try:
            scalar = value.item()
            if isinstance(scalar, (str, bool)) or scalar is None:
                return scalar
            if isinstance(scalar, int):
                return int(scalar)
            if isinstance(scalar, float):
                return float(scalar)
        except Exception:
            pass
    return str(value)


def _flatten_config_for_wandb(
    payload: dict[str, Any],
    prefix: str = "",
) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for raw_key, raw_value in payload.items():
        key = str(raw_key).strip()
        if key == "":
            continue
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(raw_value, dict):
            nested = _flatten_config_for_wandb(raw_value, prefix=full_key)
            if nested:
                flat.update(nested)
            else:
                flat[full_key] = {}
            continue
        flat[full_key] = _json_compatible_config_value(raw_value)
    return flat


def _build_wandb_config_payload(
    source_cfg: dict[str, Any],
    tracker_cfg: TrackerConfig,
) -> dict[str, Any]:
    payload = _flatten_config_for_wandb(source_cfg)
    payload["logging_name"] = tracker_cfg.run_name
    payload["experiment_name"] = tracker_cfg.project_name
    payload["wandb_run_name"] = tracker_cfg.run_name
    payload["wandb_project_name"] = tracker_cfg.project_name
    if tracker_cfg.wandb_group:
        payload["wandb_group"] = tracker_cfg.wandb_group

    dataset_name = _cfg_get(source_cfg, "dataset.name", "")
    algorithm_name = _cfg_get(source_cfg, "algorithm.name", "")
    model_name = _cfg_get(source_cfg, "model.name", "")
    if str(dataset_name).strip():
        payload["dataset_name"] = str(dataset_name).strip()
    if str(algorithm_name).strip():
        payload["algorithm_name"] = str(algorithm_name).strip()
    if str(model_name).strip():
        payload["model_name"] = str(model_name).strip()
    return payload


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

    def __init__(self, config: DictConfig | dict, run_id: str):
        source_cfg = _resolve_config_payload(config)
        cfg = _extract_tracker_config(config)
        run_id_text = str(run_id).strip()
        if run_id_text == "":
            raise ValueError("run_id must be a non-empty string.")
        self.backend = cfg.backend
        self._writer = None
        self._wandb = None
        self._run = None
        self._metrics_json_path = None
        self._metrics_records: list[dict[str, Any]] = []

        run_dir = Path(cfg.log_dir) / cfg.project_name / cfg.run_name / run_id_text
        run_dir.mkdir(parents=True, exist_ok=True)
        self._metrics_json_path = run_dir / "metrics.json"
        if self._metrics_json_path.exists():
            try:
                content = json.loads(
                    self._metrics_json_path.read_text(encoding="utf-8")
                )
                if isinstance(content, list):
                    self._metrics_records = content
            except Exception:
                self._metrics_records = []

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

            wandb_config = _build_wandb_config_payload(source_cfg, cfg)
            init_kwargs = {
                "project": cfg.project_name,
                "name": cfg.run_name,
                "dir": cfg.log_dir,
                "mode": cfg.wandb_mode,
                "reinit": "finish_previous",
                "config": wandb_config,
            }
            tags = list(dict.fromkeys(cfg.wandb_tags or []))
            if tags:
                init_kwargs["tags"] = tags
            seed_note = f"seed:{cfg.experiment_seed}"
            init_kwargs["notes"] = (
                f"{cfg.wandb_notes}\n{seed_note}".strip()
                if cfg.wandb_notes
                else seed_note
            )
            if cfg.wandb_entity:
                init_kwargs["entity"] = cfg.wandb_entity
            if cfg.wandb_group:
                init_kwargs["group"] = cfg.wandb_group

            self._wandb = wandb
            self._run = wandb.init(**init_kwargs)
            if self._run is not None and wandb_config:
                self._run.config.update(wandb_config, allow_val_change=True)
            return

        raise ValueError(
            "logging.backend must be one of: none, file, console, tensorboard, wandb"
        )

    @staticmethod
    def _to_json_compatible(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                str(k): ExperimentTracker._to_json_compatible(v)
                for k, v in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [ExperimentTracker._to_json_compatible(v) for v in value]
        if isinstance(value, (str, bool)) or value is None:
            return value
        if isinstance(value, int):
            return int(value)
        if isinstance(value, float):
            return float(value)
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                pass
        return str(value)

    @staticmethod
    def _flatten_numeric_metrics(metrics: Dict[str, Any], prefix: str = "") -> Dict[str, float]:
        flat: Dict[str, float] = {}
        for key, val in metrics.items():
            name = f"{prefix}/{key}" if prefix else str(key)
            if (
                name == "assigned_local_steps"
                or name.startswith("assigned_local_steps/")
                or name == "local_displacement_by_client"
                or name.startswith("local_displacement_by_client/")
            ):
                continue
            if isinstance(val, dict):
                flat.update(ExperimentTracker._flatten_numeric_metrics(val, prefix=name))
                continue
            if isinstance(val, (int, float)):
                flat[name] = float(val)
                continue
            if hasattr(val, "item"):
                try:
                    scalar = val.item()
                    if isinstance(scalar, (int, float)):
                        flat[name] = float(scalar)
                except Exception:
                    continue
        return flat

    def log_metrics(self, step: int, metrics: Dict[str, Any]) -> None:
        if not metrics:
            return
        if self._metrics_json_path is not None:
            payload = {
                "round": int(step),
                "metrics": self._to_json_compatible(metrics),
            }
            self._metrics_records.append(payload)
            self._metrics_json_path.write_text(
                json.dumps(self._metrics_records, indent=4, ensure_ascii=False),
                encoding="utf-8",
            )
        if self.backend in {"none", "file", "console"}:
            return
        if self.backend == "tensorboard" and self._writer is not None:
            for key, val in self._flatten_numeric_metrics(metrics).items():
                self._writer.add_scalar(key, float(val), global_step=int(step))
            self._writer.flush()
            return
        if self.backend == "wandb" and self._wandb is not None:
            payload = self._flatten_numeric_metrics(metrics)
            payload["round"] = int(step)
            self._wandb.log(_remap_server_wandb_payload(payload), step=int(step))

    def close(self) -> None:
        if self.backend == "tensorboard" and self._writer is not None:
            self._writer.close()
            self._writer = None
        if self.backend == "wandb" and self._run is not None:
            self._run.finish()
            self._run = None


def create_experiment_tracker(
    config: DictConfig | dict, run_id: str
) -> ExperimentTracker:
    return ExperimentTracker(config, run_id=run_id)
