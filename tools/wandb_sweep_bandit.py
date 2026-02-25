#!/usr/bin/env python
"""WandB sweep runner placeholder for APPFL-SIM bandit schedulers.

This script expects to be launched by `wandb agent`.
It dispatches one APPFL-SIM run and forwards selected sweep params
as dotlist overrides to `python -m appfl_sim.runner`.

Notes:
- `posterior_variance` is mapped to SWTS `likelihood_variance`
  because SWTS config uses likelihood/prior variance knobs.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import wandb

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SWUCB_CONFIG = "appfl_sim/config/adaptive_local_steps/cifar10_diri/swucb.yaml"
DEFAULT_SWTS_CONFIG = "appfl_sim/config/adaptive_local_steps/cifar10_diri/swts.yaml"


def _cfg_get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def main() -> int:
    run = wandb.init()
    if run is None:
        raise RuntimeError("wandb.init() returned None")

    cfg = dict(wandb.config)
    scheduler = str(cfg.get("scheduler", "swucb")).strip().lower()
    if scheduler not in {"swucb", "swts"}:
        raise ValueError(f"Unsupported scheduler={scheduler}; expected swucb or swts")

    config_path_key = f"config_path_{scheduler}"
    config_path = str(
        cfg.get(
            "config_path",
            cfg.get(
                config_path_key,
                DEFAULT_SWUCB_CONFIG if scheduler == "swucb" else DEFAULT_SWTS_CONFIG,
            ),
        )
    )

    overrides: list[str] = []
    overrides.append(f"logging.backend={cfg.get('logging_backend', 'wandb')}")
    if "wandb_entity" in cfg and str(cfg["wandb_entity"]).strip():
        overrides.append(f"logging.configs.wandb_entity={cfg['wandb_entity']}")
    if "seed" in cfg:
        overrides.append(f"experiment.seed={int(cfg['seed'])}")

    # Use run id/name to avoid accidental collisions when many agents run.
    run_name = str(cfg.get("logging_name", f"sweep_{scheduler}_{run.id}"))
    overrides.append(f"logging.name={run_name}")

    if "window_size" in cfg:
        overrides.append(f"algorithm.scheduler_kwargs.window_size={int(cfg['window_size'])}")

    if scheduler == "swucb":
        if "exploration_alpha" in cfg:
            overrides.append(
                f"algorithm.scheduler_kwargs.exploration_alpha={float(cfg['exploration_alpha'])}"
            )
    else:
        # Requested as posterior_variance; mapped to implemented likelihood_variance.
        if "posterior_variance" in cfg:
            overrides.append(
                f"algorithm.scheduler_kwargs.likelihood_variance={float(cfg['posterior_variance'])}"
            )
        if "prior_variance" in cfg:
            overrides.append(
                f"algorithm.scheduler_kwargs.prior_variance={float(cfg['prior_variance'])}"
            )

    cmd = [
        sys.executable,
        "-m",
        "appfl_sim.runner",
        "--config",
        config_path,
        *overrides,
    ]

    env = dict(os.environ)
    cuda_visible_devices = _cfg_get(cfg, "cuda_visible_devices", None)
    if cuda_visible_devices is not None and str(cuda_visible_devices).strip():
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)

    print("[wandb_sweep_bandit] command:", " ".join(cmd), flush=True)
    result = subprocess.run(cmd, cwd=str(ROOT), env=env, check=False)
    return int(result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
