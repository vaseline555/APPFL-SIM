#!/usr/bin/env python
"""WandB sweep runner placeholder for APPFL-SIM bandit schedulers.

This script expects to be launched by `wandb agent`.
It dispatches one APPFL-SIM run and forwards selected sweep params
as dotlist overrides to `python -m appfl_sim.runner`.

Notes:
- `posterior_variance` is mapped to `likelihood_variance` for TS variants
  (`swts`, `dsts`) because config keys use likelihood/prior variance knobs.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CONFIG_BY_SCHEDULER = {
    "swucb": "appfl_sim/config/adaptive_local_steps/cifar10_diri/swucb.yaml",
    "swts": "appfl_sim/config/adaptive_local_steps/cifar10_diri/swts.yaml",
    "dsucb": "appfl_sim/config/adaptive_local_steps/cifar10_diri/dsucb.yaml",
    "dsts": "appfl_sim/config/adaptive_local_steps/cifar10_diri/dsts.yaml",
    "dslinucb_r": "appfl_sim/config/adaptive_local_steps/cifar10_diri/dslinucb_r.yaml",
    "dslints_r": "appfl_sim/config/adaptive_local_steps/cifar10_diri/dslints_r.yaml",
    "dslinucb_c": "appfl_sim/config/adaptive_local_steps/cifar10_diri/dslinucb_c.yaml",
    "dslints_c": "appfl_sim/config/adaptive_local_steps/cifar10_diri/dslints_c.yaml",
}


def _cfg_get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _parse_agent_cli(argv: list[str]) -> dict[str, str]:
    cfg: dict[str, str] = {}
    for token in argv:
        text = str(token).strip()
        if not text.startswith("--"):
            continue
        if "=" not in text:
            continue
        key, value = text[2:].split("=", 1)
        key = str(key).strip()
        if key:
            cfg[key] = str(value).strip()
    return cfg


def main() -> int:
    cfg = _parse_agent_cli(sys.argv[1:])
    sweep_run_id = str(os.environ.get("WANDB_RUN_ID", "")).strip()
    scheduler = str(cfg.get("scheduler", "swucb")).strip().lower()
    if scheduler not in DEFAULT_CONFIG_BY_SCHEDULER:
        supported = ", ".join(sorted(DEFAULT_CONFIG_BY_SCHEDULER.keys()))
        raise ValueError(
            f"Unsupported scheduler={scheduler}; expected one of: {supported}"
        )

    config_path_key = f"config_path_{scheduler}"
    config_path = str(
        cfg.get(
            "config_path",
            cfg.get(
                config_path_key,
                DEFAULT_CONFIG_BY_SCHEDULER[scheduler],
            ),
        )
    )

    overrides: list[str] = []
    overrides.append(f"logging.backend={cfg.get('logging_backend', 'wandb')}")
    backend_override = str(
        cfg.get("backend_override", os.environ.get("SWEEP_BACKEND", ""))
    ).strip().lower()
    if backend_override in {"serial", "nccl", "gloo"}:
        overrides.append(f"experiment.backend={backend_override}")
    if "wandb_entity" in cfg and str(cfg["wandb_entity"]).strip():
        overrides.append(f"logging.configs.wandb_entity={cfg['wandb_entity']}")
    if "seed" in cfg:
        overrides.append(f"experiment.seed={int(cfg['seed'])}")

    # Use run id/name to avoid accidental collisions when many agents run.
    run_suffix = sweep_run_id if sweep_run_id else "local"
    run_name = str(cfg.get("logging_name", f"sweep_{scheduler}_{run_suffix}"))
    overrides.append(f"logging.name={run_name}")

    if "window_size" in cfg:
        overrides.append(f"algorithm.scheduler_kwargs.window_size={int(cfg['window_size'])}")
    if "discount_gamma" in cfg:
        overrides.append(
            f"algorithm.scheduler_kwargs.discount_gamma={float(cfg['discount_gamma'])}"
        )
    if "ridge_alpha" in cfg:
        overrides.append(
            f"algorithm.scheduler_kwargs.ridge_alpha={float(cfg['ridge_alpha'])}"
        )
    if "context_dim" in cfg:
        overrides.append(f"algorithm.scheduler_kwargs.context_dim={int(cfg['context_dim'])}")

    if scheduler in {"swucb", "dsucb"}:
        if "exploration_alpha" in cfg:
            overrides.append(
                f"algorithm.scheduler_kwargs.exploration_alpha={float(cfg['exploration_alpha'])}"
            )
    elif scheduler in {"swts", "dsts"}:
        # Requested as posterior_variance; mapped to implemented likelihood_variance.
        if "posterior_variance" in cfg:
            overrides.append(
                f"algorithm.scheduler_kwargs.likelihood_variance={float(cfg['posterior_variance'])}"
            )
        if "prior_variance" in cfg:
            overrides.append(
                f"algorithm.scheduler_kwargs.prior_variance={float(cfg['prior_variance'])}"
            )
    elif scheduler in {"dslinucb_r", "dslinucb_c"}:
        if "exploration_beta" in cfg:
            overrides.append(
                f"algorithm.scheduler_kwargs.exploration_beta={float(cfg['exploration_beta'])}"
            )
    elif scheduler in {"dslints_r", "dslints_c"}:
        if "noise_beta" in cfg:
            overrides.append(
                f"algorithm.scheduler_kwargs.noise_beta={float(cfg['noise_beta'])}"
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
    launcher_cuda = str(os.environ.get("CUDA_VISIBLE_DEVICES", "")).strip()
    cfg_cuda = str(_cfg_get(cfg, "cuda_visible_devices", "")).strip()
    if launcher_cuda:
        if cfg_cuda and cfg_cuda != launcher_cuda:
            print(
                "[wandb_sweep_bandit] ignoring sweep cuda_visible_devices="
                f"{cfg_cuda}; using launcher CUDA_VISIBLE_DEVICES={launcher_cuda}",
                flush=True,
            )
        env["CUDA_VISIBLE_DEVICES"] = launcher_cuda
    elif cfg_cuda:
        env["CUDA_VISIBLE_DEVICES"] = cfg_cuda

    print("[wandb_sweep_bandit] command:", " ".join(cmd), flush=True)
    result = subprocess.run(cmd, cwd=str(ROOT), env=env, check=False)
    return int(result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
