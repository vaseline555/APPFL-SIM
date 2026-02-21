from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from omegaconf import DictConfig
from appfl_sim.logger import ServerAgentFileLogger
from appfl_sim.metrics import parse_metric_names
from appfl_sim.misc.config_utils import _cfg_bool, _cfg_get


try:
    from tqdm.auto import tqdm as _tqdm
except Exception:  # pragma: no cover
    _tqdm = None

def _new_progress(total: int, desc: str, enabled: bool):
    if not enabled or _tqdm is None or int(total) <= 0:
        return None
    label = f"appfl-sim: ✅[{str(desc)}]"
    return _tqdm(
        total=int(total),
        desc=label,
        leave=False,
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
    )

def _emit_logging_policy_message(
    policy: Dict[str, object],
    num_clients: int,
    logger: Optional[ServerAgentFileLogger] = None,
) -> None:
    requested = str(policy["requested_scheme"])
    effective = str(policy["effective_scheme"])
    basis_clients = int(policy.get("basis_clients", num_clients))
    total_clients = int(policy.get("total_clients", num_clients))
    forced_server_only = bool(policy.get("forced_server_only", False))

    def _info(msg: str) -> None:
        if logger is not None:
            logger.info(msg)
        else:
            print(msg)

    def _warn(msg: str) -> None:
        if logger is not None:
            logger.warning(msg)
        else:
            print(msg)

    if forced_server_only:
        _info(
            "Per-client file logging disabled because `num_sampled_clients` < `num_clients`. "
            "Using `server-only` logging for performance."
        )
        return

    if requested == "auto" and effective == "server_only":
        _info(
            "Client logging auto-switched to `server_only` for this run."
        )
        return
    if requested == "server_only":
        _info("Using `logging_scheme`=`server_only` (server-side metrics only).")
        return
    if requested == "both":
        _warn(
            f"Per-client logging is explicitly enabled with sampled_clients={basis_clients} "
            f"(total_clients={total_clients}). This may produce large I/O overhead."
        )

def _emit_client_state_policy_message(
    policy: Dict[str, object],
    logger: Optional[ServerAgentFileLogger] = None,
) -> None:
    stateful = bool(policy.get("stateful", False))
    mode = "stateful/persistent" if stateful else "stateless/sporadic"
    msg = f"{mode.title()} clients because `stateful={stateful}`"
    if logger is not None:
        logger.info(msg)
    else:
        print(msg)

def _emit_federated_eval_policy_message(
    config: DictConfig,
    train_client_count: int,
    holdout_client_count: int,
    logger: Optional[ServerAgentFileLogger] = None,
) -> None:
    if not _cfg_bool(config, "eval.enable_federated_eval", True):
        return
    ratio = float(_cfg_get(config, "eval.configs.client_ratio", 1.0))
    cadence = int(_cfg_get(config, "eval.every", 1))
    scheme = str(_cfg_get(config, "eval.configs.scheme", "dataset")).strip().lower()
    if ratio >= 1.0 and cadence > 0:
        return
    total_basis = int(train_client_count)
    if scheme == "client":
        total_basis = int(train_client_count) + int(holdout_client_count)
    msg = (
        "Federated eval policy: "
        f"interval={'final_only' if cadence <= 0 else cadence} "
        f"client_ratio={ratio:.4f} "
        f"basis_clients={total_basis}"
    )
    if logger is not None:
        logger.info(msg)
    else:
        print(msg)

def _warn_if_workers_pinned_to_single_device(
    config: DictConfig,
    world_size: int,
    logger: Optional[ServerAgentFileLogger] = None,
) -> None:
    if world_size <= 1:
        return
    dev = str(_cfg_get(config, "experiment.device", "cpu")).strip().lower()
    if not dev.startswith("cuda:"):
        return
    suffix = dev.split(":", 1)[1].strip()
    if not suffix.isdigit():
        return
    msg = (
        f"Device warning: `device={dev}` pins all ranks to the same GPU index. "
        "For multi-rank GPU spreading, use `device=cuda`."
    )
    if logger is not None:
        logger.warning(msg)
    else:
        print(msg)

def _log_round(
    config: DictConfig,
    round_idx: int,
    selected_count: int,
    total_train_clients: int,
    stats,
    weights,
    round_local_steps: Optional[int] = None,
    round_wall_time_sec: Optional[float] = None,
    global_gen_error: Optional[float] = None,
    global_eval_metrics: Optional[Dict[str, float]] = None,
    federated_eval_metrics: Optional[Dict[str, float]] = None,
    federated_eval_in_metrics: Optional[Dict[str, float]] = None,
    federated_eval_out_metrics: Optional[Dict[str, float]] = None,
    track_gen_rewards: bool = False,
    round_gen_reward: Optional[float] = None,
    cumulative_gen_reward: Optional[float] = None,
    logger: ServerAgentFileLogger | None = None,
    tracker=None,
):
    del weights

    def _entity_line(title: str, body: str) -> str:
        return f"  {title:<18} {body}"

    def _join_metric_parts(parts: List[str]) -> str:
        if not parts:
            return ""
        return " | ".join(f"{part:<24}" for part in parts).rstrip()

    eval_metric_order = parse_metric_names(_cfg_get(config, "eval.metrics", None))
    if not eval_metric_order:
        eval_metric_order = ["acc1"]

    def _pick_numeric(d: Dict[str, Any], candidates: List[str]) -> Optional[Tuple[str, float]]:
        for key in candidates:
            if key in d and isinstance(d[key], (int, float)):
                return key, float(d[key])
        return None

    def _collect_client_values(
        all_stats: Dict[int, Dict[str, Any]],
        candidates: List[str],
    ) -> List[float]:
        values: List[float] = []
        for row in all_stats.values():
            hit = _pick_numeric(row, candidates)
            if hit is None:
                continue
            _, value = hit
            values.append(float(value))
        return values

    round_metrics: Dict[str, object] = {
        "clients": {
            "selected": int(selected_count),
            "total": int(total_train_clients),
        }
    }
    lines = [
        "--- Round Summary ---",
        _entity_line(
            "Clients:",
            f"selected={selected_count}/{total_train_clients} "
            f"({(100.0 * float(selected_count) / float(max(1, total_train_clients))):.2f}%)",
        ),
    ]
    if round_local_steps is not None:
        round_metrics["policy"] = {"tau_t": int(round_local_steps)}
        lines.append(_entity_line("Policy:", f"tau_t={int(round_local_steps)}"))
    if isinstance(round_wall_time_sec, (int, float)):
        wall_sec = float(round_wall_time_sec)
        round_metrics["timing"] = {"round_wall_time_sec": wall_sec}
        lines.append(_entity_line("Round Time:", f"{wall_sec:.3f}s"))

    if stats:
        train_parts: List[str] = []
        training_metrics: Dict[str, Dict[str, float]] = {}

        def _append_train_field(label: str, candidates: List[str]) -> bool:
            vals = _collect_client_values(stats, candidates)
            if not vals:
                return False
            avg_value = float(np.mean(vals))
            std_value = float(np.std(vals))
            training_metrics[label] = {"avg": avg_value, "std": std_value}
            train_parts.append(f"{label}: {avg_value:.4f}/{std_value:.4f}")
            return True

        _append_train_field("loss", ["loss"])
        for metric_name in eval_metric_order:
            _append_train_field(metric_name, [f"metric_{metric_name}", metric_name])

        if train_parts:
            round_metrics["training"] = training_metrics
            lines.append(_entity_line("Training:", _join_metric_parts(train_parts)))

    def _append_local_eval_block(title: str, json_key: str, prefix: str) -> None:
        if not stats:
            return
        parts: List[str] = []
        section: Dict[str, Dict[str, float]] = {}

        def _append_field(label: str, candidates: List[str]) -> bool:
            vals = _collect_client_values(stats, candidates)
            if not vals:
                return False
            avg_value = float(np.mean(vals))
            std_value = float(np.std(vals))
            section[label] = {"avg": avg_value, "std": std_value}
            parts.append(f"{label}: {avg_value:.4f}/{std_value:.4f}")
            return True

        _append_field("loss", [f"{prefix}loss"])
        for metric_name in eval_metric_order:
            _append_field(
                metric_name,
                [f"{prefix}metric_{metric_name}", f"{prefix}{metric_name}"],
            )

        if parts:
            round_metrics[json_key] = section
            lines.append(_entity_line(f"{title}:", _join_metric_parts(parts)))

    def _append_eval_block(
        title: str,
        json_key: str,
        metrics: Optional[Dict[str, float]],
        with_client_std: bool = False,
    ) -> None:
        if metrics is None:
            return

        parts: List[str] = []
        section_metrics: Dict[str, object] = {}
        used_raw_keys: set[str] = set()

        def _append_eval_field(label: str, candidates: List[str]) -> bool:
            hit = _pick_numeric(metrics, candidates)
            if hit is None:
                return False
            raw_key, value = hit
            if raw_key in used_raw_keys:
                return False
            used_raw_keys.add(raw_key)
            if with_client_std:
                std_key = f"{raw_key}_std"
                if std_key in metrics and isinstance(metrics[std_key], (int, float)):
                    std_val = float(metrics[std_key])
                    section_metrics[label] = {
                        "avg": float(value),
                        "std": std_val,
                    }
                    parts.append(f"{label}: {float(value):.4f}/{std_val:.4f}")
                    return True
            section_metrics[label] = float(value)
            parts.append(f"{label}: {float(value):.4f}")
            return True

        _append_eval_field("loss", ["loss"])
        for metric_name in eval_metric_order:
            _append_eval_field(metric_name, [f"metric_{metric_name}", metric_name])

        if parts:
            lines.append(_entity_line(f"{title}:", _join_metric_parts(parts)))
            round_metrics[json_key] = section_metrics

    def _append_federated_extrema(metrics: Optional[Dict[str, float]]) -> None:
        if metrics is None:
            return
        extrema = {}
        for key, value in metrics.items():
            if (
                not key.endswith("_min")
                or not isinstance(value, (int, float))
                or f"{key[:-4]}_max" not in metrics
                or not isinstance(metrics[f"{key[:-4]}_max"], (int, float))
            ):
                continue
            base = key[:-4]
            display_base = base[7:] if base.startswith("metric_") else base
            extrema[base] = {
                "label": display_base,
                "min": float(value),
                "max": float(metrics[f"{base}_max"]),
            }
        if not extrema:
            return

        ordered_keys: List[str] = []
        if "loss" in extrema:
            ordered_keys.append("loss")
        for metric_name in eval_metric_order:
            for candidate in (f"metric_{metric_name}", metric_name):
                if candidate in extrema and candidate not in ordered_keys:
                    ordered_keys.append(candidate)
                    break
        if len(ordered_keys) <= 1 and "accuracy" in extrema and "accuracy" not in ordered_keys:
            ordered_keys.append("accuracy")
        if not ordered_keys:
            ordered_keys = sorted(extrema.keys())

        shown_keys = ordered_keys[:4]
        parts = [
            f"{extrema[name]['label']}[min,max]=[{extrema[name]['min']:.4f},{extrema[name]['max']:.4f}]"
            for name in shown_keys
        ]
        if len(ordered_keys) > len(shown_keys):
            parts.append(f"...(+{len(ordered_keys) - len(shown_keys)} more)")
        lines.append(
            _entity_line(
                "Federated Extrema:",
                _join_metric_parts(parts),
            )
        )
        round_metrics["fed_extrema"] = {
            extrema[name]["label"]: {
                "min": float(extrema[name]["min"]),
                "max": float(extrema[name]["max"]),
            }
            for name in ordered_keys
        }

    def _append_local_gen_error() -> None:
        if not stats:
            return
        vals = _collect_client_values(stats, ["local_gen_error"])
        if not vals:
            return
        avg_value = float(np.mean(vals))
        std_value = float(np.std(vals))
        round_metrics["local_gen_error"] = {"avg": avg_value, "std": std_value}
        lines.append(
            _entity_line(
                "Local Gen. Error:",
                _join_metric_parts([f"err.: {avg_value:.4f}/{std_value:.4f}"]),
            )
        )

    def _append_global_gen_error() -> None:
        if not isinstance(global_gen_error, (int, float)):
            return
        value = float(global_gen_error)
        round_metrics["global_gen_error"] = value
        lines.append(
            _entity_line(
                "Global Gen. Error:",
                _join_metric_parts([f"err.: {value:.4f}"]),
            )
        )

    def _append_gen_reward() -> None:
        if not track_gen_rewards:
            return
        section: Dict[str, float | None] = {}
        round_text = "n/a"
        cum_text = "n/a"
        if isinstance(round_gen_reward, (int, float)):
            section["round"] = float(round_gen_reward)
            round_text = f"{float(round_gen_reward):.4f}"
        else:
            section["round"] = None
        if isinstance(cumulative_gen_reward, (int, float)):
            section["cumulative"] = float(cumulative_gen_reward)
            cum_text = f"{float(cumulative_gen_reward):.4f}"
        else:
            section["cumulative"] = None
        round_metrics["gen_reward"] = section
        lines.append(
            _entity_line(
                "Gen. Reward:",
                _join_metric_parts([f"round: {round_text}", f"cumulative: {cum_text}"]),
            )
        )


    do_pre_val = _cfg_bool(config, "eval.do_pre_evaluation", True)
    do_post_val = _cfg_bool(config, "eval.do_post_evaluation", True)
    if do_pre_val:
        _append_local_eval_block("Local Pre-val.", "local_pre_val", "pre_val_")
    if do_post_val:
        _append_local_eval_block("Local Post-val.", "local_post_val", "post_val_")
    if do_pre_val:
        _append_local_eval_block("Local Pre-test.", "local_pre_test", "pre_test_")
    if do_post_val:
        _append_local_eval_block("Local Post-test.", "local_post_test", "post_test_")
    _append_local_gen_error()
    _append_global_gen_error()
    _append_gen_reward()

    _append_eval_block(
        "Global Eval.", "global_eval", global_eval_metrics, with_client_std=False
    )
    _append_eval_block(
        "Federated Eval.", "fed_eval", federated_eval_metrics, with_client_std=True
    )
    _append_eval_block(
        "Federated Eval(In).", "fed_eval_in", federated_eval_in_metrics, with_client_std=True
    )
    _append_eval_block(
        "Federated Eval(Out).",
        "fed_eval_out",
        federated_eval_out_metrics,
        with_client_std=True,
    )
    if federated_eval_metrics is not None:
        _append_federated_extrema(federated_eval_metrics)
    elif federated_eval_in_metrics is not None:
        _append_federated_extrema(federated_eval_in_metrics)

    log = "\n".join(lines)
    if logger is not None:
        logger.info(log, round_label=f"Round {round_idx:04d}")
    else:
        print(log)
    if tracker is not None:
        tracker.log_metrics(step=round_idx, metrics=round_metrics)

def _new_server_logger(config: DictConfig, mode: str, run_ts: str) -> ServerAgentFileLogger:
    run_dir = (
        Path(str(_cfg_get(config, "logging.path", "./logs")))
        / str(_cfg_get(config, "experiment.name", "appfl-sim"))
        / run_ts
    )
    mode_text = str(mode).strip().lower()
    file_name = "server.log"
    if "-rank" in mode_text:
        suffix = mode_text.split("-rank", 1)[1].strip("-_ ")
        if suffix.isdigit() and int(suffix) != 0:
            file_name = f"server-rank{suffix}.log"
    return ServerAgentFileLogger(
        file_dir=str(run_dir),
        file_name=file_name,
        experiment_id=str(_cfg_get(config, "experiment.name", "appfl-sim")),
    )

def _resolve_run_log_dir(config: DictConfig, run_id: str) -> str:
    return str(
        Path(str(_cfg_get(config, "logging.path", "./logs")))
        / str(_cfg_get(config, "experiment.name", "appfl-sim"))
        / str(run_id)
    )

def _resolve_run_timestamp(config: DictConfig, preset: Optional[str] = None) -> str:
    if config is None:
        seed_text = "0"
    else:
        seed_text = str(_cfg_get(config, "experiment.seed", 0))
    run_ts = str(preset if preset is not None else "").strip()
    if run_ts == "":
        run_ts = seed_text
    return run_ts

def _start_summary_lines(
    mode: str,
    config: DictConfig,
    num_clients: int,
    train_client_count: int,
    holdout_client_count: int,
    num_sampled_clients: int,
) -> str:
    sampled_pct = (
        100.0 * float(num_sampled_clients) / float(max(1, train_client_count))
    )
    lines = [
        f"Start {mode.upper()} simulation",
        f"  * Experiment: {_cfg_get(config, 'experiment.name', 'appfl-sim')}",
        f"  * Algorithm: {_cfg_get(config, 'algorithm.name', 'fedavg')}",
        f"  * Dataset: {_cfg_get(config, 'dataset.name', 'MNIST')}",
        f"  * Rounds: {_cfg_get(config, 'train.num_rounds', 20)}",
        f"  * Total Clients: {num_clients}",
        f"  * Sampled Clients/Round: {num_sampled_clients}/{train_client_count} ({sampled_pct:.2f}%)",
        f"  * Evaluation Scheme: {_cfg_get(config, 'eval.configs.scheme', 'dataset')}",
    ]
    if str(_cfg_get(config, "eval.configs.scheme", "dataset")) == "client":
        lines.append(f"  * Holdout Clients (evaluation): {holdout_client_count}")
    return "\n".join(lines)
