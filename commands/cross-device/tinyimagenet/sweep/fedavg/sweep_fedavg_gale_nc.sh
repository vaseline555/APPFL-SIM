#!/bin/bash -l

WANDB_ENTITY="${WANDB_ENTITY:-vaseline555}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-GALE_SWEEP}"
SWEEP_NAME="${SWEEP_NAME:-sweep_fedavg_gale_nc}"
RUN_CAP="${RUN_CAP:-60}"
GPU_IDS=(0 1 2 3)

if [[ "${WANDB_MODE}" != "online" ]]; then
  echo "W&B sweeps require WANDB_MODE=online." >&2
  exit 1
fi

tmpdir="$(mktemp -d)"
trap 'rm -rf "${tmpdir}"' EXIT
DOLLAR='$'

launcher_program="${tmpdir}/sweep_fedavg_gale_nc_launcher.py"
cat > "${launcher_program}" <<'PY'
import argparse
import os
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--exploration_alpha", required=True, type=float)
parser.add_argument("--discount_gamma", required=True, type=float)
parser.add_argument("--lr_decay_gamma", required=True, type=float)
args = parser.parse_args()

wandb_entity = os.environ.get("WANDB_ENTITY", "vaseline555")
wandb_mode = os.environ.get("WANDB_MODE", "online")
wandb_project = os.environ.get("WANDB_PROJECT", "GALE_SWEEP")

cmd = [
    sys.executable,
    "-m",
    "appfl_sim.runner",
    "--config",
    "appfl_sim/config/cross-device/tinyimagenet/gale_avg.yaml",
    "logging.backend=file",
    f"logging.configs.wandb_entity={wandb_entity}",
    f"logging.configs.wandb_mode={wandb_mode}",
    "logging.configs.wandb_group=gale_avg",
    "logging.configs.wandb_tags=cross-device,tinyimagenet,gale_avg,sweep,bayes",
    "optimizer.lr=0.001",
    "algorithm.scheduler_kwargs.reward_scale=10",
    f"algorithm.scheduler_kwargs.exploration_alpha={args.exploration_alpha}",
    f"algorithm.scheduler_kwargs.discount_gamma={args.discount_gamma}",
    f"optimizer.lr_decay.gamma={args.lr_decay_gamma}",
    f"experiment.name={wandb_project}",
    "logging.name="
    + f"sweep_fedavg_gale_nc_{args.discount_gamma:.2f}_{args.exploration_alpha:g}_{args.lr_decay_gamma:.2f}",
]

raise SystemExit(subprocess.call(cmd))
PY

sweep_config="${tmpdir}/sweep_fedavg_gale_nc.yaml"
cat > "${sweep_config}" <<EOF
program: ${launcher_program}
name: ${SWEEP_NAME}
method: bayes
metric:
  goal: minimize
  name: test/global_eval_loss
run_cap: ${RUN_CAP}
parameters:
  exploration_alpha:
    values: [0.0001, 0.0003, 0.0005, 0.001]
  discount_gamma:
    values: [0.80, 0.90, 0.95, 0.98, 0.99]
  lr_decay_gamma:
    values: [0.97, 0.98, 0.99]
command:
  - ${DOLLAR}{env}
  - ${DOLLAR}{interpreter}
  - ${DOLLAR}{program}
  - ${DOLLAR}{args_no_boolean_flags}
EOF

echo "Creating W&B sweep '${SWEEP_NAME}' in project '${WANDB_PROJECT}'"
sweep_output="$(wandb sweep --project "${WANDB_PROJECT}" --entity "${WANDB_ENTITY}" --name "${SWEEP_NAME}" "${sweep_config}" 2>&1)"
printf '%s\n' "${sweep_output}"

sweep_id="$(printf '%s\n' "${sweep_output}" | sed -n 's/.*Run sweep agent with: wandb agent \(.*\)$/\1/p' | tail -n 1)"
if [[ -z "${sweep_id}" ]]; then
  sweep_id="$(printf '%s\n' "${sweep_output}" | sed -n 's/.*Created sweep with ID: \([^[:space:]]*\)$/\1/p' | tail -n 1)"
fi

if [[ -z "${sweep_id}" ]]; then
  echo "Failed to parse W&B sweep ID from wandb sweep output." >&2
  exit 1
fi

echo "Starting sweep agents for: ${sweep_id}"
for gpu_id in "${GPU_IDS[@]}"; do
  echo "Launching agent on CUDA_VISIBLE_DEVICES=${gpu_id}"
  CUDA_VISIBLE_DEVICES="${gpu_id}" wandb agent --forward-signals "${sweep_id}" &
done

wait
