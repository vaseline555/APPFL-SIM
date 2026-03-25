#!/bin/bash

#SBATCH -J APPFL-SIM-SWEEPS
#SBATCH --output=genfl_sweeps_%j.out
#SBATCH --error=genfl_sweeps_%j.log
#SBATCH -p main
#SBATCH -N 1
#SBATCH --gres=gpu:h200:4
#SBATCH -c 128
#SBATCH --mem=128G
#SBATCH -t 48:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SLURM_SUBMIT_DIR:-${SCRIPT_DIR}}"
if [[ ! -d "${ROOT_DIR}/appfl_sim" ]]; then
  ROOT_DIR="${SCRIPT_DIR}"
fi
cd "${ROOT_DIR}"

# Fixed 8-case launch plan:
# - CIFAR10 IID: GPU 0
# - CIFAR10 non-IID: GPU 1
# - MNIST IID: GPU 2
# - MNIST non-IID: GPU 3
CASES=(
  "appfl_sim/config/sweeps/dc/cifar10_iid_dsucb.yaml:0"
  "appfl_sim/config/sweeps/dc/cifar10_iid_dsts.yaml:0"
  "appfl_sim/config/sweeps/dc/cifar10_iid_dslinucb_c.yaml:0"
  "appfl_sim/config/sweeps/dc/cifar10_iid_dslinucb_r.yaml:0"
  "appfl_sim/config/sweeps/dc/cifar10_iid_dslints_c.yaml:0"
  "appfl_sim/config/sweeps/dc/cifar10_iid_dslints_r.yaml:0"
  "appfl_sim/config/sweeps/dc/cifar10_non_iid_dsucb.yaml:1"
  "appfl_sim/config/sweeps/dc/cifar10_non_iid_dsts.yaml:1"
  "appfl_sim/config/sweeps/dc/cifar10_non_iid_dslinucb_c.yaml:1"
  "appfl_sim/config/sweeps/dc/cifar10_non_iid_dslinucb_r.yaml:1"
  "appfl_sim/config/sweeps/dc/cifar10_non_iid_dslints_c.yaml:1"
  "appfl_sim/config/sweeps/dc/cifar10_non_iid_dslints_r.yaml:1"
  "appfl_sim/config/sweeps/dc/mnist_iid_dsucb.yaml:2"
  "appfl_sim/config/sweeps/dc/mnist_iid_dsts.yaml:2"
  "appfl_sim/config/sweeps/dc/mnist_iid_dslinucb_c.yaml:2"
  "appfl_sim/config/sweeps/dc/mnist_iid_dslinucb_r.yaml:2"
  "appfl_sim/config/sweeps/dc/mnist_iid_dslints_c.yaml:2"
  "appfl_sim/config/sweeps/dc/mnist_iid_dslints_r.yaml:2"
  "appfl_sim/config/sweeps/dc/mnist_non_iid_dsucb.yaml:3"
  "appfl_sim/config/sweeps/dc/mnist_non_iid_dsts.yaml:3"
  "appfl_sim/config/sweeps/dc/mnist_non_iid_dslinucb_c.yaml:3"
  "appfl_sim/config/sweeps/dc/mnist_non_iid_dslinucb_r.yaml:3"
  "appfl_sim/config/sweeps/dc/mnist_non_iid_dslints_c.yaml:3"
  "appfl_sim/config/sweeps/dc/mnist_non_iid_dslints_r.yaml:3"
)

pids=()
for item in "${CASES[@]}"; do
  sweep_cfg="${item%%:*}"
  gpu_id="${item##*:}"
  echo "[run_sweep_all] Launching ${sweep_cfg} on GPU ${gpu_id}"
  SWEEP_CONFIG="${sweep_cfg}" \
  CUDA_DEVICES="${gpu_id}" \
  AGENT_COUNT=1 \
  bash "${ROOT_DIR}/run_sweep.sh" &
  pids+=("$!")
done

trap 'echo "[run_sweep_all] Caught signal, stopping child launchers..."; kill "${pids[@]}" 2>/dev/null || true' INT TERM
wait "${pids[@]}"
echo "[run_sweep_all] All sweeps completed."
