#!/bin/bash

#SBATCH -J APPFL-SIM
#SBATCH --output=genfl_%j.out
#SBATCH --error=genfl_%j.log
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
source .venv/bin/activate

# Usage examples:
#   sbatch run_sweep.sh
#   SWEEP_CONFIG=appfl_sim/config/sweeps/mnist_iid.yaml AGENT_COUNT=2 sbatch run_sweep.sh
#   SWEEP_PATH=entity/project/abcd1234 sbatch run_sweep.sh
SWEEP_CONFIG="${SWEEP_CONFIG:-appfl_sim/config/sweeps/cifar10_non_iid.yaml}"
AGENT_COUNT="${AGENT_COUNT:-1}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3}"
SWEEP_PATH="${SWEEP_PATH:-}"

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
echo "[run_sweep] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[run_sweep] SWEEP_CONFIG=${SWEEP_CONFIG}"
echo "[run_sweep] AGENT_COUNT=${AGENT_COUNT}"

if [[ -z "${SWEEP_PATH}" ]]; then
  echo "[run_sweep] Creating a new sweep..."
  SWEEP_CREATE_LOG="$(mktemp)"
  wandb sweep "${SWEEP_CONFIG}" 2>&1 | tee "${SWEEP_CREATE_LOG}"

  SWEEP_PATH="$(grep -Eo '[^[:space:]]+/[^[:space:]]+/[A-Za-z0-9]+' "${SWEEP_CREATE_LOG}" | tail -n 1 || true)"
  rm -f "${SWEEP_CREATE_LOG}"

  if [[ -z "${SWEEP_PATH}" ]]; then
    echo "[run_sweep] ERROR: failed to parse sweep path from wandb sweep output."
    echo "[run_sweep] Set SWEEP_PATH manually, e.g., entity/project/sweep_id"
    exit 1
  fi
fi

echo "[run_sweep] Using SWEEP_PATH=${SWEEP_PATH}"

pids=()
for ((i=1; i<=AGENT_COUNT; i++)); do
  echo "[run_sweep] Starting wandb agent ${i}/${AGENT_COUNT}"
  wandb agent "${SWEEP_PATH}" &
  pids+=("$!")
done

trap 'echo "[run_sweep] Caught signal, stopping agents..."; kill "${pids[@]}" 2>/dev/null || true' INT TERM
wait "${pids[@]}"
