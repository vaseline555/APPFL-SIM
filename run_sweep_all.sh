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

SWEEP_DIR="${SWEEP_DIR:-appfl_sim/config/sweeps}"
AGENT_COUNT="${AGENT_COUNT:-4}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3}"

# Resolve SWEEP_DIR robustly:
# - absolute path: use as-is
# - common typo prefix "ROOT_DIR/": interpret relative to repo root
# - any other relative path: interpret relative to repo root
if [[ "${SWEEP_DIR}" == ROOT_DIR/* ]]; then
  SWEEP_DIR="${ROOT_DIR}/${SWEEP_DIR#ROOT_DIR/}"
elif [[ "${SWEEP_DIR}" != /* ]]; then
  SWEEP_DIR="${ROOT_DIR}/${SWEEP_DIR}"
fi

mapfile -t SWEEP_FILES < <(find "${SWEEP_DIR}" -maxdepth 1 -type f -name '*.yaml' | sort)
if [[ ${#SWEEP_FILES[@]} -eq 0 ]]; then
  echo "[run_sweep_all] ERROR: no sweep yaml files found in ${SWEEP_DIR}"
  exit 1
fi

echo "[run_sweep_all] Found ${#SWEEP_FILES[@]} sweep configs."
printf '[run_sweep_all] - %s\n' "${SWEEP_FILES[@]}"

for sweep_cfg in "${SWEEP_FILES[@]}"; do
  echo
  echo "[run_sweep_all] Starting sweep for ${sweep_cfg}"
  SWEEP_CONFIG="${sweep_cfg}" \
  AGENT_COUNT="${AGENT_COUNT}" \
  CUDA_DEVICES="${CUDA_DEVICES}" \
  bash "${ROOT_DIR}/run_sweep.sh"
  echo "[run_sweep_all] Completed ${sweep_cfg}"
done

echo "[run_sweep_all] All sweeps completed."
