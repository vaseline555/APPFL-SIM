#!/bin/bash -l
#PBS -N GALE
#PBS -A PPFL_FM
#PBS -q preemptable
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l walltime=2:30:00
#PBS -l filesystems=home:eagle
#PBS -r y
#PBS -k doe
#PBS -j oe

# Polaris environment setup
set -euo pipefail

cd "${PBS_O_WORKDIR:-$PWD}"

module use /soft/modulefiles
module load conda
conda activate base

export http_proxy="http://proxy.alcf.anl.gov:3128"
export https_proxy="http://proxy.alcf.anl.gov:3128"
export ftp_proxy="http://proxy.alcf.anl.gov:3128"

# Generate uncorrelated seeds for each run
mapfile -t SEEDS < <(
python3 - <<'PY'
from numpy.random import SeedSequence
MASTER_SEED = 555
NUM_RUNS = 3

children = SeedSequence(MASTER_SEED).spawn(NUM_RUNS)
for child in children:
    print(int(child.generate_state(1, dtype="uint64")[0]))
PY
)

# Run
for i in "${!SEEDS[@]}"; do
  python -m appfl_sim.runner \
    --config appfl_sim/config/cross-silo/cifar100/fedavg.yaml \
      logging.configs.wandb_entity=vaseline555 train.local_epochs=8 \
        optimizer.lr=0.001 optimizer.lr_decay.gamma=0.95 \
          "experiment.seed=${SEEDS[$i]}" \
            experiment.name=GALE logging.name="main_cifar100_fedavg_8_${SEEDS[$i]}" &
done
wait