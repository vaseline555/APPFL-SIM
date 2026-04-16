#!/bin/bash -l
#PBS -N GALE
#PBS -A PPFL_FM
#PBS -q preemptable
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l walltime=0:10:00
#PBS -l filesystems=home:eagle
#PBS -r y
#PBS -k doe
#PBS -j oe

set -euo pipefail

cd "${PBS_O_WORKDIR:-$PWD}"

echo "Client-wise contextual GALE-FedAvg is not supported for cross-device TinyImageNet."
echo "Reason: DslinucbCScheduler requires full participation, but this setup samples 5 of 500 clients per round."
echo "Use the round-wise scripts in this directory instead."
exit 1
