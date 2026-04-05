#!/bin/bash

#SBATCH -J GALE
#SBATCH --output=GALE_%j.out
#SBATCH --error=GALE_%j.log
#SBATCH -A m5073 
#SBATCH -C gpu                    
#SBATCH -q regular                 
#SBATCH -N 1                   
#SBATCH -G 2                       
#SBATCH -t 2:00:00    

LRS=(0.0017782 0.001 0.0005623 0.0003162)

# Submit one Slurm job per learning rate, then let the child job handle alpha sweep.
if [[ -z "${GALE_SWEEP_LR:-}" ]]; then
  for lr in "${LRS[@]}"; do
    sbatch --export=ALL,GALE_SWEEP_LR="$lr" "$0"
  done
  exit 0
fi

# Load environment
source .venv/bin/activate

# CIFAR-10 Dirichlet Non-IID
lr="${GALE_SWEEP_LR}"

for alpha in 0.0001 0.001 0.01 0.1 10; do
    python -m appfl_sim.runner \
      --config appfl_sim/config/adaptive_local_steps/cifar100_diri/dsucb.yaml \
        logging.configs.wandb_entity=vaseline555 \
          optimizer.lr=$lr optimizer.lr_decay.milestones="50,75" optimizer.lr_decay.gamma=0.5 \
            algorithm.scheduler_kwargs.discount_gamma=0.90 \
              algorithm.scheduler_kwargs.exploration_alpha=$alpha \
                experiment.name=GALE_NC_CIFAR100 logging.name="GALE_90_${alpha}_${lr}" &
done
wait
