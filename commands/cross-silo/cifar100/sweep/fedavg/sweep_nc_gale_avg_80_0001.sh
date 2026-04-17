#!/bin/bash

#SBATCH -J GALE_FEDPROX
#SBATCH --output=GALE_FEDPROX_%j.out
#SBATCH --error=GALE_FEDPROX_%j.log
#SBATCH -A m5073 
#SBATCH -C gpu                    
#SBATCH -q regular                 
#SBATCH -N 1                   
#SBATCH -G 4                       
#SBATCH -t 3:00:00    

# Load environment
source .venv/bin/activate

# Run
# for lr_decay in 0.98 0.99 0.995; do
#   python -m appfl_sim.runner \
#     --config appfl_sim/config/cross-silo/cifar100/gale_avg.yaml \
#       logging.configs.wandb_entity=vaseline555 \
#         optimizer.lr=0.001 optimizer.lr_decay.gamma=$lr_decay \
#           algorithm.scheduler_kwargs.discount_gamma=0.80 \
#             algorithm.scheduler_kwargs.exploration_alpha=0.0001 \
#               algorithm.scheduler_kwargs.reward_scale=1 \
#                 experiment.name=GALE_CIFAR100_SWEEP logging.name="sweep_gale_avg_80_0.0001_${lr_decay}" &
# done
# wait

python -m appfl_sim.runner \
  --config appfl_sim/config/cross-silo/cifar100/gale_avg.yaml \
    logging.configs.wandb_entity=vaseline555 \
      optimizer.lr=0.001 optimizer.lr_decay.gamma=0.98 \
        algorithm.scheduler_kwargs.discount_gamma=0.80 \
          algorithm.scheduler_kwargs.exploration_alpha=0.0001 \
            algorithm.scheduler_kwargs.reward_scale=1 \
              experiment.name=GALE_CIFAR100_SWEEP logging.name="sweep_gale_avg_80_0.0001_0.98" & 

python -m appfl_sim.runner \
  --config appfl_sim/config/cross-silo/cifar100/gale_avg.yaml \
    logging.configs.wandb_entity=vaseline555 \
      optimizer.lr=0.001 optimizer.lr_decay.gamma=0.99 \
        algorithm.scheduler_kwargs.discount_gamma=0.80 \
          algorithm.scheduler_kwargs.exploration_alpha=0.0001 \
            algorithm.scheduler_kwargs.reward_scale=1 \
              experiment.name=GALE_CIFAR100_SWEEP logging.name="sweep_gale_avg_80_0.0001_0.99" & 

python -m appfl_sim.runner \
  --config appfl_sim/config/cross-silo/cifar100/gale_avg.yaml \
    logging.configs.wandb_entity=vaseline555 \
      optimizer.lr=0.001 optimizer.lr_decay.gamma=0.995 \
        algorithm.scheduler_kwargs.discount_gamma=0.80 \
          algorithm.scheduler_kwargs.exploration_alpha=0.0001 \
            algorithm.scheduler_kwargs.reward_scale=1 \
              experiment.name=GALE_CIFAR100_SWEEP logging.name="sweep_gale_avg_80_0.0001_0.995" &
