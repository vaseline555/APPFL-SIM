#!/bin/bash

#SBATCH -J GenFL
#SBATCH --output=genfl_disc_%j.out
#SBATCH --error=genfl_disc_%j.log
#SBATCH -p main                
#SBATCH -N 1                   
#SBATCH --gres=gpu:h200:4      
#SBATCH -c 128              
#SBATCH --mem=128G           
#SBATCH -t 36:00:00                    

# Load environment
source .venv/bin/activate

# Loop seeds
for s in 0; do
  # CIFAR-10 Dirichlet Non-IID
  CUDA_VISIBLE_DEVICES=2,3 python -m appfl_sim.runner --config appfl_sim/config/adaptive_local_steps/cifar10_diri/dsucb.yaml logging.configs.wandb_entity=vaseline555 experiment.seed=$s &
  CUDA_VISIBLE_DEVICES=2,3 python -m appfl_sim.runner --config appfl_sim/config/adaptive_local_steps/cifar10_diri/dsts.yaml logging.configs.wandb_entity=vaseline555 experiment.seed=$s
  wait
  CUDA_VISIBLE_DEVICES=2,3 python -m appfl_sim.runner --config appfl_sim/config/adaptive_local_steps/cifar10_diri/dslints_r.yaml logging.configs.wandb_entity=vaseline555 experiment.seed=$s &
  CUDA_VISIBLE_DEVICES=2,3 python -m appfl_sim.runner --config appfl_sim/config/adaptive_local_steps/cifar10_diri/dslints_c.yaml logging.configs.wandb_entity=vaseline555 experiment.seed=$s
  wait
  CUDA_VISIBLE_DEVICES=2,3 python -m appfl_sim.runner --config appfl_sim/config/adaptive_local_steps/cifar10_diri/dslinucb_r.yaml logging.configs.wandb_entity=vaseline555 experiment.seed=$s &
  CUDA_VISIBLE_DEVICES=2,3 python -m appfl_sim.runner --config appfl_sim/config/adaptive_local_steps/cifar10_diri/dslinucb_c.yaml logging.configs.wandb_entity=vaseline555 experiment.seed=$s
  wait 
done