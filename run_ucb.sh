# CIFAR-10 Dirichlet Non-IID
for gamma in 0.67 0.80 0.90 0.95; do
  CUDA_VISIBLE_DEVICES=0,1,2,3 python -m appfl_sim.runner \
    --config appfl_sim/config/adaptive_local_steps/cifar10_diri/dsucb.yaml \
      logging.configs.wandb_entity=vaseline555 experiment.seed=42 \
        algorithm.scheduler_kwargs.discount_gamma=$gamma \
          algorithm.scheduler_kwargs.exploration_alpha=0.1 \
            experiment.name=GenFL_Discounted2 \
              logging.name="cifar10_diri_dsucb_cost_${gamma}_01" &

  CUDA_VISIBLE_DEVICES=0,1,2,3 python -m appfl_sim.runner \
    --config appfl_sim/config/adaptive_local_steps/cifar10_diri/dsucb.yaml \
      logging.configs.wandb_entity=vaseline555 experiment.seed=42 \
        algorithm.scheduler_kwargs.discount_gamma=$gamma \
          algorithm.scheduler_kwargs.exploration_alpha=0.01 \
            experiment.name=GenFL_Discounted2 \
              logging.name="cifar10_diri_dsucb_cost_${gamma}_001" &

  CUDA_VISIBLE_DEVICES=0,1,2,3 python -m appfl_sim.runner \
    --config appfl_sim/config/adaptive_local_steps/cifar10_diri/dsucb.yaml \
      logging.configs.wandb_entity=vaseline555 experiment.seed=42 \
        algorithm.scheduler_kwargs.discount_gamma=$gamma \
          algorithm.scheduler_kwargs.exploration_alpha=0.001 \
            experiment.name=GenFL_Discounted2 \
              logging.name="cifar10_diri_dsucb_cost_${gamma}_0001"
  wait
done