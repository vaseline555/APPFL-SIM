# APPFL-SIM
> PoC Here, Port Later.

`APPFL-SIM` is a simulation-focused spinoff of [APPFL](https://appfl.ai/en/latest/) for lightweight federated learning PoC research.

## Supported features

- APPFL-style algorithm skeleton (`aggregator`, `scheduler`, `trainer`).
- Lean simulation runtime (`agent`, `loaders`, `datasets`, `models`, `metrics`).
- Serial and distributed backends via `torch.distributed` (`serial`, `nccl`, `gloo`).
- Configuration-first algorithm wiring and reproducible experiment logging.

## Algorithm naming convention

`algorithm.name` resolves by convention unless explicit component overrides are set:

- `<PascalCase(name)>Aggregator`
- `<PascalCase(name)>Scheduler`
- `<PascalCase(name)>Trainer`

Examples:

- `fedavg` -> `FedavgAggregator`, `FedavgScheduler`, `FedavgTrainer`
- `fedprox` -> `FedproxAggregator`, `FedproxScheduler`, `FedproxTrainer`
- `fednova` -> `FednovaAggregator`, `FednovaScheduler`, `FednovaTrainer`
- `scaffold` -> `ScaffoldAggregator`, `FedavgScheduler`, `ScaffoldTrainer` (via explicit scheduler override in configs; `ScaffoldScheduler` remains a compatibility alias)
- `fedadam` -> `FedadamAggregator`, `FedadamScheduler`, `FedadamTrainer`
- `dsucb` -> `DsucbAggregator`, `DsucbScheduler`, `DsucbTrainer`
- `gale_avg` -> explicit override to `FedavgAggregator` plus adaptive GALE scheduler/trainer components
- `gale_prox` -> explicit override to `FedproxAggregator` plus adaptive GALE scheduler/trainer components

New algorithms should provide all three classes, even when scheduler/trainer just inherit defaults.

## Install

```bash
pip install -e .
```

## Run

```bash
python -m appfl_sim.runner \
  --config appfl_sim/config/examples/split/mnist_iid.yaml
```

Use CLI overrides for quick smoke runs:

```bash
python -m appfl_sim.runner \
  --config appfl_sim/config/examples/mnist_quickstart.yaml \
  experiment.backend=serial \
  experiment.device=cpu \
  experiment.server_device=cpu \
  train.num_rounds=1 train.num_clients=2 train.num_sampled_clients=2
```

## Configuration references

- Main guide: `CONFIG_GUIDES.md`

## Config examples

- Base template: `appfl_sim/config/examples/simulation.yaml`
- Split examples: `appfl_sim/config/examples/split/*.yaml`
- Logging examples: `appfl_sim/config/examples/logging/*.yaml`
- Metric examples: `appfl_sim/config/examples/metrics/*.yaml`
- Algorithm configs: `appfl_sim/config/algorithms/*.yaml`
