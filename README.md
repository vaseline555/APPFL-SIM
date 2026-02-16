# appfl[sim]

`appfl[sim]` is a simulation-focused spinoff of APPFL for lightweight federated learning PoC research.

## Design goals

- APPFL-style backbone (aggregator/trainer/communicator abstractions).
- Lightweight simulation hierarchy (`algorithm`, `agent`, `loaders`, `models`, `datasets`).
- MPI-backed synchronous simulation with one server rank and multiple client ranks.
- PoC support for single-node and multi-node multi-GPU execution.

## Install

```bash
cd source
pip install -e '.[sim]'
```

## Run

Serial:

```bash
PYTHONPATH=. .venv/bin/python -m appfl_sim.runner --config appfl_sim/config/examples/split/mnist_iid.yaml
```

MPI:

```bash
PYTHONPATH=. mpiexec -n 4 .venv/bin/python -m appfl_sim.runner \
  --config appfl_sim/config/examples/split/mnist_iid.yaml \
  backend=mpi
```

- Rank `0`: server
- Rank `1..N-1`: client workers (each rank can simulate multiple clients)

MPI PoC knobs for multi-node/multi-GPU:

- `mpi_dataset_download_mode=rank0|local_rank0|all|none`
  - `rank0`: only global rank 0 downloads, then others read with `download=false` (shared filesystem).
  - `local_rank0`: one rank per node downloads, then same-node peers read with `download=false`.
  - `all`: every rank downloads.
  - `none`: no rank downloads (data must already exist).
- `mpi_use_local_rank_device=true|false`
  - when `true`, client GPU mapping uses local rank (recommended for multi-node).
- `mpi_log_rank_mapping=true|false`
  - when `true`, each worker prints `(rank, local_rank, device, num_local_clients)`.

Example (multi-node style launch):

```bash
PYTHONPATH=. mpiexec -n 8 .venv/bin/python -m appfl_sim.runner \
  --config appfl_sim/config/examples/split/mnist_iid.yaml \
  backend=mpi device=cuda server_device=cpu \
  mpi_dataset_download_mode=local_rank0 \
  mpi_use_local_rank_device=true
```

Custom dataset loader example:

```bash
PYTHONPATH=. .venv/bin/python -m appfl_sim.runner \
  --backend serial \
  --dataset MyDataset \
  --dataset-loader custom \
  --custom-dataset-loader mypkg.data:build_dataset \
  --custom-dataset-kwargs '{"root": "./data"}'
```

Custom dataset path example:

```bash
PYTHONPATH=. .venv/bin/python -m appfl_sim.runner \
  --backend serial \
  --dataset MyDataset \
  --dataset-loader custom \
  --custom-dataset-path /path/to/custom_artifact_dir
```

Custom loader contract:

- Return `(split_map, client_datasets, server_dataset, args)`, or
- Return legacy `(split_map, client_datasets, args)`, or
- Return dict with keys `split_map` and `client_datasets` (optional: `server_dataset`, `args`).
- Or provide local artifacts under `custom_dataset_path` (`train.pt/.npz`, optional `test.pt/.npz`).

Default simulation config lives at:

- `appfl_sim/config/examples/simulation.yaml`
- Split examples: `appfl_sim/config/examples/split/*.yaml`
- Logging examples: `appfl_sim/config/examples/logging/*.yaml`
- Algorithm placeholders: `appfl_sim/config/algorithms/*.yaml`
- Evaluation-focused examples: `appfl_sim/config/algorithms/evaluation/*.yaml`
- HF template: `appfl_sim/config/examples/external_datasets/hf/template.yaml`
- timm template: `appfl_sim/config/examples/external_datasets/timm/template.yaml`

Model backend control:

- `model.source=appfl` for local models in `appfl_sim/models`
- `model.source=timm` for timm models (`model.name` must be exact timm model name)
- `model.source=hf` for HuggingFace models (`model.name` must be exact model card id)
- `model.source=auto` resolves `appfl -> timm -> hf`

To avoid config sprawl, keep one template per backend and override model/dataset at CLI:

```bash
PYTHONPATH=. .venv/bin/python -m appfl_sim.runner \
  --config appfl_sim/config/examples/external_datasets/hf/template.yaml \
  model.name=bert-base-uncased \
  experiment_name=hf-bert-sanity
```

HuggingFace API note:

- Public models do not require an HF API token.
- A token is only needed for private/gated repositories.

Logging backend control:

- `logging_backend=file` (default file logger)
- `logging_backend=tensorboard` (writes to `log_dir/project_name`)
- `logging_backend=wandb` (uses `project_name` as WandB project)
- `experiment_name` is used as run name.
- WandB online mode requires prior CLI authentication:
  `wandb login` (or `WANDB_API_KEY`).

Evaluation control:

- `enable_global_eval=true|false`:
  Global evaluation on `server_dataset` (runs only when a separate server eval split exists).
- `enable_federated_eval=true|false`:
  Federated evaluation across client holdout sets.
- `federated_eval_scheme=holdout_dataset|holdout_client`:
  - `holdout_dataset`:
    selected training clients are evaluated each round;
    all training clients are evaluated every `federated_eval_every` rounds and final round.
  - `holdout_client`:
    in-client (training pool) evaluation each round;
    every `federated_eval_every` rounds and final round, both in-client and out-client
    (holdout pool) are evaluated and logged independently.
- `holdout_eval_num_clients` / `holdout_eval_client_ratio`:
  size of out-client holdout pool for `federated_eval_scheme=holdout_client`.

Built-in dataset loader modes:

- `auto` (routes by dataset name)
- `custom` (user callable/path-based parser)
- `external` (external data sources: `hf`, `timm`)
- `torchvision`, `torchtext`, `torchaudio`, `medmnist`
- `flamby` (adapted from APPFL example loader)
- `leaf` (adapted LEAF preprocessed-json loader)
- `tff` (from `tff.simulation.datasets`)

FLamby note:

- Supported (AAggFF-aligned) datasets: `HEART`, `ISIC2019`, `IXITINY`.
- Set `flamby_data_terms_accepted=true` after accepting the source dataset terms.

Internal parser modules live under `appfl_sim/datasets/`:

- `torchvisionparser.py`
- `torchtextparser.py`
- `torchaudioparser.py`
- `medmnistparser.py`
- `flambyparser.py`
- `leafparser.py`
- `tffparser.py`

Researcher extension docs:

- Implementation playbook: `ALGORITHM_PLAYBOOK.md`
- APPFL plugin exporter: `tools/export_appfl_plugin.py`
- Exporter usage guide: `tools/EXPORT_APPFL_PLUGIN_GUIDE.md`

## Notes

- This package prioritizes simulation/PoC speed over production features.
