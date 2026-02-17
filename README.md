# APPFL-SIM
> PoC Here, Port Later.

`APPFL-SIM` is a simulation-focused spinoff of [`APPFL`](https://appfl.ai/en/latest/) for lightweight federated learning PoC research.

## Supported features

- `APPFL`-style backbone (aggregator/trainer/communicator abstractions).
- Lightweight simulation hierarchy (`algorithm`, `agent`, `loaders`, `models`, `datasets`).
- MPI-backed synchronous simulation with one server rank and multiple client ranks.
- PoC support for single-/multi-node multi-GPU execution.


## Install
### Install OpenMPI (if not exists - run `mpiexec` to check)
```bash
# Download
wget (tar.gz url retrieved from https://www.open-mpi.org/software/ompi/v5.0/)
tar -zxvf (downloaded tar.gz)
cd (unpakced directory)

# Configure with prefix and options (e.g., `--with-slurm`)
./configure --prefix=$HOME/openmpi-install --with-slurm
make -j4
make install

# Update PATH
export PATH=$HOME/openmpi-install/bin:$PATH
export LD_LIBRARY_PATH=$HOME/openmpi-install/lib:$LD_LIBRARY_PATH

# Refersh
source ~/.bashrc
```

### Install `APPFL-SIM`

```bash
pip install -e .
```


## Run

Serial:

```bash
# 3 clients
python -m appfl_sim.runner \
  --config appfl_sim/config/examples/split/mnist_iid.yaml 
```

MPI:

```bash
# 3 clients
python -m appfl_sim.runner \
  --config appfl_sim/config/examples/split/mnist_iid.yaml \
  backend=mpi 
```

- `backend=mpi` auto-launches MPI when not already inside an MPI job.
- Auto-launch first checks `PATH`; if not found, it also checks common prefixes
  such as `$HOME/openmpi-install/bin`, `$MPI_HOME/bin`, `$OPENMPI_HOME/bin`,
  `$OMPI_HOME/bin`, and `$I_MPI_ROOT/bin`.
- Worker ranks are auto-sized independently of logical `num_clients` (server rank is added automatically).
- Pin worker count via `mpi_num_workers=<N>` when needed.
- If you already run inside `mpiexec`/`mpirun` (e.g., scheduler scripts), auto-launch is skipped.

- Rank `0`: server
- Rank `1,...,n-1`: client workers (each rank can simulate multiple clients)

## Default setting

Default simulation config lives at:

- `appfl_sim/config/examples/simulation.yaml`
- Split examples: `appfl_sim/config/examples/split/*.yaml`
- Logging examples: `appfl_sim/config/examples/logging/*.yaml`
- Metrics examples: `appfl_sim/config/examples/metrics/*.yaml`
- Algorithm placeholders: `appfl_sim/config/algorithms/*.yaml`
- Evaluation-focused examples: `appfl_sim/config/algorithms/evaluation/*.yaml`
- HF template: `appfl_sim/config/examples/external_datasets/hf/template.yaml`
- timm template: `appfl_sim/config/examples/external_datasets/timm/template.yaml`

## MPI setting

MPI runtime knobs for multi-node/multi-GPU:

- `mpi_dataset_download_mode=rank0|local_rank0|all|none` (default: `rank0`)
  - `rank0`: only global rank `0` downloads, then other ranks load with `download=false`.
  - `local_rank0`: one rank per node downloads, then same-node peers load with `download=false`.
  - `all`: every rank downloads.
  - `none`: no rank downloads (dataset must already exist).
  - supported aliases: `rank0_then_barrier`, `root`, `local_rank0_then_barrier`, `node_leader`.
- `mpi_use_local_rank_device=true|false` (default: `true`)
  - when `true`, client device assignment uses local rank (recommended for multi-node GPU jobs).
- `mpi_log_rank_mapping=true|false` (default: `false`)
  - when `true`, each worker prints rank mapping details:
    `rank`, `local_rank`, `client_device`, `num_local_clients`.

Client logging scale knobs (applies to serial and MPI):

- `client_logging_scheme=auto|per_client|aggregated` (default: `auto`)
  - `auto`: per-client logging is enabled only when `num_clients <= per_client_logging_threshold`.
  - `per_client`: always log per client.
  - `aggregated`: disable per-client files and keep server-side metrics only.
- `per_client_logging_threshold` (default: `10`)
  - cutoff used by `client_logging_scheme=auto`.
- `per_client_logging_warning_threshold` (default: `50`)
  - warning threshold when `client_logging_scheme=per_client`.
- `aggregated_logging_scheme=server_only` (default: `server_only`)
  - currently supported aggregated mode.

Example (multi-node style launch):

```bash
# 3 clients
python -m appfl_sim.runner \
  --config appfl_sim/config/examples/split/mnist_iid.yaml \
  backend=mpi device=cuda server_device=cpu \
  mpi_dataset_download_mode=local_rank0 \
  mpi_use_local_rank_device=true
```

## Dataset setting

Built-in dataset loader modes:

- `auto` (routes by dataset name)
- `custom` (user callable/path-based parser)
- `external` (external data sources: `hf`, `timm`)
- `torchvision`, `torchtext`, `torchaudio`, `medmnist`
- `flamby` (adapted from APPFL example loader)
- `leaf` (adapted LEAF loader with optional auto download+preprocess)
- `tff` (from `tff.simulation.datasets`)

FLamby note:

- Supported datasets without copyright concerns: `HEART`, `ISIC2019`, `IXITINY`.
- Set `flamby_data_terms_accepted=true` after accepting the source dataset terms.

Internal parser modules live under `appfl_sim/datasets/`:

- `torchvisionparser.py`
- `torchtextparser.py`
- `torchaudioparser.py`
- `medmnistparser.py`
- `flambyparser.py`
- `leafparser.py`
- `tffparser.py`
- `customparser.py`
- `externalparser.py`

Fixed-pool dataset client selection knobs (`leaf` / `flamby` / `tff`):

- `infer_num_clients=true` (or `<prefix>_infer_num_clients=true`) to use full available client pool.
- `client_subsample_num` / `client_subsample_ratio` (or `<prefix>_...`) for pool subsampling.
- `client_subsample_mode=random|first|last` and `client_subsample_seed` for deterministic selection.

LEAF auto-bootstrap:

- If `train/`, `test/`, or `all_data/` is missing and `download=true`, `leafparser` will:
  download raw files using AAggFF LEAF URLs/MD5 rules, run dataset-specific preprocess,
  and create train/test client splits automatically.
- If data is missing and `download=false`, it raises a clear error.


Custom dataset loader example:

```bash
python -m appfl_sim.runner \
  --backend serial \
  --dataset MyDataset \
  --dataset-loader custom \
  --custom-dataset-loader mypkg.data:build_dataset \
  --custom-dataset-kwargs '{"root": "./data"}'
```

Custom dataset path example:

```bash
python -m appfl_sim.runner \
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

## Model setting

Model backend control:

- `model.source=appfl` for local models in `appfl_sim/models`
- `model.source=timm` for timm models (`model.name` must be exact timm model name)
- `model.source=hf` for HuggingFace models (`model.name` must be exact model card id)
- `model.source=auto` resolves `appfl -> timm -> hf`

To avoid config sprawl, keep one template per backend and override model/dataset at CLI:

```bash
python -m appfl_sim.runner \
  --config appfl_sim/config/examples/external_datasets/hf/template.yaml \
  model.name=bert-base-uncased \
  experiment_name=hf-bert-sanity
```

HuggingFace API note:

- Public models do not require an HF API token.
- A token is only needed for private/gated repositories.


## Logging setting

Logging backend control:

- `logging_backend=file` (default file logger)
- `logging_backend=tensorboard` (writes to `log_dir/project_name`)
- `logging_backend=wandb` (uses `project_name` as WandB project)
- `experiment_name` is used as run name.
- For `none|file|console`, round metrics are saved to
  `log_dir/exp_name/run_timestamp/metrics.json`.
  The file is a JSON array of records:
  `{"round": <int>, "metrics": {...}}`.
- WandB online mode requires prior CLI authentication:
  `wandb login` (or `WANDB_API_KEY`).


## Evaluation setting

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

## Client Initialization

Client initialization policy for scalability:

- `client_init_mode=auto|eager|on_demand` (default: `auto`)
- `client_init_on_demand_threshold=1000`:
  when `client_init_mode=auto`, on-demand mode is enabled if
  `num_clients > client_init_on_demand_threshold`.
- `client_processing_chunk_size=256`:
  chunk size used by on-demand mode while training/evaluating clients.

Guidance:

- Cross-silo / moderate client counts: `eager` is usually fine and lower-overhead.
- Cross-device / very large client counts: use `on_demand` (or `auto` with threshold).

## Metrics

Supported metric keys (from `appfl_sim/metrics/metricszoo.py`):

- `acc1`
- `acc5`
- `auroc`
- `auprc` (binary classification)
- `youdenj` (binary classification)
- `f1`
- `precision`
- `recall`
- `seqacc` (sequence token accuracy; ignores targets with `-1`)
- `mse`
- `rmse`
- `mae`
- `mape`
- `r2`
- `d2`
- `dice`
- `balacc`

## Documents

Researcher extension docs:

- Implementation playbook: `ALGORITHM_PLAYBOOK.md`
- APPFL plugin exporter: `tools/export_appfl_plugin.py`
- Exporter usage guide: `tools/EXPORT_APPFL_PLUGIN_GUIDE.md`
- Metrics to CSV converter: `tools/metrics_json_to_csv.py`
