# Researcher Playbook: Add Your Own FL Algorithm

For the first-time contributors.

Goal:
- Implement your FL idea in `APPFL-SIM` for simulation.
- Export your algorithm into `APPFL` in plug-and-play manner.

---

## Device and Distributed Backends
- `backend=serial`: single-process baseline (recommended for single-node/single-GPU small runs).
- `backend=nccl`: multi-process multi-GPU runtime via `torch.distributed` + NCCL.
- `backend=gloo`: CPU-oriented multi-process runtime via `torch.distributed` + Gloo.
- `device=cuda` is recommended for `nccl` to map ranks across available GPUs.
- `device=cpu` is recommended for `gloo`.
- GPU subset control for `nccl`: set `CUDA_VISIBLE_DEVICES` before launch.  
  Example: `CUDA_VISIBLE_DEVICES=1,3 python -m appfl_sim.runner --config appfl_sim/config/examples/backend/nccl.yaml`


## 0) Before you start

Sanity run first:

```bash
python -m appfl_sim.runner --config appfl_sim/config/examples/split/mnist_iid.yaml
```

If this works, your environment is ready.

---

## 1) What you actually need to implement

Most new FL methods need these pieces:

1. Aggregator (required)
- File: `appfl_sim/algorithm/aggregator/<algo>_aggregator.py`
- Inherit: `BaseAggregator`
- Implement:
  - `aggregate(...)`
  - `get_parameters(...)`

2. Trainer (optional, but common)
- Only needed if your local client update is custom.
- File: `appfl_sim/algorithm/trainer/<algo>_trainer.py`
- Inherit: `BaseTrainer`
- Implement:
  - `train(...)`
  - `get_parameters(...)`

3. Scheduler (optional)
- Needed for async/custom scheduling.
- File: `appfl_sim/algorithm/scheduler/<algo>_scheduler.py`
- Inherit: `BaseScheduler`
- Implement:
  - `schedule(...)`
  - `get_num_global_epochs(...)`

4. Config example (required)
- Add: `appfl_sim/config/algorithms/<algo>.yaml`

Important current note:
- You do **not** typically need to edit `runner.py` for new algorithms.
- Add your classes under `appfl_sim/algorithm/...`, then set config keys:
  - `algorithm=<algo_label>` (metadata + default class-name inference)
  - optional explicit overrides: `aggregator`, `scheduler`, `trainer`

---

## 2) Start with the smallest possible implementation

### Step A: create your aggregator file

Copy a simple one as a starting point:
- `appfl_sim/algorithm/aggregator/fedavg_aggregator.py`

Then rename class and edit logic.

### Step B: add config placeholder

Create:
- `appfl_sim/config/algorithms/<algo>.yaml`

Minimum idea:

```yaml
algorithm: <algo>
aggregator: <YourAggregatorClass>
scheduler: SyncScheduler
trainer: VanillaTrainer
num_clients: 3
num_rounds: 2
num_sampled_clients: 3
```

### Step C: run tiny experiment

Use MNIST + 3 clients + 1-2 rounds first. Do not start with a complex dataset.

---

## 3) Data loading contract (custom datasets)

If you have your own dataset code, use:
- `dataset_loader=custom`
- `custom_dataset_loader=package.module:function`

Your function should return:

```python
split_map, client_datasets, server_dataset, args
```

Where:
- `split_map`: client split info (dict-like)
- `client_datasets`: list of `(train_dataset, test_dataset)` or `(train_dataset, val_dataset, test_dataset)` for each client
- `server_dataset`: global test/eval dataset (or `None`)
- `args`: metadata namespace, including at least:
  - `num_clients`
  - `num_classes`
  - `input_shape`

Tip:
- If your data is image/audio-like, set `need_embedding=False`.
- If your data is token/text-like, set `need_embedding=True`, `seq_len`, and `num_embeddings`.

---

## 4) Metrics: easiest path first

Two options:

1. Easy (recommended for first prototype)
- Log scalar values in train/eval loops:
  - `loss`
  - `accuracy`
  - `num_examples`

2. Advanced (registry style)
- Add class in `appfl_sim/metrics/metricszoo.py`
- Inherit `BaseMetric`
- Implement:
  - `collect(pred, true)`
  - `summarize()`

Use advanced mode only after your base algorithm is stable.

---

## 5) Common mistakes checklist

Before debugging deeply, check these first:

- Your class name matches config name exactly.
- Dataset returns `(train_dataset, test_dataset)` per client.
- `args.num_classes` is correct.
- `args.input_shape` matches model input.
- `model.source` and `model.name` are exact names (no aliases).
- Start with `backend=serial` before MPI.

---

## 6) Export to APPFL-main (production conversion)

Use exporter script:
- `tools/export_appfl_plugin.py`

### Safe mode (recommended)
Generates an artifact folder without touching `APPFL-main`:

```bash
PYTHONPATH=. .venv/bin/python tools/export_appfl_plugin.py \
  --algorithm myalgo \
  --aggregator-source appfl_sim/algorithm/aggregator/myalgo_aggregator.py \
  --aggregator-class MyAlgoAggregator \
  --trainer-source appfl_sim/algorithm/trainer/myalgo_trainer.py \
  --trainer-class MyAlgoTrainer \
  --scheduler-source appfl_sim/algorithm/scheduler/myalgo_scheduler.py \
  --scheduler-class MyAlgoScheduler \
  --output-dir build/appfl_plugin_myalgo
```

### Direct mode (writes into APPFL-main)

```bash
PYTHONPATH=. .venv/bin/python tools/export_appfl_plugin.py \
  --algorithm myalgo \
  --aggregator-source appfl_sim/algorithm/aggregator/myalgo_aggregator.py \
  --aggregator-class MyAlgoAggregator \
  --appfl-root ../APPFL-main
```

Exporter output includes:
- APPFL-style module files under `src/appfl/algorithm/...`
- Auto-vendored `src/appfl/metrics/*` when exported trainer depends on `MetricsManager`.
- Config template under `config/algorithms/<algo>.yaml`
- Patch/install instructions (artifact mode)
- Optional compatibility audit with APPFL tree:
  `--check-appfl-root /path/to/APPFL-main`
