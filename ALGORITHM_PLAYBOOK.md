# Researcher Playbook: Add Your Own FL Algorithm

This guide is for first-time contributors (including entry-level graduate students).

Goal:
- Implement your FL idea in `appfl_sim` for simulation.
- Optionally export it into `APPFL-main` in plug-and-play format.

Estimated time:
- First working prototype: 1-2 hours.

---

## 0) Before you start

Run from:

```bash
cd /Users/vaseline555/Desktop/workspace/APPFL_SIM/source
```

Sanity run first:

```bash
PYTHONPATH=. .venv/bin/python -m appfl_sim.runner \
  --config appfl_sim/config/examples/split/mnist_iid.yaml \
  backend=serial num_clients=3 num_rounds=1
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
- The simulation runner currently uses a FedAvg-style loop by default.
- If your algorithm needs different server behavior, route `config.algorithm` to your custom server path in `appfl_sim/runner.py`.

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
num_clients: 3
num_rounds: 2
client_fraction: 1.0
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
- `client_datasets`: list of `(train_dataset, test_dataset)` for each client
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
- Config template under `config/algorithms/<algo>.yaml`
- Patch/install instructions (artifact mode)

---

## 7) Suggested learning path (first week)

1. Day 1: Run baseline MNIST simulation.
2. Day 2: Modify aggregator only.
3. Day 3: Add custom metric.
4. Day 4: Add custom trainer.
5. Day 5: Export plugin artifact and test insertion in APPFL-main copy.

Keep each step small and runnable.
