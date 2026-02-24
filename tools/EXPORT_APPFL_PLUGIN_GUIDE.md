# Export APPFL Plugin Guide (Friendly Version)

This guide explains how to use:
- `tools/export_appfl_plugin.py`

It helps you move your algorithm module(s) from `appfl_sim` into an APPFL fork in a plug-and-play style.

Naming rule:
- Exported class names are validated against `algorithm`:
  - `<PascalCase(algorithm)>Aggregator`
  - `<PascalCase(algorithm)>Scheduler`
  - `<PascalCase(algorithm)>Trainer`

---

## 1) What this tool does

Given your aggregator/trainer/scheduler files, it will:

1. Validate class names exist.
2. Copy modules into APPFL-style folders:
- `src/appfl/algorithm/aggregator/`
- `src/appfl/algorithm/trainer/`
- `src/appfl/algorithm/scheduler/`
3. Rewrite imports from `appfl_sim` -> `appfl` in copied files.
4. Generate a config template YAML for your algorithm.
5. Output a machine-readable manifest JSON.
6. Auto-add `src/appfl/metrics/*` support files when your exported trainer imports metrics manager APIs.

---

## 2) Two modes

### A) Artifact mode (safe, recommended first)

Does not modify your APPFL fork directly.

```bash
cd /Users/vaseline555/Desktop/workspace/APPFL_SIM/APPFL-SIM
PYTHONPATH=. .venv/bin/python tools/export_appfl_plugin.py \
  --algorithm myalgo \
  --aggregator-source appfl_sim/algorithm/aggregator/myalgo_aggregator.py \
  --aggregator-class MyAlgoAggregator \
  --trainer-source appfl_sim/algorithm/trainer/myalgo_trainer.py \
  --trainer-class MyAlgoTrainer \
  --scheduler-source appfl_sim/algorithm/scheduler/myalgo_scheduler.py \
  --scheduler-class MyAlgoScheduler \
  --output-dir build/appfl_plugin_myalgo \
  --check-appfl-root /path/to/APPFL-main
```

### B) Direct mode (writes into APPFL fork)

```bash
python tools/export_appfl_plugin.py \
  --algorithm myalgo \
  --aggregator-source appfl_sim/algorithm/aggregator/myalgo_aggregator.py \
  --aggregator-class MyAlgoAggregator \
  --trainer-source appfl_sim/algorithm/trainer/myalgo_trainer.py \
  --trainer-class MyAlgoTrainer \
  --scheduler-source appfl_sim/algorithm/scheduler/myalgo_scheduler.py \
  --scheduler-class MyAlgoScheduler \
  --appfl-root /path/to/your/APPFL-fork
```

---

## 3) Expected output (example)

When it runs successfully, you should see JSON like:

```json
{
  "algorithm": "demo_algo",
  "mode": "direct",
  "output_root": ".../APPFL-main",
  "copied": [
    {
      "kind": "aggregator",
      "class": "DemoAlgoAggregator",
      "dst": ".../src/appfl/algorithm/aggregator/demo_fedavg_aggregator.py"
    }
  ],
  "metrics_support_files": [
    ".../src/appfl/metrics/__init__.py",
    ".../src/appfl/metrics/manager.py"
  ],
  "config_template": ".../config/algorithms/demo_algo.yaml"
}
```

This means files were generated/copied correctly.
If `unresolved_appfl_imports` is empty, import compatibility check passed.

---

## 4) Where files should go in your APPFL fork

If you use Artifact mode, move files as follows:

1. Aggregator module
- From: <`<APPFL PATH>/src/appfl/algorithm/aggregator/*.py`
- To: `<APPFL_FORK>/src/appfl/algorithm/aggregator/`

2. Trainer module
- From: <`<APPFL PATH>/src/appfl/algorithm/trainer/*.py`
- To: `<APPFL_FORK>/src/appfl/algorithm/trainer/`

3. Scheduler module
- From: <`<APPFL PATH>/src/appfl/algorithm/scheduler/*.py`
- To: `<APPFL_FORK>/src/appfl/algorithm/scheduler/`

4. Metrics support module (only when generated)
- From: <`<APPFL PATH>/src/appfl/metrics/*.py`
- To: `<APPFL_FORK>/src/appfl/metrics/`

5. `__init__.py` patches
- Apply snippets in: <`<APPFL PATH>/patches/*.txt`
- Target files:
  - `<APPFL_FORK>/src/appfl/algorithm/aggregator/__init__.py`
  - `<APPFL_FORK>/src/appfl/algorithm/trainer/__init__.py`
  - `<APPFL_FORK>/src/appfl/algorithm/scheduler/__init__.py`

6. Config template
- Generated at: <`<APPFL PATH>/config/algorithms/<algo>.yaml`
- Suggested placement in APPFL fork:
  - `<APPFL_FORK>/examples/resources/configs/` (for runnable examples), or
  - your own experiment config folder.

---

## 5) Real trial status (validated)

Validation in this repository:
- Exported `SwtsAggregator + SwtsTrainer + SwtsScheduler` artifact.
- Ran compatibility audit against `/Users/vaseline555/Desktop/workspace/APPFL_SIM/APPFL-main`.
- `unresolved_appfl_imports` came back empty.
- When trainer used metrics manager imports, exporter generated `src/appfl/metrics/*` support files automatically.

---

## 6) Quick verification commands in your fork

```bash
# 1) confirm symbols are exported
rg "MyAlgoAggregator|MyAlgoTrainer|MyAlgoScheduler" \
  src/appfl/algorithm/aggregator/__init__.py \
  src/appfl/algorithm/trainer/__init__.py \
  src/appfl/algorithm/scheduler/__init__.py

# 2) ensure no appfl_sim imports remain
rg "appfl_sim" src/appfl/algorithm/aggregator src/appfl/algorithm/trainer src/appfl/algorithm/scheduler
```

---

## 7) Common mistakes

- Wrong class name in CLI (`--aggregator-class` etc.).
- Using a source file that does not define the class.
- Forgetting to patch `__init__.py` in artifact mode.
- Testing in an APPFL environment missing optional deps used by your modules.

If you are unsure, start with Artifact mode, verify, then move to Direct mode.
