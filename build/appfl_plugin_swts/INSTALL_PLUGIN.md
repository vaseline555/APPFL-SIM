# Install Plugin: swts

1. Copy generated files under `src/appfl/algorithm/*` into APPFL-main at the same paths.
2. For each component, patch APPFL `__init__.py` using snippets under `patches/`.
3. Copy config template from `config/algorithms/`.

Components:
- aggregator: `SwtsAggregator` from `/Users/vaseline555/Desktop/workspace/APPFL_SIM/APPFL-SIM/appfl_sim/algorithm/aggregator/swts_aggregator.py`
- trainer: `SwtsTrainer` from `/Users/vaseline555/Desktop/workspace/APPFL_SIM/APPFL-SIM/appfl_sim/algorithm/trainer/swts_trainer.py`
- scheduler: `SwtsScheduler` from `/Users/vaseline555/Desktop/workspace/APPFL_SIM/APPFL-SIM/appfl_sim/algorithm/scheduler/swts_scheduler.py`
