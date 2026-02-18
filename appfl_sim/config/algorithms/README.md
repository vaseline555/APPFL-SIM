# Algorithm Placeholders

This folder provides placeholder simulation configs for algorithms implemented in
`appfl_sim/algorithm/{aggregator,scheduler,trainer}`.

Notes:
- These are intended as starter configs for users.
- Current simulator path uses APPFL-style `ServerAgent` and `ClientAgent`.
- New algorithm wiring is config-driven (`aggregator`, `scheduler`, `trainer`),
  so users do not need to edit `runner.py`.
- Evaluation-focused examples are under `appfl_sim/config/algorithms/evaluation/`.
