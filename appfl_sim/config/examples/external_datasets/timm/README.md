# timm config template

Use `template.yaml` as a base and override exact timm model name via CLI:

```bash
PYTHONPATH=. .venv/bin/python -m appfl_sim.runner \
  --config appfl_sim/config/examples/external_datasets/timm/template.yaml \
  model.name=mobilevit_xxs \
  experiment_name=timm-mobilevit-test
```

This package intentionally keeps only a small template set.
Keep project-specific runs in your own config repo/folder and layer them with `--config` + CLI overrides.
