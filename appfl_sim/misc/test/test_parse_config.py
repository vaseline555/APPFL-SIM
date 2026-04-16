from pathlib import Path

import pytest

import appfl_sim.runner as runner


def test_parse_config_uses_explicit_config_when_default_missing(tmp_path, monkeypatch):
    config_path = tmp_path / "custom.yaml"
    config_path.write_text(
        "\n".join(
            [
                "experiment:",
                "  backend: serial",
                "model:",
                "  name: dummy",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    missing_default = tmp_path / "missing-default.yaml"
    monkeypatch.setattr(runner, "_default_config_path", lambda: Path(missing_default))

    backend, cfg = runner.parse_config(["--config", str(config_path)])

    assert backend == "serial"
    assert str(cfg.experiment.backend) == "serial"


def test_parse_config_still_requires_default_without_explicit_config(tmp_path, monkeypatch):
    missing_default = tmp_path / "missing-default.yaml"
    monkeypatch.setattr(runner, "_default_config_path", lambda: Path(missing_default))

    with pytest.raises(FileNotFoundError, match="Default config not found"):
        runner.parse_config([])
