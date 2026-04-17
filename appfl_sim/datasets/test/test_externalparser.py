import os

from appfl_sim.datasets.externalparser import (
    _hf_prepared_cache_available,
    _load_hf_dataset_with_cache_preference,
)


class _CaptureLogger:
    def __init__(self):
        self.infos = []
        self.warnings = []

    def info(self, msg, *args):
        self.infos.append(msg % args if args else msg)

    def warning(self, msg, *args):
        self.warnings.append(msg % args if args else msg)


def test_hf_prepared_cache_available_detects_complete_cache(tmp_path):
    cache_root = tmp_path / "zh-plus___tiny-imagenet" / "default" / "0.0.0" / "hash"
    cache_root.mkdir(parents=True)
    (cache_root / "dataset_info.json").write_text("{}", encoding="utf-8")
    (cache_root / "tiny-imagenet-train.arrow").write_bytes(b"arrow")

    assert _hf_prepared_cache_available(str(tmp_path), "zh-plus/tiny-imagenet")


def test_hf_prepared_cache_available_rejects_incomplete_cache(tmp_path):
    cache_root = tmp_path / "zh-plus___tiny-imagenet" / "default" / "0.0.0" / "hash"
    cache_root.mkdir(parents=True)
    (cache_root / "dataset_info.json").write_text("{}", encoding="utf-8")

    assert not _hf_prepared_cache_available(str(tmp_path), "zh-plus/tiny-imagenet")


def test_load_hf_dataset_with_cache_preference_tries_offline_first_and_restores_env(tmp_path, monkeypatch):
    cache_root = tmp_path / "zh-plus___tiny-imagenet" / "default" / "0.0.0" / "hash"
    cache_root.mkdir(parents=True)
    (cache_root / "dataset_info.json").write_text("{}", encoding="utf-8")
    (cache_root / "tiny-imagenet-train.arrow").write_bytes(b"arrow")

    logger = _CaptureLogger()
    calls = []

    def fake_load_dataset(*args, **kwargs):
        calls.append(os.environ.get("HF_HUB_OFFLINE"))
        if len(calls) == 1:
            raise RuntimeError("offline miss")
        return {"train": []}

    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)

    result = _load_hf_dataset_with_cache_preference(
        load_dataset_fn=fake_load_dataset,
        dataset_name="zh-plus/tiny-imagenet",
        config_name="",
        load_kwargs={"cache_dir": str(tmp_path)},
        active_logger=logger,
        tag="HF",
    )

    assert result == {"train": []}
    assert calls == ["1", None]
    assert os.environ.get("HF_HUB_OFFLINE") is None
    assert logger.infos
    assert logger.warnings
