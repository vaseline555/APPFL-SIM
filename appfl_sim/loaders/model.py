from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

import torch

from appfl_sim.models import MODEL_REGISTRY, get_model_class


@dataclass
class ModelSpec:
    source: str
    name: str
    num_classes: int
    in_channels: int
    input_shape: Tuple[int, ...]
    context: Dict[str, Any]
    model_kwargs: Dict[str, Any]
    timm_pretrained: bool
    timm_kwargs: Dict[str, Any]
    hf_task: str
    hf_pretrained: bool
    hf_local_files_only: bool
    hf_trust_remote_code: bool
    hf_gradient_checkpointing: bool
    hf_cache_dir: str
    hf_kwargs: Dict[str, Any]
    hf_config_overrides: Dict[str, Any]


def _path_get(cfg: Dict[str, Any], path: str, default: Any = None) -> Any:
    parts = [p for p in str(path).split(".") if p]
    cur: Any = cfg
    for part in parts:
        if isinstance(cur, dict):
            if part not in cur:
                return default
            cur = cur[part]
            continue
        return default
    return default if cur is None else cur


def _as_dict(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _safe_int(value: Any, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except Exception:
        return default


def _safe_float(value: Any, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except Exception:
        return default


def _safe_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"1", "true", "yes", "y", "on"}:
            return True
        if v in {"0", "false", "no", "n", "off"}:
            return False
        return default
    return bool(value)


def _default_model_context(
    cfg: Dict[str, Any], input_shape: Tuple[int, ...], num_classes: int
) -> Dict[str, Any]:
    channels = int(input_shape[0]) if len(input_shape) >= 1 else 1
    spatial = input_shape[1:] if len(input_shape) >= 2 else (1,)
    if len(spatial) == 0:
        spatial = (1,)

    inferred_resize = int(spatial[0])
    inferred_in_features = 1
    for dim in input_shape:
        inferred_in_features *= int(dim)

    model_configs = _as_dict(_path_get(cfg, "model.configs", {}))
    context = {
        "model_name": _path_get(cfg, "model.name", "SimpleCNN"),
        "num_classes": _safe_int(num_classes, 0),
        "in_channels": channels,
        "in_features": inferred_in_features,
        "resize": inferred_resize,
        "hidden_size": _safe_int(model_configs.get("hidden_size", 64), 64),
        "dropout": _safe_float(model_configs.get("dropout", 0.0), 0.0),
        "num_layers": _safe_int(model_configs.get("num_layers", 2), 2),
        "num_embeddings": _safe_int(model_configs.get("num_embeddings", 10000), 10000),
        "embedding_size": _safe_int(model_configs.get("embedding_size", 128), 128),
        "seq_len": _safe_int(model_configs.get("seq_len", 128), 128),
        "B": _safe_int(_path_get(cfg, "train.batch_size", 32), 32),
    }
    context.update(model_configs)
    return context


def _parse_model_spec(
    cfg: Dict[str, Any],
    input_shape: Tuple[int, ...],
    num_classes: int,
) -> ModelSpec:
    context = _default_model_context(cfg, input_shape=input_shape, num_classes=num_classes)
    source = str(_path_get(cfg, "model.backend", "auto")).lower()
    name = str(_path_get(cfg, "model.name", "SimpleCNN")).strip()
    if source in {"timm", "hf"} and not name:
        raise ValueError(
            f"model.backend={source} requires model.name to be set to the exact backend name/card."
        )
    model_configs = _as_dict(_path_get(cfg, "model.configs", {}))
    model_path = str(_path_get(cfg, "model.path", "./appfl_sim/models"))
    in_channels = int(model_configs.get("in_channels", context.get("in_channels", 1)))
    resolved_num_classes = int(
        model_configs.get("num_classes", context.get("num_classes", num_classes))
    )

    return ModelSpec(
        source=source,
        name=name,
        num_classes=resolved_num_classes,
        in_channels=in_channels,
        input_shape=tuple(input_shape),
        context=context,
        model_kwargs=model_configs,
        timm_pretrained=_safe_bool(model_configs.get("pretrained", False), False),
        timm_kwargs=_as_dict(model_configs.get("timm_kwargs", {})),
        hf_task=str(model_configs.get("hf_task", "sequence_classification")).strip().lower(),
        hf_pretrained=_safe_bool(model_configs.get("pretrained", False), False),
        hf_local_files_only=_safe_bool(model_configs.get("hf_local_files_only", False), False),
        hf_trust_remote_code=_safe_bool(model_configs.get("hf_trust_remote_code", False), False),
        hf_gradient_checkpointing=_safe_bool(model_configs.get("hf_gradient_checkpointing", False), False),
        hf_cache_dir=str(model_configs.get("hf_cache_dir", model_path)),
        hf_kwargs=_as_dict(model_configs.get("hf_kwargs", {})),
        hf_config_overrides=_as_dict(model_configs.get("hf_config_overrides", {})),
    )


def _load_appfl_model(spec: ModelSpec):
    if spec.name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise ValueError(
            f"model.backend=local requires an exact local model name. "
            f"Got '{spec.name}'. Available: {available}"
        )

    model_class = get_model_class(spec.name)

    context = dict(spec.context)
    context.update(spec.model_kwargs)
    context["model_name"] = spec.name
    context["num_classes"] = spec.num_classes
    context["in_channels"] = spec.in_channels

    signature = inspect.signature(model_class.__init__)
    model_args: Dict[str, Any] = {}
    missing = []
    for name, param in signature.parameters.items():
        if name == "self":
            continue
        if name in context:
            model_args[name] = context[name]
            continue
        if param.default is not inspect._empty:
            model_args[name] = param.default
            continue
        missing.append(name)

    if missing:
        raise ValueError(
            f"Missing required model args for {spec.name}: {missing}. "
            "Pass them through config/CLI."
        )
    return model_class(**model_args)


def _resolve_timm_model_name(spec: ModelSpec) -> str:
    return spec.name


def _load_timm_model(spec: ModelSpec):
    try:
        import timm
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "timm backend requested but timm is not installed. "
            "Install with: pip install timm"
        ) from e

    timm_name = _resolve_timm_model_name(spec)
    create_kwargs = dict(spec.model_kwargs)
    create_kwargs.update(spec.timm_kwargs)

    create_kwargs.setdefault("num_classes", int(spec.num_classes))
    create_kwargs.setdefault("in_chans", int(spec.in_channels))
    pretrained = bool(spec.timm_pretrained)

    try:
        return timm.create_model(timm_name, pretrained=pretrained, **create_kwargs)
    except Exception as e:
        raise ValueError(
            f"Failed to create timm model '{timm_name}'. "
            "Provide an exact timm model name in model_name."
        ) from e


class _HFAdapter(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, task: str) -> None:
        super().__init__()
        self.model = model
        self.task = task

    def forward(self, x):
        kwargs: Dict[str, Any] = {}

        if isinstance(x, dict):
            kwargs = x
        elif isinstance(x, (list, tuple)) and len(x) == 2 and all(torch.is_tensor(v) for v in x):
            kwargs = {
                "input_ids": x[0].long(),
                "attention_mask": x[1].long(),
            }
        elif torch.is_tensor(x):
            if self.task in {"sequence_classification", "token_classification", "causal_lm"}:
                if x.ndim >= 3 and x.shape[1] >= 2:
                    kwargs = {
                        "input_ids": x[:, 0].long(),
                        "attention_mask": x[:, 1].long(),
                    }
                else:
                    kwargs = {"input_ids": x.long()}
            elif self.task == "vision_classification":
                kwargs = {"pixel_values": x.float()}
            else:
                kwargs = {"input_ids": x.long()}
        else:
            raise TypeError(f"Unsupported HuggingFace input type: {type(x)}")

        outputs = self.model(**kwargs)
        if hasattr(outputs, "logits"):
            return outputs.logits
        if isinstance(outputs, tuple) and len(outputs) > 0:
            return outputs[0]
        if hasattr(outputs, "last_hidden_state"):
            return outputs.last_hidden_state
        return outputs


def _resolve_hf_model_id(spec: ModelSpec) -> str:
    return spec.name


def _resolve_hf_task(spec: ModelSpec) -> str:
    task = str(spec.hf_task).strip().lower()
    if task:
        return task
    if len(spec.input_shape) >= 2:
        return "vision_classification"
    return "sequence_classification"


def _build_hf_scratch_config(model_id: str, spec: ModelSpec, task: str):
    from transformers import AutoConfig

    overrides = dict(spec.hf_config_overrides)
    overrides.setdefault("num_labels", int(spec.num_classes))
    if task in {"sequence_classification", "token_classification", "causal_lm"}:
        overrides.setdefault("vocab_size", int(spec.context.get("num_embeddings", 10000)))
    try:
        return AutoConfig.from_pretrained(
            model_id,
            cache_dir=str(spec.hf_cache_dir),
            local_files_only=bool(spec.hf_local_files_only),
            trust_remote_code=bool(spec.hf_trust_remote_code),
            **overrides,
        )
    except Exception as e:
        raise ValueError(
            f"pretrained=false requires accessible HF config for model '{model_id}'. "
            "Use `model.configs.hf_local_files_only=false` to allow download, "
            "or cache it first. If this model card is not a Transformers model "
            "(missing `config.json`), choose a Transformers-compatible model id."
        ) from e


def _load_hf_model(spec: ModelSpec):
    try:
        from transformers import (
            AutoModelForCausalLM,
            AutoModelForImageClassification,
            AutoModelForSequenceClassification,
            AutoModelForTokenClassification,
        )
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "huggingface backend requested but transformers is not installed. "
            "Install with: pip install transformers"
        ) from e

    task = _resolve_hf_task(spec)
    model_id = _resolve_hf_model_id(spec)
    pretrained = bool(spec.hf_pretrained)
    local_files_only = bool(spec.hf_local_files_only)
    trust_remote_code = bool(spec.hf_trust_remote_code)

    common_kwargs = dict(spec.model_kwargs)
    for reserved_key in (
        "pretrained",
        "hf_task",
        "hf_local_files_only",
        "hf_trust_remote_code",
        "hf_gradient_checkpointing",
        "hf_kwargs",
        "hf_config_overrides",
        "hf_cache_dir",
        "timm_kwargs",
    ):
        common_kwargs.pop(reserved_key, None)
    common_kwargs.update(spec.hf_kwargs)

    if task == "sequence_classification":
        model_cls = AutoModelForSequenceClassification
    elif task == "token_classification":
        model_cls = AutoModelForTokenClassification
    elif task == "causal_lm":
        model_cls = AutoModelForCausalLM
    elif task == "vision_classification":
        model_cls = AutoModelForImageClassification
    else:
        raise ValueError(
            "Unsupported hf task: "
            f"{task}. Use one of: sequence_classification, token_classification, causal_lm, vision_classification"
        )

    if pretrained:
        model = model_cls.from_pretrained(
            model_id,
            num_labels=int(spec.num_classes),
            cache_dir=str(spec.hf_cache_dir),
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
            **common_kwargs,
        )
    else:
        config = _build_hf_scratch_config(model_id, spec, task)
        model = model_cls.from_config(config)

    if bool(spec.hf_gradient_checkpointing) and hasattr(
        model, "gradient_checkpointing_enable"
    ):
        model.gradient_checkpointing_enable()

    return _HFAdapter(model=model, task=task)


def _is_timm_candidate(name: str) -> bool:
    try:
        import timm

        return name in set(timm.list_models())
    except Exception:
        return False


def _is_hf_candidate(name: str, spec: ModelSpec) -> bool:
    return "/" in str(name)


def _resolve_source(spec: ModelSpec) -> str:
    source = spec.source.lower()
    if source == "local":
        return "appfl"
    if source in {"timm", "hf"}:
        return source
    if source != "auto":
        raise ValueError("model.backend must be one of: auto, local, timm, hf")

    if spec.name in MODEL_REGISTRY:
        return "appfl"
    if _is_timm_candidate(spec.name):
        return "timm"
    if _is_hf_candidate(spec.name, spec):
        return "hf"

    raise ValueError(
        f"Unable to resolve model source for exact name '{spec.name}'. "
        "Set model.backend explicitly and provide exact backend model name/card."
    )


def load_model(
    cfg: Dict[str, Any],
    input_shape: Tuple[int, ...],
    num_classes: int,
):
    """Unified model factory with explicit backend controls.

    Backends:
    - ``local``: local models in ``appfl_sim.models``.
    - ``timm``: exact model name via ``model.name``.
    - ``hf``: exact model card id via ``model.name``.

    Aliases are intentionally disabled to avoid implicit behavior.
    """
    spec = _parse_model_spec(cfg=cfg, input_shape=input_shape, num_classes=num_classes)
    source = _resolve_source(spec)

    if source == "appfl":
        return _load_appfl_model(spec)
    if source == "timm":
        return _load_timm_model(spec)
    if source == "hf":
        return _load_hf_model(spec)

    raise RuntimeError(f"Unsupported resolved model source: {source}")
