#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class Component:
    kind: str  # aggregator | trainer | scheduler
    source: Path
    class_name: str

    @property
    def module_name(self) -> str:
        return self.source.stem


def _must_exist(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def _ensure_class_exists(py_file: Path, class_name: str) -> None:
    tree = ast.parse(py_file.read_text(encoding="utf-8"))
    names = [n.name for n in tree.body if isinstance(n, ast.ClassDef)]
    if class_name not in names:
        raise ValueError(
            f"Class '{class_name}' is not defined in {py_file}. "
            f"Found classes: {names}"
        )


def _copy_component(component: Component, target_algorithm_dir: Path) -> tuple[Path, str]:
    dst_dir = target_algorithm_dir / component.kind
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / component.source.name
    src_text = component.source.read_text(encoding="utf-8")
    dst_text = _rewrite_imports_for_appfl(src_text)
    dst.write_text(dst_text, encoding="utf-8")
    return dst, dst_text


def _rewrite_imports_for_appfl(text: str) -> str:
    """Rewrite simulation package imports to APPFL package imports."""
    rewritten = re.sub(r"\bappfl_sim\b", "appfl", text)
    return rewritten


def _uses_appfl_metrics(text: str) -> bool:
    return bool(re.search(r"\b(?:from|import)\s+appfl\.metrics\b", text))


def _export_metrics_support(target_appfl_pkg_dir: Path) -> list[Path]:
    """Copy appfl_sim metrics package as appfl/metrics support into target tree."""
    src_metrics_dir = Path(__file__).resolve().parents[1] / "appfl_sim" / "metrics"
    _must_exist(src_metrics_dir, "appfl_sim metrics package")
    dst_metrics_dir = target_appfl_pkg_dir / "metrics"
    dst_metrics_dir.mkdir(parents=True, exist_ok=True)

    copied: list[Path] = []
    for src in sorted(src_metrics_dir.glob("*.py")):
        text = src.read_text(encoding="utf-8")
        text = _rewrite_imports_for_appfl(text)
        dst = dst_metrics_dir / src.name
        dst.write_text(text, encoding="utf-8")
        copied.append(dst)
    return copied


def _extract_appfl_import_modules(text: str) -> set[str]:
    modules: set[str] = set()
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return modules

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name
                if name == "appfl" or name.startswith("appfl."):
                    modules.add(name)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module
            if mod and (mod == "appfl" or mod.startswith("appfl.")):
                modules.add(mod)
    return modules


def _appfl_module_exists(appfl_src_dir: Path, module: str) -> bool:
    if module == "appfl":
        return (appfl_src_dir / "__init__.py").exists()
    if not module.startswith("appfl."):
        return True
    rel_parts = module.split(".")[1:]
    rel = Path(*rel_parts)
    module_file = (appfl_src_dir / rel).with_suffix(".py")
    module_pkg_init = appfl_src_dir / rel / "__init__.py"
    return module_file.exists() or module_pkg_init.exists()


def _check_appfl_import_compatibility(
    rewritten_texts: list[str],
    appfl_root: Path,
    overlay_root: Optional[Path] = None,
) -> list[str]:
    appfl_src_dir = appfl_root / "src" / "appfl"
    if not appfl_src_dir.exists():
        raise FileNotFoundError(f"APPFL source directory not found: {appfl_src_dir}")
    overlay_src_dir = None
    if overlay_root is not None:
        candidate = overlay_root / "src" / "appfl"
        if candidate.exists():
            overlay_src_dir = candidate

    unresolved: set[str] = set()
    for text in rewritten_texts:
        for module in _extract_appfl_import_modules(text):
            if overlay_src_dir is not None and _appfl_module_exists(overlay_src_dir, module):
                continue
            if not _appfl_module_exists(appfl_src_dir, module):
                unresolved.add(module)
    return sorted(unresolved)


def _append_once(path: Path, text: str) -> None:
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    if text in existing:
        return
    if existing and not existing.endswith("\n"):
        existing += "\n"
    path.write_text(existing + text + "\n", encoding="utf-8")


def _patch_init(init_file: Path, module_name: str, class_name: str) -> None:
    import_line = f"from .{module_name} import {class_name}"
    _append_once(init_file, import_line)

    content = init_file.read_text(encoding="utf-8")
    if "__all__" in content:
        _append_once(init_file, f'__all__.append("{class_name}")')
    else:
        _append_once(init_file, f'__all__ = ["{class_name}"]')


def _write_patch_snippet(patch_dir: Path, component: Component) -> None:
    patch_dir.mkdir(parents=True, exist_ok=True)
    path = patch_dir / f"{component.kind}___init__.txt"
    text = (
        f"# Add to src/appfl/algorithm/{component.kind}/__init__.py\n"
        f"from .{component.module_name} import {component.class_name}\n"
        f"__all__.append(\"{component.class_name}\")\n"
    )
    path.write_text(text, encoding="utf-8")


def _write_config_template(out_dir: Path, algorithm: str, trainer: Optional[Component], scheduler: Optional[Component], aggregator: Component) -> Path:
    cfg_dir = out_dir / "config" / "algorithms"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = cfg_dir / f"{algorithm}.yaml"

    trainer_name = trainer.class_name if trainer is not None else "VanillaTrainer"
    scheduler_name = scheduler.class_name if scheduler is not None else "SyncScheduler"

    text = (
        f"# APPFL algorithm template for {algorithm}\n"
        "client_configs:\n"
        "  train_configs:\n"
        f"    trainer: {trainer_name}\n"
        "\n"
        "server_configs:\n"
        "  num_clients: 3\n"
        f"  aggregator: {aggregator.class_name}\n"
        f"  scheduler: {scheduler_name}\n"
        "  aggregator_kwargs: {}\n"
        "  scheduler_kwargs:\n"
        "    num_clients: 3\n"
    )
    cfg_path.write_text(text, encoding="utf-8")
    return cfg_path


def _write_install_guide(out_dir: Path, algorithm: str, components: list[Component]) -> Path:
    guide = out_dir / "INSTALL_PLUGIN.md"
    lines = [
        f"# Install Plugin: {algorithm}",
        "",
        "1. Copy generated files under `src/appfl/algorithm/*` into APPFL-main at the same paths.",
        "2. For each component, patch APPFL `__init__.py` using snippets under `patches/`.",
        "3. Copy config template from `config/algorithms/`.",
        "",
        "Components:",
    ]
    for comp in components:
        lines.append(f"- {comp.kind}: `{comp.class_name}` from `{comp.source}`")

    guide.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return guide


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export appfl_sim algorithm component(s) into APPFL-main compatible module layout."
    )
    p.add_argument("--algorithm", required=True, help="Algorithm slug (e.g., myalgo)")

    p.add_argument("--aggregator-source", required=True)
    p.add_argument("--aggregator-class", required=True)

    p.add_argument("--trainer-source", default="")
    p.add_argument("--trainer-class", default="")

    p.add_argument("--scheduler-source", default="")
    p.add_argument("--scheduler-class", default="")

    p.add_argument(
        "--output-dir",
        default="",
        help="Output artifact directory. Default: build/appfl_plugin_<algorithm>",
    )
    p.add_argument(
        "--appfl-root",
        default="",
        help="If set, directly copy/patch under <appfl-root>/src/appfl/algorithm",
    )
    p.add_argument(
        "--check-appfl-root",
        default="",
        help="Optional APPFL root for compatibility import audit in artifact mode.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    algorithm = str(args.algorithm).strip().lower()
    if not algorithm:
        raise ValueError("--algorithm must be non-empty")

    aggregator = Component(
        kind="aggregator",
        source=Path(args.aggregator_source).resolve(),
        class_name=str(args.aggregator_class).strip(),
    )
    _must_exist(aggregator.source, "Aggregator source")
    _ensure_class_exists(aggregator.source, aggregator.class_name)

    trainer = None
    if str(args.trainer_source).strip() or str(args.trainer_class).strip():
        if not (str(args.trainer_source).strip() and str(args.trainer_class).strip()):
            raise ValueError("Provide both --trainer-source and --trainer-class")
        trainer = Component(
            kind="trainer",
            source=Path(args.trainer_source).resolve(),
            class_name=str(args.trainer_class).strip(),
        )
        _must_exist(trainer.source, "Trainer source")
        _ensure_class_exists(trainer.source, trainer.class_name)

    scheduler = None
    if str(args.scheduler_source).strip() or str(args.scheduler_class).strip():
        if not (str(args.scheduler_source).strip() and str(args.scheduler_class).strip()):
            raise ValueError("Provide both --scheduler-source and --scheduler-class")
        scheduler = Component(
            kind="scheduler",
            source=Path(args.scheduler_source).resolve(),
            class_name=str(args.scheduler_class).strip(),
        )
        _must_exist(scheduler.source, "Scheduler source")
        _ensure_class_exists(scheduler.source, scheduler.class_name)

    components = [aggregator] + ([trainer] if trainer else []) + ([scheduler] if scheduler else [])

    if str(args.appfl_root).strip():
        appfl_root = Path(args.appfl_root).resolve()
        target_algorithm_dir = appfl_root / "src" / "appfl" / "algorithm"
        _must_exist(target_algorithm_dir, "APPFL algorithm directory")
        out_dir = appfl_root
        direct_mode = True
    else:
        out_dir = (
            Path(args.output_dir).resolve()
            if str(args.output_dir).strip()
            else (Path.cwd() / "build" / f"appfl_plugin_{algorithm}").resolve()
        )
        target_algorithm_dir = out_dir / "src" / "appfl" / "algorithm"
        direct_mode = False

    copied = []
    rewritten_texts: list[str] = []
    for comp in components:
        dst, rewritten = _copy_component(comp, target_algorithm_dir)
        rewritten_texts.append(rewritten)
        copied.append({"kind": comp.kind, "class": comp.class_name, "src": str(comp.source), "dst": str(dst)})

    metrics_support_files: list[str] = []
    if any(_uses_appfl_metrics(text) for text in rewritten_texts):
        target_appfl_pkg_dir = target_algorithm_dir.parent
        exported = _export_metrics_support(target_appfl_pkg_dir)
        metrics_support_files = [str(path) for path in exported]
        for file_path in exported:
            rewritten_texts.append(file_path.read_text(encoding="utf-8"))

    if direct_mode:
        for comp in components:
            init_file = target_algorithm_dir / comp.kind / "__init__.py"
            _must_exist(init_file, f"APPFL {comp.kind} __init__.py")
            _patch_init(init_file, comp.module_name, comp.class_name)
    else:
        patch_dir = out_dir / "patches"
        for comp in components:
            _write_patch_snippet(patch_dir, comp)
        _write_install_guide(out_dir, algorithm, components)

    cfg_path = _write_config_template(out_dir, algorithm, trainer, scheduler, aggregator)

    check_root = None
    if direct_mode:
        check_root = appfl_root
    elif str(args.check_appfl_root).strip():
        check_root = Path(args.check_appfl_root).resolve()

    unresolved_imports: list[str] = []
    if check_root is not None:
        unresolved_imports = _check_appfl_import_compatibility(
            rewritten_texts,
            check_root,
            overlay_root=None if direct_mode else out_dir,
        )
        if unresolved_imports:
            unresolved_text = ", ".join(unresolved_imports)
            raise RuntimeError(
                "Exported plugin contains imports not found in target APPFL tree: "
                f"{unresolved_text}"
            )

    manifest = {
        "algorithm": algorithm,
        "mode": "direct" if direct_mode else "artifact",
        "output_root": str(out_dir),
        "copied": copied,
        "metrics_support_files": metrics_support_files,
        "config_template": str(cfg_path),
        "compatibility_check_root": str(check_root) if check_root is not None else "",
        "unresolved_appfl_imports": unresolved_imports,
    }
    manifest_path = out_dir / "plugin_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
