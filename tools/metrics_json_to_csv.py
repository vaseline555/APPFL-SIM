#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _flatten_numeric_metrics(
    obj: Dict[str, Any], parent: str = "", sep: str = "."
) -> Dict[str, float]:
    flat: Dict[str, float] = {}
    for key, value in obj.items():
        name = f"{parent}{sep}{key}" if parent else str(key)
        if isinstance(value, dict):
            flat.update(_flatten_numeric_metrics(value, parent=name, sep=sep))
            continue
        if isinstance(value, (int, float)):
            flat[name] = float(value)
    return flat


def _load_records(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("metrics.json must contain a JSON array.")
    records: List[Dict[str, Any]] = []
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Entry #{idx} must be an object.")
        round_id = item.get("round")
        metrics = item.get("metrics", {})
        if not isinstance(round_id, int):
            raise ValueError(f"Entry #{idx} has invalid/missing 'round' (expected int).")
        if not isinstance(metrics, dict):
            raise ValueError(f"Entry #{idx} has invalid 'metrics' (expected object).")
        records.append({"round": round_id, "metrics": metrics})
    return records


def _write_wide_csv(
    records: Iterable[Dict[str, Any]],
    output_path: Path,
    sep: str,
) -> None:
    rows: List[Dict[str, Any]] = []
    all_columns = set()

    for record in records:
        row = {"round": int(record["round"])}
        flat = _flatten_numeric_metrics(record["metrics"], sep=sep)
        row.update(flat)
        rows.append(row)
        all_columns.update(flat.keys())

    fieldnames = ["round"] + sorted(all_columns)
    with output_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_long_csv(
    records: Iterable[Dict[str, Any]],
    output_path: Path,
    sep: str,
) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["round", "metric", "value"])
        writer.writeheader()
        for record in records:
            round_id = int(record["round"])
            flat = _flatten_numeric_metrics(record["metrics"], sep=sep)
            for key in sorted(flat.keys()):
                writer.writerow({"round": round_id, "metric": key, "value": flat[key]})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert APPFL-SIM metrics.json to CSV."
    )
    parser.add_argument("metrics_json", help="Path to metrics.json")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output CSV path (default: <metrics_json_stem>.csv)",
    )
    parser.add_argument(
        "--format",
        choices=["wide", "long"],
        default="wide",
        help="CSV format: wide(one row per round) or long(one row per metric).",
    )
    parser.add_argument(
        "--sep",
        default="_",
        help="Metric key separator for flattened names (default: '_').",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.metrics_json).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else input_path.with_suffix(".csv")
    )

    records = _load_records(input_path)
    if args.format == "long":
        _write_long_csv(records, output_path=output_path, sep=args.sep)
    else:
        _write_wide_csv(records, output_path=output_path, sep=args.sep)

    print(f"Wrote CSV: {output_path}")


if __name__ == "__main__":
    main()
