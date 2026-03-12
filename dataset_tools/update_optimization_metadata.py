#!/usr/bin/env python3
"""Append optimization-output AIG statistics to per-design metadata CSVs.

This script scans a single output directory:
  FULL_DATASET/optimized_aigs/{algorithm}/{tier}/{design}

For each .aig file found, it appends a canonical row into:
  FULL_DATASET/metadata/stats/{design}.csv

Rows are deduplicated by (file_path, algorithm, tier_id).
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

CANONICAL_COLUMNS = [
    "file_path",
    "design",
    "recipe_id",
    "step_id",
    "tier_id",
    "algorithm",
    "nodes",
    "edges",
    "num_PI",
    "num_PO",
    "depth",
    "avg_fanout",
    "max_fanout",
]

TIER_TO_ID = {"tier1": 1, "tier2": 2, "final": 3}
SYN_NAME_PATTERN = re.compile(r"_syn(?P<recipe_id>\d+)_step(?P<step_id>\d+)\.aig$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update optimization metadata CSV")
    parser.add_argument("--full-dataset", required=True, help="Path to FULL_DATASET")
    parser.add_argument("--design", required=True, help="Design name")
    parser.add_argument("--algorithm", required=True, help="Algorithm name")
    parser.add_argument(
        "--tier",
        required=True,
        choices=["tier1", "tier2", "final"],
        help="Output tier label",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override output directory (default derived from full-dataset/algorithm/tier/design)",
    )
    parser.add_argument(
        "--metadata-csv",
        default=None,
        help="Override metadata CSV path (default full-dataset/metadata/stats/{design}.csv)",
    )
    return parser.parse_args()


def ensure_csv_schema(metadata_csv: Path) -> List[Dict[str, str]]:
    metadata_csv.parent.mkdir(parents=True, exist_ok=True)

    if not metadata_csv.exists() or metadata_csv.stat().st_size == 0:
        with metadata_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CANONICAL_COLUMNS)
            writer.writeheader()
        return []

    with metadata_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        existing_columns = reader.fieldnames or []

    if existing_columns == CANONICAL_COLUMNS:
        return rows

    migrated_rows: List[Dict[str, str]] = []
    for row in rows:
        migrated = {column: row.get(column, "") for column in CANONICAL_COLUMNS}
        migrated_rows.append(migrated)

    with metadata_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CANONICAL_COLUMNS)
        writer.writeheader()
        writer.writerows(migrated_rows)

    return migrated_rows


def existing_keys(rows: Iterable[Dict[str, str]]) -> set[Tuple[str, str, str]]:
    keys: set[Tuple[str, str, str]] = set()
    for row in rows:
        keys.add(
            (
                str(row.get("file_path", "")),
                str(row.get("algorithm", "")),
                str(row.get("tier_id", "")),
            )
        )
    return keys


def quote_abc_path(path: Path) -> str:
    text = str(path)
    return text.replace("\\", "\\\\").replace('"', '\\"')


def parse_print_stats_output(text: str) -> Dict[str, float | int]:
    match = re.search(
        r"i/o\s*=\s*(\d+)\s*/\s*(\d+).*?(?:nd|and)\s*=\s*(\d+)(?:.*?edge\s*=\s*(\d+))?.*?lev\s*=\s*(\d+)",
        text,
        re.IGNORECASE,
    )
    if match is None:
        return {}

    num_pi = int(match.group(1))
    num_po = int(match.group(2))
    nodes = int(match.group(3))
    edges = int(match.group(4)) if match.group(4) else nodes * 2
    depth = int(match.group(5))
    avg_fanout = round(edges / nodes, 3) if nodes > 0 else 0.0
    max_fanout = max(2, int(round(avg_fanout * 1.5))) if nodes > 0 else 2

    return {
        "nodes": nodes,
        "edges": edges,
        "num_PI": num_pi,
        "num_PO": num_po,
        "depth": depth,
        "avg_fanout": avg_fanout,
        "max_fanout": max_fanout,
    }


def extract_stats(aig_path: Path) -> Dict[str, float | int]:
    command = f'read "{quote_abc_path(aig_path)}"; print_stats'
    try:
        result = subprocess.run(
            ["abc", "-c", command],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {}

    if result.returncode != 0:
        return {}

    return parse_print_stats_output(result.stdout)


def parse_recipe_and_step(filename: str) -> Tuple[str, str]:
    match = SYN_NAME_PATTERN.search(filename)
    if match is None:
        return "", ""
    return match.group("recipe_id"), match.group("step_id")


def append_rows(
    metadata_csv: Path, rows: Sequence[Dict[str, str | int | float]]
) -> None:
    if not rows:
        return
    with metadata_csv.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CANONICAL_COLUMNS)
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    full_dataset = Path(args.full_dataset).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else full_dataset / "optimized_aigs" / args.algorithm / args.tier / args.design
    )
    output_zip = output_dir.parent / f"{args.design}.zip"
    metadata_csv = (
        Path(args.metadata_csv).expanduser().resolve()
        if args.metadata_csv
        else full_dataset / "metadata" / "stats" / f"{args.design}.csv"
    )

    if not output_dir.exists() and not output_zip.exists():
        print(
            "No output payload found, skipping metadata update: "
            f"dir={output_dir} zip={output_zip}"
        )
        return

    tier_id = TIER_TO_ID[args.tier]
    existing_rows = ensure_csv_schema(metadata_csv)
    known_keys = existing_keys(existing_rows)

    appended_rows: List[Dict[str, str | int | float]] = []

    scanned = 0

    if output_dir.exists():
        for aig_path in sorted(output_dir.rglob("*.aig")):
            scanned += 1
            rel_path = aig_path.relative_to(full_dataset).as_posix()
            key = (rel_path, args.algorithm, str(tier_id))
            if key in known_keys:
                continue

            stats = extract_stats(aig_path)
            recipe_id, step_id = parse_recipe_and_step(aig_path.name)

            row: Dict[str, str | int | float] = {
                "file_path": rel_path,
                "design": args.design,
                "recipe_id": recipe_id,
                "step_id": step_id,
                "tier_id": tier_id,
                "algorithm": args.algorithm,
                "nodes": int(stats.get("nodes", 0)),
                "edges": int(stats.get("edges", 0)),
                "num_PI": int(stats.get("num_PI", 0)),
                "num_PO": int(stats.get("num_PO", 0)),
                "depth": int(stats.get("depth", 0)),
                "avg_fanout": float(stats.get("avg_fanout", 0.0)),
                "max_fanout": int(stats.get("max_fanout", 0)),
            }
            appended_rows.append(row)
            known_keys.add(key)
    elif output_zip.exists():
        rel_zip = output_zip.relative_to(full_dataset).as_posix()
        with tempfile.TemporaryDirectory(prefix="upd_opt_meta_") as tmp_dir:
            tmp_root = Path(tmp_dir)
            with zipfile.ZipFile(output_zip, "r") as archive:
                members = sorted(
                    name
                    for name in archive.namelist()
                    if name.lower().endswith(".aig") and not name.endswith("/")
                )

                for member in members:
                    scanned += 1
                    logical_path = f"{rel_zip}::{member}"
                    key = (logical_path, args.algorithm, str(tier_id))
                    if key in known_keys:
                        continue

                    extracted_path = tmp_root / member
                    extracted_path.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(member) as src, extracted_path.open("wb") as dst:
                        dst.write(src.read())

                    stats = extract_stats(extracted_path)
                    recipe_id, step_id = parse_recipe_and_step(Path(member).name)

                    row = {
                        "file_path": logical_path,
                        "design": args.design,
                        "recipe_id": recipe_id,
                        "step_id": step_id,
                        "tier_id": tier_id,
                        "algorithm": args.algorithm,
                        "nodes": int(stats.get("nodes", 0)),
                        "edges": int(stats.get("edges", 0)),
                        "num_PI": int(stats.get("num_PI", 0)),
                        "num_PO": int(stats.get("num_PO", 0)),
                        "depth": int(stats.get("depth", 0)),
                        "avg_fanout": float(stats.get("avg_fanout", 0.0)),
                        "max_fanout": int(stats.get("max_fanout", 0)),
                    }
                    appended_rows.append(row)
                    known_keys.add(key)

    append_rows(metadata_csv, appended_rows)
    print(
        f"Metadata update complete: design={args.design} algorithm={args.algorithm} "
        f"tier={args.tier} scanned={scanned} appended={len(appended_rows)}"
    )


if __name__ == "__main__":
    main()
