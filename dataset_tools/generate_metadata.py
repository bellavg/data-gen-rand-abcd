#!/usr/bin/env python3
"""
Metadata Collection and Organization Script for Full AIG Dataset.

This script COLLECTS existing metadata from Random AIG and OpenABC-D datasets
and moves/converts them to the canonical FULL_DATASET format.

The metadata already exists in the source datasets:
- Random AIG: bench/{design}/metadata/{design}.csv
- OpenABC-D: statistics/ (various CSV/PKL files)

Usage:
    python generate_metadata.py /path/to/FULL_DATASET --workers 4 --validate --summary
"""

import argparse
import glob
import json
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

try:
    import pandas as pd
except ImportError:
    print("Warning: pandas not installed. Install with: pip install pandas")
    sys.exit(1)

# Canonical CSV header as defined in README
CANONICAL_HEADER = "file_path,design,recipe_id,step_id,tier_id,nodes,edges,num_PI,num_PO,depth,avg_fanout,max_fanout"

# Design lists from reorganize_datasets.py
RANDOM_DESIGNS = ["128", "256", "512", "1024", "2048", "4096", "8192", "16384"]
OPENABC_DESIGNS = [
    "i2c",
    "spi",
    "des3_area",
    "ss_pcm",
    "usb_phy",
    "sasc",
    "wb_dma",
    "simple_spi",
    "dynamic_node",
    "aes",
    "pci",
    "ac97_ctrl",
    "mem_ctrl",
    "tv80",
    "fpu",
    "wb_conmax",
    "tinyRocket",
    "aes_xcrypt",
    "aes_secworks",
    "jpeg",
    "bp_be",
    "ethernet",
    "vga_lcd",
    "picosoc",
    "dft",
    "idft",
    "fir",
    "iir",
    "sha256",
]

ALL_DESIGNS = RANDOM_DESIGNS + OPENABC_DESIGNS


def find_source_datasets(full_dataset_path):
    """
    Find potential source dataset locations based on common patterns.
    """
    potential_sources = {"random": [], "openabc": []}

    # Common locations for Random AIG dataset
    random_patterns = [
        os.path.join(full_dataset_path, "..", "OPENABC_DATASET"),
        os.path.join(full_dataset_path, "..", "random_dataset", "OPENABC_DATASET"),
        "/scratch-shared/*/openabc_full/OPENABC_DATASET",
        # Add user's current structure
        os.path.join(full_dataset_path, "..", "OPENABC_DATASET"),
    ]

    # Common locations for OpenABC-D dataset
    openabc_patterns = [
        "/scratch-shared/igardner1/openabc_full/OPENABC_DATASET",
        "/scratch-shared/*/openabc_full/OPENABC_DATASET",  # Generic user pattern
        os.path.join(full_dataset_path, "..", "OPENABC_DATASET"),  # Current location
        os.path.join(full_dataset_path, "..", "openabc_dataset", "OPENABC_DATASET"),
        # Check if statistics exist in current location
        os.path.join(full_dataset_path, "..", "OPENABC_DATASET", "statistics"),
        # Server-side patterns
        "/data/*/OPENABC_DATASET",
        "/home/*/OPENABC_DATASET",
    ]

    # Check Random AIG sources
    for pattern in random_patterns:
        if os.path.exists(pattern):
            bench_path = os.path.join(pattern, "bench")
            if os.path.exists(bench_path):
                potential_sources["random"].append(pattern)

    # Check OpenABC-D sources
    for pattern in openabc_patterns:
        if os.path.exists(pattern):
            if "statistics" in pattern:
                # Direct statistics folder
                potential_sources["openabc"].append(os.path.dirname(pattern))
            else:
                stats_path = os.path.join(pattern, "statistics")
                if os.path.exists(stats_path):
                    potential_sources["openabc"].append(pattern)

    return potential_sources


def collect_random_metadata(random_source, full_dataset_path, design):
    """
    Collect metadata for a Random AIG design from existing CSV files.
    """
    design_metadata_path = os.path.join(
        random_source, "bench", design, "metadata", f"{design}.csv"
    )
    target_metadata_path = os.path.join(
        full_dataset_path, "metadata", "stats", f"{design}.csv"
    )

    if not os.path.exists(design_metadata_path):
        print(
            f"  Warning: Metadata file not found for {design}: {design_metadata_path}"
        )
        return False

    try:
        # Read existing CSV and enforce canonical format for random dataset
        df = pd.read_csv(design_metadata_path)

        if df.empty:
            print(f"  ✗ Error processing {design}: CSV is empty")
            return False

        expected_columns = CANONICAL_HEADER.split(",")
        if list(df.columns) != expected_columns:
            print(
                f"  ✗ Error processing {design}: non-canonical columns. "
                f"Expected {expected_columns}, got {list(df.columns)}"
            )
            return False

        expected_prefix = f"base_aigs/{design}/"
        if not df["file_path"].astype(str).str.startswith(expected_prefix).all():
            print(
                f"  ✗ Error processing {design}: file_path values are not canonical "
                f"(must start with {expected_prefix})"
            )
            return False

        if not df["design"].astype(str).eq(design).all():
            print(
                f"  ✗ Error processing {design}: design column contains values other than '{design}'"
            )
            return False

        shutil.copy2(design_metadata_path, target_metadata_path)
        print(f"  ✓ Copied {design}.csv (canonical random metadata)")

        return True

    except (FileNotFoundError, pd.errors.EmptyDataError, ValueError) as e:
        print(f"  ✗ Error processing {design}: {e}")
        return False


def collect_openabc_metadata(openabc_source, full_dataset_path, design):
    """
    Collect metadata for an OpenABC-D design from statistics folder.
    OpenABC-D has metadata in:
    - statistics/adp/{design}.csv (area, delay, power)
    - statistics/finalAig/processed_{design}.csv (graph characteristics)
    """
    stats_path = os.path.join(openabc_source, "statistics")
    target_metadata_path = os.path.join(
        full_dataset_path, "metadata", "stats", f"{design}.csv"
    )

    # Look for design-specific CSV files in statistics folder
    possible_files = [
        # finalAig folder has graph characteristics
        os.path.join(stats_path, "finalAig", f"processed_{design}.csv"),
        os.path.join(stats_path, "finalAig", f"{design}.csv"),
        # adp folder has area/delay/power info
        os.path.join(stats_path, "adp", f"{design}.csv"),
        # Direct in statistics folder
        os.path.join(stats_path, f"{design}.csv"),
        os.path.join(stats_path, f"processed_{design}.csv"),
        # Check for PKL files that might need conversion
        os.path.join(stats_path, f"{design}.pkl"),
        os.path.join(stats_path, "synthesisstatistics.pickle"),
    ]

    source_file = None
    file_type = None

    for file_path in possible_files:
        if os.path.exists(file_path):
            source_file = file_path
            file_type = "pkl" if file_path.endswith(".pkl") else "csv"
            break

    if not source_file:
        print(f"  Warning: No metadata found for {design} in {stats_path}")
        return False

    try:
        if file_type == "csv":
            # Read and convert CSV
            df = pd.read_csv(source_file)
            canonical_df = convert_to_canonical_format(df, design, "openabc")
            canonical_df.to_csv(target_metadata_path, index=False)
            print(f"  ✓ Converted {design}.csv to canonical format")

        elif file_type == "pkl":
            # Read PKL and convert to CSV
            if "synthesisstatistics.pickle" in source_file:
                # Handle the main synthesis statistics pickle file
                df = pd.read_pickle(source_file)
                # This file contains all designs, filter for current design
                if isinstance(df, dict) and design in df:
                    design_data = pd.DataFrame(df[design])
                    canonical_df = convert_to_canonical_format(
                        design_data, design, "openabc"
                    )
                else:
                    print(
                        f"  Warning: Could not find {design} in synthesis statistics pickle"
                    )
                    return False
            else:
                # Regular pickle file for specific design
                df = pd.read_pickle(source_file)
                canonical_df = convert_to_canonical_format(df, design, "openabc")

            canonical_df.to_csv(target_metadata_path, index=False)
            print(f"  ✓ Converted {design}.pkl to canonical CSV format")

        return True

    except (FileNotFoundError, pd.errors.EmptyDataError, ValueError) as e:
        print(f"  ✗ Error processing {design}: {e}")
        return False


def convert_to_canonical_format(
    df,
    design,
    _source_type,
    force_file_path=False,
    force_design=False,
):
    """
    Convert various metadata formats to canonical format defined in README.
    """
    # Create canonical DataFrame with source index preserved.
    # This avoids scalar assignments landing on an empty frame and later becoming NaN.
    canonical_columns = CANONICAL_HEADER.split(",")

    # Map common column names to canonical names
    column_mapping = {
        # Common variations
        "file": "file_path",
        "filename": "file_path",
        "path": "file_path",
        "design_name": "design",
        "des": "design",
        "sid": "recipe_id",
        "synth_id": "recipe_id",
        "recipe": "recipe_id",
        "step": "step_id",
        "AND": "nodes",
        "and": "nodes",
        "gate_count": "nodes",
        "num_gates": "nodes",
        # Keep BUFF/NOT as-is; edges are derived explicitly below
        "edge_count": "edges",
        "PI": "num_PI",
        "primary_inputs": "num_PI",
        "PO": "num_PO",
        "primary_outputs": "num_PO",
        "LP": "depth",  # OpenABC-D uses LP for longest path
        "longest_path": "depth",
        "levels": "depth",
        "lev": "depth",
        # OpenABC-D specific mappings
        "area": "area",  # Will be ignored for canonical format
        "delay": "delay",  # Will be ignored for canonical format
    }

    # Apply column mapping
    df_mapped = df.rename(columns=column_mapping)
    canonical_df = pd.DataFrame(index=df_mapped.index, columns=canonical_columns)

    # Fill in canonical columns with available data
    for col in canonical_columns:
        if col in df_mapped.columns and not (
            (col == "file_path" and force_file_path)
            or (col == "design" and force_design)
        ):
            canonical_df[col] = df_mapped[col]
        elif col == "design":
            canonical_df[col] = design
        elif col == "tier_id":
            canonical_df[col] = ""  # Empty for base AIGs
        elif col == "file_path":
            # Defer file_path generation until all key columns are available.
            canonical_df[col] = ""
        elif col in ["avg_fanout", "max_fanout"]:
            # Fill placeholders now; recomputed below when possible
            if col == "avg_fanout":
                canonical_df[col] = 2.0
            else:  # max_fanout
                canonical_df[col] = 2
        else:
            # Default values for missing columns
            canonical_df[col] = (
                0 if col in ["nodes", "edges", "num_PI", "num_PO", "depth"] else ""
            )

    # Ensure design is always populated for this per-design conversion.
    canonical_df["design"] = canonical_df["design"].fillna(design)
    canonical_df.loc[
        canonical_df["design"].astype(str).str.strip().eq(""), "design"
    ] = design

    # Build canonical file_path from recipe_id/step_id with robust fallback to orig.
    recipe_vals = pd.to_numeric(canonical_df["recipe_id"], errors="coerce")
    step_vals = pd.to_numeric(canonical_df["step_id"], errors="coerce")
    recipe_int = recipe_vals.round().astype("Int64")
    step_int = step_vals.round().astype("Int64")
    generated_file_path = recipe_int.combine(
        step_int,
        lambda recipe_id, step_id: (
            f"base_aigs/{design}/{design}_syn{int(recipe_id)}_step{int(step_id)}.aig"
            if pd.notna(recipe_id) and pd.notna(step_id)
            else f"base_aigs/{design}/{design}_orig.aig"
        ),
    )

    if force_file_path:
        canonical_df["file_path"] = generated_file_path
    else:
        canonical_df["file_path"] = canonical_df["file_path"].fillna("")
        empty_path_mask = canonical_df["file_path"].astype(str).str.strip().eq("")
        canonical_df.loc[empty_path_mask, "file_path"] = generated_file_path.loc[
            empty_path_mask
        ]

    # Numeric coercion for robust arithmetic
    for numeric_col in [
        "nodes",
        "edges",
        "num_PI",
        "num_PO",
        "depth",
        "avg_fanout",
        "max_fanout",
    ]:
        canonical_df[numeric_col] = pd.to_numeric(
            canonical_df[numeric_col], errors="coerce"
        )

    # Derive edges more accurately for OpenABC-style inputs.
    # Priority: explicit edge_count -> BUFF+NOT -> existing edges -> 2*nodes fallback.
    edge_series = pd.Series(
        [None] * len(df_mapped), index=df_mapped.index, dtype="float64"
    )

    if "edges" in df_mapped.columns:
        edge_series = pd.to_numeric(df_mapped["edges"], errors="coerce")

    if "BUFF" in df_mapped.columns or "NOT" in df_mapped.columns:
        buff = (
            pd.to_numeric(df_mapped["BUFF"], errors="coerce")
            if "BUFF" in df_mapped.columns
            else 0
        )
        inv = (
            pd.to_numeric(df_mapped["NOT"], errors="coerce")
            if "NOT" in df_mapped.columns
            else 0
        )
        buff_not_sum = buff.fillna(0) + inv.fillna(0)
        edge_series = edge_series.fillna(buff_not_sum)

    nodes_num = pd.to_numeric(canonical_df["nodes"], errors="coerce")
    edge_series = edge_series.fillna(nodes_num * 2)
    canonical_df["edges"] = edge_series.fillna(0).astype(int)

    # Recompute avg_fanout from edges/nodes where possible
    safe_nodes = nodes_num.where(nodes_num > 0)
    computed_avg = (canonical_df["edges"] / safe_nodes).fillna(0).round(3)
    canonical_df["avg_fanout"] = computed_avg

    # Keep max_fanout if present and valid, else use a conservative estimate
    canonical_df["max_fanout"] = pd.to_numeric(
        canonical_df["max_fanout"], errors="coerce"
    )
    estimated_max = computed_avg.apply(
        lambda x: max(2, int(round(x * 1.5))) if x > 0 else 2
    )
    canonical_df["max_fanout"] = (
        canonical_df["max_fanout"].fillna(estimated_max).astype(int)
    )

    return canonical_df


def validate_metadata_file(metadata_path):
    """
    Validate that a metadata CSV file follows the canonical format.
    """
    try:
        df = pd.read_csv(metadata_path)
        expected_columns = CANONICAL_HEADER.split(",")

        if list(df.columns) != expected_columns:
            return (
                False,
                f"Column mismatch. Expected: {expected_columns}, Got: {list(df.columns)}",
            )

        # Basic data validation
        required_numeric = ["nodes", "edges", "num_PI", "num_PO", "depth", "max_fanout"]
        for col in required_numeric:
            if not pd.api.types.is_numeric_dtype(df[col]):
                return False, f"Column {col} should be numeric"

        return True, "Valid"

    except (FileNotFoundError, pd.errors.EmptyDataError, ValueError) as e:
        return False, f"Error reading file: {e}"


def create_dataset_summary(full_dataset_path):
    """
    Create a summary of the collected metadata.
    """
    metadata_dir = os.path.join(full_dataset_path, "metadata", "stats")
    summary = {
        "timestamp": datetime.now().isoformat(),
        "totals": {"designs": 0, "files": 0, "random_designs": 0, "openabc_designs": 0},
        "designs": {},
    }

    for csv_file in glob.glob(os.path.join(metadata_dir, "*.csv")):
        design = os.path.splitext(os.path.basename(csv_file))[0]

        if design == "dataset_summary":
            continue

        try:
            df = pd.read_csv(csv_file)
            file_count = len(df)

            summary["designs"][design] = {
                "file_count": file_count,
                "source": "random" if design in RANDOM_DESIGNS else "openabc",
            }

            summary["totals"]["files"] += file_count
            summary["totals"]["designs"] += 1

            if design in RANDOM_DESIGNS:
                summary["totals"]["random_designs"] += 1
            else:
                summary["totals"]["openabc_designs"] += 1

        except (FileNotFoundError, pd.errors.EmptyDataError, ValueError) as e:
            print(f"Warning: Could not process {csv_file}: {e}")

    # Save summary
    summary_path = os.path.join(metadata_dir, "dataset_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nDataset Summary:")
    print(f"  Total designs: {summary['totals']['designs']}")
    print(f"  Random designs: {summary['totals']['random_designs']}")
    print(f"  OpenABC designs: {summary['totals']['openabc_designs']}")
    print(f"  Total metadata entries: {summary['totals']['files']}")
    print(f"  Summary saved to: {summary_path}")

    return summary


def process_design_metadata(design, sources, full_dataset_path):
    """
    Process metadata for a single design.
    Returns: (design, success, message)
    """
    if design in RANDOM_DESIGNS and sources["random"]:
        for random_source in sources["random"]:
            if collect_random_metadata(random_source, full_dataset_path, design):
                return design, True, "random"
        return design, False, "Random metadata not found"

    if design in OPENABC_DESIGNS and sources["openabc"]:
        for openabc_source in sources["openabc"]:
            if collect_openabc_metadata(openabc_source, full_dataset_path, design):
                return design, True, "openabc"
        return design, False, "OpenABC metadata not found"

    return design, False, "No source found"


def main():
    parser = argparse.ArgumentParser(
        description="Collect and organize metadata for Full AIG Dataset"
    )
    parser.add_argument("full_dataset_path", help="Path to FULL_DATASET directory")
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of parallel workers"
    )
    parser.add_argument("--design", help="Process only specific design")
    parser.add_argument(
        "--validate", action="store_true", help="Validate generated CSV files"
    )
    parser.add_argument(
        "--summary", action="store_true", help="Generate dataset summary"
    )
    parser.add_argument("--lib", help="Library file (for compatibility)")
    parser.add_argument(
        "--random-source", help="Override Random AIG dataset source path"
    )
    parser.add_argument(
        "--openabc-source", help="Override OpenABC-D dataset source path"
    )
    parser.add_argument(
        "--source-scope",
        choices=["all", "random", "openabc"],
        default="all",
        help="Limit processing to a dataset source type",
    )

    args = parser.parse_args()

    if not os.path.exists(args.full_dataset_path):
        print(f"Full dataset path does not exist: {args.full_dataset_path}")
        sys.exit(1)

    metadata_dir = os.path.join(args.full_dataset_path, "metadata", "stats")
    os.makedirs(metadata_dir, exist_ok=True)

    print("Collecting metadata for Full AIG Dataset")
    print(f"Output directory: {metadata_dir}")

    # Find source datasets
    if args.random_source or args.openabc_source:
        sources = {
            "random": [args.random_source] if args.random_source else [],
            "openabc": [args.openabc_source] if args.openabc_source else [],
        }
    else:
        print("Searching for source datasets...")
        sources = find_source_datasets(args.full_dataset_path)

    print(f"Found Random AIG sources: {sources['random']}")
    print(f"Found OpenABC-D sources: {sources['openabc']}")

    # Process designs
    if args.design:
        designs_to_process = [args.design]
    elif args.source_scope == "random":
        designs_to_process = RANDOM_DESIGNS
    elif args.source_scope == "openabc":
        designs_to_process = OPENABC_DESIGNS
    else:
        designs_to_process = ALL_DESIGNS

    if args.source_scope == "random" and not sources["random"]:
        print("Error: --source-scope random requires a Random AIG source")
        sys.exit(1)
    if args.source_scope == "openabc" and not sources["openabc"]:
        print("Error: --source-scope openabc requires an OpenABC-D source")
        sys.exit(1)

    success_count = 0
    total_count = len(designs_to_process)

    if args.workers > 1 and len(designs_to_process) > 1:
        worker_count = min(args.workers, len(designs_to_process))
        print(f"\nProcessing designs in parallel with {worker_count} workers...")

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(
                    process_design_metadata,
                    design,
                    sources,
                    args.full_dataset_path,
                ): design
                for design in designs_to_process
            }

            for future in as_completed(futures):
                design = futures[future]
                try:
                    _, success, message = future.result()
                    if success:
                        success_count += 1
                        print(f"  ✓ {design}: processed ({message})")
                    else:
                        print(f"  ✗ {design}: {message}")
                except (OSError, ValueError) as exc:
                    print(f"  ✗ {design}: unexpected error - {exc}")
    else:
        for design in designs_to_process:
            print(f"\nProcessing {design}...")
            _, success, message = process_design_metadata(
                design,
                sources,
                args.full_dataset_path,
            )
            if success:
                success_count += 1
            else:
                print(f"  {message} for {design}")

    # Validation
    if args.validate:
        print("\nValidating metadata files...")
        validation_errors = []

        for csv_file in glob.glob(os.path.join(metadata_dir, "*.csv")):
            if "dataset_summary" in csv_file:
                continue

            design = os.path.splitext(os.path.basename(csv_file))[0]
            is_valid, message = validate_metadata_file(csv_file)

            if is_valid:
                print(f"  ✓ {design}.csv: {message}")
            else:
                print(f"  ✗ {design}.csv: {message}")
                validation_errors.append(f"{design}: {message}")

        if validation_errors:
            print("\nValidation errors found:")
            for error in validation_errors:
                print(f"  - {error}")

    # Generate summary
    if args.summary:
        create_dataset_summary(args.full_dataset_path)

    print(f"\n{'=' * 60}")
    print("METADATA COLLECTION COMPLETE")
    print(f"{'=' * 60}")
    print(f"Successfully processed: {success_count}/{total_count} designs")
    print(f"Metadata files saved to: {metadata_dir}")

    if success_count < total_count:
        print(f"Warning: {total_count - success_count} designs could not be processed")
        print("Check that source dataset paths are correct and contain metadata files")


if __name__ == "__main__":
    main()
