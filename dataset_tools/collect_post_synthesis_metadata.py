#!/usr/bin/env python3
"""
Post-synthesis metadata collector for OpenABC dataset.
Analyzes synthesis log files and AIG files to extract circuit statistics.
"""

import argparse
import csv
import os
import re
import subprocess
import zipfile
from typing import Any, Dict, List, Tuple


def check_log_for_errors(log_content: str) -> Tuple[List[str], List[str]]:
    """Analyze log content for common ABC/synthesis errors."""
    errors: List[str] = []
    warnings: List[str] = []

    # Common error patterns
    error_patterns = [
        (r"\*\* cmd error:", "ABC command error"),
        (r"Error:", "General error"),
        (r"ERROR:", "System error"),
        (r"Failed to", "Operation failure"),
        (r"Cannot read", "File read error"),
        (r"Cannot write", "File write error"),
        (r"aborting", "Script abortion"),
        (r"Segmentation fault", "Segmentation fault"),
        (r"out of memory", "Memory error"),
    ]

    # Warning patterns
    warning_patterns = [
        (r"Warning:", "General warning"),
        (r"WARNING:", "System warning"),
        (r"zip error:", "Zip operation warning"),
        (r"Nothing to do!", "Empty zip warning"),
    ]

    # Check for errors
    for pattern, error_type in error_patterns:
        matches = re.findall(pattern, log_content, re.IGNORECASE)
        if matches:
            errors.append(f"{error_type}: {len(matches)} occurrences")

    # Check for warnings
    for pattern, warning_type in warning_patterns:
        matches = re.findall(pattern, log_content, re.IGNORECASE)
        if matches:
            warnings.append(f"{warning_type}: {len(matches)} occurrences")

    return errors, warnings


def parse_abc_stats_from_log(log_content: str) -> List[Dict[str, Any]]:
    """Extract ABC statistics from synthesis log content."""
    # Support multiple ABC print formats, e.g.:
    #   i/o = 128/398  nd = 984  edge = 2048  lev = 12
    #   i/o = 128/398  lat = 0    and = 984    lev = 12
    stats_pattern = re.compile(
        r"i/o\s*=\s*(\d+)\s*/\s*(\d+).*?(?:nd|and)\s*=\s*(\d+)(?:.*?edge\s*=\s*(\d+))?.*?lev\s*=\s*(\d+)",
        re.IGNORECASE,
    )

    matches = list(stats_pattern.finditer(log_content))
    if not matches:
        return []

    # Each match is (PI, PO, nodes, edges?, levels)
    parsed_stats: List[Dict[str, Any]] = []
    for match in matches:
        pi = int(match.group(1))
        po = int(match.group(2))
        nodes = int(match.group(3))
        edges = int(match.group(4)) if match.group(4) is not None else nodes * 2
        levels = int(match.group(5))
        # Calculate approximate fanout statistics
        avg_fanout = round(edges / nodes, 2) if nodes > 0 else 0
        max_fanout = max(10, int(avg_fanout * 1.5))  # Estimated max fanout

        stats = {
            "nodes": nodes,
            "edges": edges,
            "num_PI": pi,
            "num_PO": po,
            "depth": levels,
            "avg_fanout": avg_fanout,
            "max_fanout": max_fanout,
        }
        parsed_stats.append(stats)

    return parsed_stats


def extract_aig_stats_from_file(aig_file_path: str) -> Dict[str, Any]:
    """Extract basic statistics from an AIG file using ABC or aigverse."""

    # Method 1: Try ABC first
    try:
        cmd = f'abc -c "read {aig_file_path}; print_stats"'
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=30, check=False
        )

        if result.returncode == 0:
            output = result.stdout
            # Parse ABC print_stats output across formats (nd/edge or lat/and)
            stats_match = re.search(
                r"i/o\s*=\s*(\d+)\s*/\s*(\d+).*?(?:nd|and)\s*=\s*(\d+)(?:.*?edge\s*=\s*(\d+))?.*?lev\s*=\s*(\d+)",
                output,
                re.IGNORECASE,
            )
            if stats_match:
                pi = int(stats_match.group(1))
                po = int(stats_match.group(2))
                nodes = int(stats_match.group(3))
                edges = (
                    int(stats_match.group(4))
                    if stats_match.group(4) is not None
                    else nodes * 2
                )
                levels = int(stats_match.group(5))
                avg_fanout = round(edges / nodes, 2) if nodes > 0 else 0
                max_fanout = max(10, int(avg_fanout * 1.5))  # Estimated max fanout

                return {
                    "nodes": nodes,
                    "edges": edges,
                    "num_PI": pi,
                    "num_PO": po,
                    "depth": levels,
                    "avg_fanout": avg_fanout,
                    "max_fanout": max_fanout,
                }
    except (
        subprocess.TimeoutExpired,
        subprocess.CalledProcessError,
        FileNotFoundError,
    ):
        pass

    # Method 2: Try aigverse as fallback
    try:
        # Check if aigverse is available
        cmd = f"aigverse stats {aig_file_path}"
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=30, check=False
        )

        if result.returncode == 0:
            output = result.stdout
            # Parse aigverse output - adjust regex based on actual aigverse output format
            # This is a placeholder - you may need to adjust based on aigverse's actual output
            stats_match = re.search(
                r"nodes:\s*(\d+).*inputs:\s*(\d+).*outputs:\s*(\d+).*levels:\s*(\d+)",
                output,
                re.DOTALL,
            )
            if stats_match:
                nodes, pi, po, levels = map(int, stats_match.groups())
                # Estimate edges (typical AIG has ~2x edges to nodes ratio)
                edges = nodes * 2
                avg_fanout = round(edges / nodes, 2) if nodes > 0 else 0
                max_fanout = max(10, int(avg_fanout * 1.5))

                return {
                    "nodes": nodes,
                    "edges": edges,
                    "num_PI": pi,
                    "num_PO": po,
                    "depth": levels,
                    "avg_fanout": avg_fanout,
                    "max_fanout": max_fanout,
                }
    except (
        subprocess.TimeoutExpired,
        subprocess.CalledProcessError,
        FileNotFoundError,
    ):
        pass

    # Method 3: Parse AIG file directly (basic header parsing)
    try:
        with open(aig_file_path, "rb") as f:
            # Read first line of AIG file (ASCII header)
            first_line = f.readline().decode("ascii", errors="ignore").strip()
            if first_line.startswith("aig"):
                # AIG format: "aig M I L O A" where M=max_var, I=inputs, L=latches, O=outputs, A=and_gates
                parts = first_line.split()
                if len(parts) >= 6:
                    _, inputs, latches, outputs, and_gates = map(
                        int, parts[1:6]
                    )  # max_var not used
                    nodes = and_gates + inputs + latches  # Total nodes
                    edges = and_gates * 2  # Each AND gate has 2 inputs
                    levels = max(1, nodes // 10)  # Rough estimate
                    avg_fanout = round(edges / nodes, 2) if nodes > 0 else 0
                    max_fanout = max(10, int(avg_fanout * 1.5))

                    return {
                        "nodes": nodes,
                        "edges": edges,
                        "num_PI": inputs,
                        "num_PO": outputs,
                        "depth": levels,
                        "avg_fanout": avg_fanout,
                        "max_fanout": max_fanout,
                    }
    except (IOError, ValueError, UnicodeDecodeError):
        pass

    # If all methods fail, return empty dict
    print(f"Warning: Could not extract statistics from {aig_file_path}")
    return {}


def collect_metadata_for_design(design: str, base_dir: str) -> int:
    """Collect metadata for a specific design from log files and zip files."""
    print(f"Collecting metadata for design {design}...")

    bench_dir = os.path.join(base_dir, "OPENABC_DATASET", "bench", design)
    log_dir = os.path.join(bench_dir, f"log_{design}")
    metadata_dir = os.path.join(bench_dir, "metadata")

    # Ensure metadata directory exists
    os.makedirs(metadata_dir, exist_ok=True)

    # CSV file for this design
    csv_file = os.path.join(metadata_dir, f"{design}.csv")

    # CSV header
    header = "file_path,design,recipe_id,step_id,tier_id,algorithm,nodes,edges,num_PI,num_PO,depth,avg_fanout,max_fanout"

    metadata_rows: List[List[Any]] = []
    processed_recipes = 0
    error_summary: Dict[str, int] = {
        "total_errors": 0,
        "total_warnings": 0,
        "recipes_with_errors": 0,
        "recipes_with_warnings": 0,
    }
    error_details: List[str] = []

    # Process each log file

    # First, handle the original AIG file if it exists
    orig_aig_file = os.path.join(bench_dir, f"{design}_orig.aig")
    if os.path.exists(orig_aig_file):
        print(f"  Processing original AIG: {design}_orig.aig")
        # Extract stats from original AIG file
        orig_stats = extract_aig_stats_from_file(orig_aig_file)

        # Add row for original AIG - include even if stats extraction failed
        file_path = f"{design}/{design}_orig.aig"
        orig_row = [
            file_path,
            design,
            "",  # No recipe_id for original
            "",  # No step_id for original
            "",  # No tier_id for original
            "",  # No algorithm for base AIGs
            orig_stats.get("nodes", 0),
            orig_stats.get("edges", 0),
            orig_stats.get("num_PI", 0),
            orig_stats.get("num_PO", 0),
            orig_stats.get("depth", 0),
            orig_stats.get("avg_fanout", 0.0),
            orig_stats.get("max_fanout", 0),
        ]
        metadata_rows.append(orig_row)

        if not orig_stats:
            print(f"    Warning: Using zero values for stats from {orig_aig_file}")
        else:
            print(f"    ✓ Successfully extracted stats from {orig_aig_file}")
    else:
        print(f"  Warning: Original AIG file not found: {orig_aig_file}")
        # Still add a row with zero values to maintain consistency
        file_path = f"{design}/{design}_orig.aig"
        orig_row = [file_path, design, "", "", "", "", 0, 0, 0, 0, 0, 0.0, 0]
        metadata_rows.append(orig_row)

    # Debug: Check what files actually exist
    print(f"  Checking file structure in {bench_dir}...")

    # Check for ZIP files
    zip_files = [f for f in os.listdir(bench_dir) if f.endswith(".zip")]
    print(f"  Found {len(zip_files)} ZIP files")
    if len(zip_files) > 0:
        # Show sample ZIP filenames to understand naming pattern
        sample_zips = sorted(zip_files)[:5]
        print(f"  Sample ZIP files: {sample_zips}")

        # Extract recipe IDs from ZIP filenames
        zip_recipe_ids = []
        for zip_file in zip_files:
            match = re.search(r"syn(\d+)\.zip", zip_file)
            if match:
                zip_recipe_ids.append(int(match.group(1)))

        if zip_recipe_ids:
            zip_recipe_ids.sort()
            print(
                f"  Recipe ID range in ZIP files: {min(zip_recipe_ids)} to {max(zip_recipe_ids)}"
            )
            print("  Expected range: 0 to 1499")

            # Check if we need to adjust the range
            if min(zip_recipe_ids) > 0:
                print(f"  ⚠️  ZIP files start from {min(zip_recipe_ids)}, not 0!")

    # Check for log files
    if os.path.exists(log_dir):
        log_files = [f for f in os.listdir(log_dir) if f.endswith(".log")]
        print(f"  Found {len(log_files)} LOG files in {log_dir}")
        if len(log_files) > 0:
            sample_logs = sorted(log_files)[:5]
            print(f"  Sample LOG files: {sample_logs}")
    else:
        print(f"  ⚠️  Log directory does not exist: {log_dir}")

    print("  Processing recipes...")

    for recipe_id in range(1500):  # 1500 synthesis recipes (0-1499)
        log_file = os.path.join(log_dir, f"log_{design}_syn{recipe_id}.log")
        zip_file = os.path.join(bench_dir, f"syn{recipe_id}.zip")

        # Debug: print every 100th recipe to see progress
        if recipe_id % 500 == 0:
            print(
                f"  Checking recipe {recipe_id}: log={os.path.exists(log_file)}, zip={os.path.exists(zip_file)}"
            )

        if not os.path.exists(log_file):
            if recipe_id < 10:  # Only warn for first few missing files to avoid spam
                print(f"Warning: Log file missing for recipe {recipe_id}: {log_file}")
            continue

        if not os.path.exists(zip_file):
            if recipe_id < 10:  # Only warn for first few missing files to avoid spam
                print(f"Warning: Zip file missing for recipe {recipe_id}: {zip_file}")
            continue

        try:
            # Read log file content
            with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
                log_content = f.read()

            # Check for errors and warnings in log
            errors, warnings = check_log_for_errors(log_content)
            if errors:
                error_summary["recipes_with_errors"] += 1
                error_summary["total_errors"] += len(errors)
                error_details.append(f"Recipe {recipe_id}: {', '.join(errors)}")
            if warnings:
                error_summary["recipes_with_warnings"] += 1
                error_summary["total_warnings"] += len(warnings)
                # Only log significant warnings (not routine zip warnings)
                significant_warnings = [
                    w for w in warnings if "zip operation" not in w.lower()
                ]
                if significant_warnings:
                    error_details.append(
                        f"Recipe {recipe_id} warnings: {', '.join(significant_warnings)}"
                    )

            # Extract all statistics from log (one per synthesis step)
            stats_list = parse_abc_stats_from_log(log_content)

            # Get list of AIG files from zip to match with statistics
            try:
                with zipfile.ZipFile(zip_file, "r") as zf:
                    aig_files = [
                        name for name in zf.namelist() if name.endswith(".aig")
                    ]
                    aig_files.sort()  # Ensure consistent ordering
            except zipfile.BadZipFile:
                print(f"Warning: Invalid zip file for recipe {recipe_id}")
                continue

            # Match statistics with AIG files
            for step_id, (aig_file, stats) in enumerate(zip(aig_files, stats_list), 1):
                if stats:  # Only add if we have valid statistics
                    # Extract step number from filename
                    step_match = re.search(r"step(\d+)\.aig", aig_file)
                    actual_step = (
                        int(step_match.group(1)) if step_match is not None else step_id
                    )

                    # Create file path for current structure (zip-based)
                    file_path = f"{design}/syn{recipe_id}.zip/{aig_file}"

                    # For base AIGs (synthesis results), tier_id should be empty according to README
                    # tier_id is only for algorithm outputs (Orchestrate, Deepsyn, etc.)
                    tier_id = ""  # Empty for base AIGs

                    row = [
                        file_path,
                        design,
                        recipe_id,
                        actual_step,
                        tier_id,
                        "",
                        stats["nodes"],
                        stats["edges"],
                        stats["num_PI"],
                        stats["num_PO"],
                        stats["depth"],
                        stats["avg_fanout"],
                        stats["max_fanout"],
                    ]
                    metadata_rows.append(row)

            processed_recipes += 1
            if processed_recipes % 100 == 0:
                print(f"  Processed {processed_recipes}/1500 recipes")

        except (IOError, OSError, zipfile.BadZipFile, ValueError) as e:
            print(f"Error processing recipe {recipe_id}: {e}")
            continue

    # Write metadata CSV file
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        f.write(header + "\n")
        writer = csv.writer(f)
        writer.writerows(metadata_rows)

    # Write error summary file
    error_file = os.path.join(metadata_dir, f"{design}_errors.log")
    with open(error_file, "w", encoding="utf-8") as f:
        f.write(f"Error Summary for Design {design}\n")
        f.write("=" * 40 + "\n")
        f.write(f"Total recipes processed: {processed_recipes}\n")
        f.write(f"Recipes with errors: {error_summary['recipes_with_errors']}\n")
        f.write(f"Recipes with warnings: {error_summary['recipes_with_warnings']}\n")
        f.write(f"Total error occurrences: {error_summary['total_errors']}\n")
        f.write(f"Total warning occurrences: {error_summary['total_warnings']}\n")
        f.write("\n")

        if error_details:
            f.write("Detailed Error/Warning Log:\n")
            f.write("-" * 30 + "\n")
            for detail in error_details:
                f.write(f"{detail}\n")
        else:
            f.write("No errors or significant warnings found!\n")

    print(
        f"✓ Collected metadata for {processed_recipes} recipes ({len(metadata_rows)} entries)"
    )
    print(f"  Metadata saved to: {csv_file}")
    print(f"  Error log saved to: {error_file}")

    # Print error summary
    if error_summary["recipes_with_errors"] > 0:
        print(f"  ⚠️  {error_summary['recipes_with_errors']} recipes had errors")
    if error_summary["recipes_with_warnings"] > 0:
        print(f"  ⚠️  {error_summary['recipes_with_warnings']} recipes had warnings")
    if error_summary["total_errors"] == 0 and error_summary["total_warnings"] == 0:
        print("  ✓ No errors or warnings found!")

    return len(metadata_rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect post-synthesis metadata from log files"
    )
    parser.add_argument("--home", required=True, help="Base directory path")
    parser.add_argument(
        "--design",
        help="Specific design to process (e.g., 128). If not provided, processes all designs",
    )

    args = parser.parse_args()

    designs: List[str] = ["128", "256", "512", "1024", "2048", "4096", "8192", "16384"]

    if args.design:
        if args.design not in designs:
            print(f"Error: Invalid design {args.design}. Valid designs: {designs}")
            return
        designs = [args.design]

    total_entries: int = 0

    print("=" * 50)
    print("POST-SYNTHESIS METADATA COLLECTION")
    print("=" * 50)

    for design in designs:
        entries = collect_metadata_for_design(design, args.home)
        total_entries += entries
        print()

    print(f"✓ Total metadata entries collected: {total_entries}")
    print("Metadata collection complete!")


if __name__ == "__main__":
    main()
