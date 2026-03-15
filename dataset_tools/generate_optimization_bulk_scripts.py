#!/usr/bin/env python3
"""
Generate per-design optimization runner scripts for FULL_DATASET.

Artifacts are generated per:
- algorithm
- design
- input tier (base_aigs, tier1, tier2)

Scripts are stored as zipped bundles per design under:
    FULL_DATASET/synScripts/optimization/{algorithm}/{design}.zip
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List

ALGORITHMS = ["Orchestrate", "Deepsyn", "Syn4", "C2RS"]
RANDOM_AIG_DESIGNS = ["128", "256", "512", "1024", "2048", "4096", "8192", "16384"]
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
INPUT_SOURCES = ["base_aigs", "tier1", "tier2"]
INPUT_SOURCE_TO_OUTPUT = {
    "base_aigs": "tier1",
    "tier1": "tier2",
    "tier2": "tier3",
}
INPUT_SOURCE_TO_LABEL = {
    "base_aigs": "base",
    "tier1": "tier1",
    "tier2": "tier2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate optimization bulk scripts")
    parser.add_argument(
        "--base-dir",
        default=os.path.expanduser("~/data-gen-rand-abcd"),
        help="Project base directory",
    )
    parser.add_argument(
        "--full-dataset",
        default=f"/scratch-shared/{os.environ.get('USER', 'USER')}/FULL_DATASET",
        help="Path to FULL_DATASET",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to optimization config JSON (default: <base-dir>/dataset_tools/optimization_config.json)",
    )
    parser.add_argument(
        "--design-group",
        choices=["all", "random", "openabc"],
        default="all",
        help="Design subset to target (default: all)",
    )
    parser.add_argument(
        "--designs",
        nargs="+",
        default=None,
        help=(
            "Explicit design names to include. Accepts space-separated and/or comma-separated values "
            "(e.g., --designs 128 256 or --designs 128,256)."
        ),
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=["all"],
        help=(
            "Algorithms to include. Accepts 'all' or a space/comma-separated list "
            "of Orchestrate, Deepsyn, Syn4, C2RS."
        ),
    )
    parser.add_argument(
        "--input-source",
        choices=["all", "base_aigs", "tier1", "tier2"],
        default="base_aigs",
        help="Input source(s) to generate for (default: base_aigs)",
    )
    return parser.parse_args()


def load_config(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    missing = [
        algorithm
        for algorithm in ALGORITHMS
        if algorithm not in data.get("algorithms", {})
    ]
    if missing:
        raise ValueError(f"Missing algorithm configs: {missing}")
    return data


def discover_designs(base_aigs_dir: Path) -> List[str]:
    if not base_aigs_dir.exists():
        raise FileNotFoundError(f"base_aigs directory not found: {base_aigs_dir}")
    designs = sorted([path.name for path in base_aigs_dir.iterdir() if path.is_dir()])
    if not designs:
        raise ValueError(f"No design directories found in: {base_aigs_dir}")
    return designs


def normalize_inputs(raw_values: List[str] | None) -> List[str]:
    if not raw_values:
        return []
    normalized: List[str] = []
    for token in raw_values:
        for piece in token.split(","):
            value = piece.strip()
            if value:
                normalized.append(value)
    return normalized


def select_algorithms(requested_algorithms: List[str]) -> List[str]:
    if not requested_algorithms:
        return list(ALGORITHMS)

    lowered = [item.lower() for item in requested_algorithms]
    if len(lowered) == 1 and lowered[0] == "all":
        return list(ALGORITHMS)

    selected: List[str] = []
    for algorithm in requested_algorithms:
        if algorithm not in ALGORITHMS:
            raise ValueError(
                "Invalid algorithm selection: "
                f"{algorithm}. Allowed values: {', '.join(ALGORITHMS)} or 'all'."
            )
        if algorithm not in selected:
            selected.append(algorithm)

    if not selected:
        raise ValueError("No algorithms selected.")
    return selected


def select_input_sources(requested_input_source: str) -> List[str]:
    if requested_input_source == "all":
        return list(INPUT_SOURCES)
    return [requested_input_source]


def select_designs(
    available_designs: List[str],
    design_group: str,
    explicit_designs: List[str],
) -> List[str]:
    available = set(available_designs)
    group_candidates = {
        "random": set(RANDOM_AIG_DESIGNS),
        "openabc": set(OPENABC_DESIGNS),
        "all": available,
    }
    selected = available.intersection(group_candidates[design_group])

    if explicit_designs:
        requested = set(explicit_designs)
        missing = sorted(requested.difference(available))
        if missing:
            raise ValueError(
                "Requested designs not found in base_aigs: " + ", ".join(missing)
            )
        selected = selected.intersection(requested)

    if not selected:
        raise ValueError(
            "No designs selected after applying filters. "
            f"design_group={design_group}, explicit_designs={explicit_designs or '[]'}"
        )
    return sorted(selected)


def shell_quote_single(value: str) -> str:
    return value.replace("'", "'\"'\"'")


def render_shard_script(
    algorithm: str,
    design: str,
    full_dataset: str,
    abc_rc: str,
    input_source: str,
    algo_cfg: Dict,
    timeout_seconds: int | None,
) -> str:
    command_template = str(algo_cfg.get("command_template", "")).strip()
    if not command_template:
        raise ValueError(f"Missing command_template for algorithm {algorithm}")

    command_template_escaped = shell_quote_single(command_template)

    output_tier = INPUT_SOURCE_TO_OUTPUT[input_source]
    input_label = INPUT_SOURCE_TO_LABEL[input_source]
    input_root = (
        "base_aigs"
        if input_source == "base_aigs"
        else f"optimized_aigs/${{ALGORITHM}}/{input_source}"
    )

    timeout_seconds_value = "" if timeout_seconds is None else str(timeout_seconds)

    return f"""#!/bin/bash
set -euo pipefail

ALGORITHM="{algorithm}"
DESIGN="{design}"
FULL_DATASET="{full_dataset}"
ABC_RC="{abc_rc}"
INPUT_SOURCE="{input_source}"
OUTPUT_TIER="{output_tier}"
INPUT_LABEL="{input_label}"
RUNTIME_TIMEOUT_SECONDS="{timeout_seconds_value}"

COMMAND_TEMPLATE='{command_template_escaped}'

INPUT_ROOT="{input_root}"
OUTPUT_ROOT="optimized_aigs/${{ALGORITHM}}/${{OUTPUT_TIER}}"
RAW_LOG_ROOT="metadata/raw_logs/${{DESIGN}}/${{OUTPUT_TIER}}/${{ALGORITHM}}"

in_dir="${{FULL_DATASET}}/${{INPUT_ROOT}}/${{DESIGN}}"
out_tier_dir="${{FULL_DATASET}}/${{OUTPUT_ROOT}}"
output_zip_path="${{out_tier_dir}}/${{DESIGN}}.zip"
log_dir="${{FULL_DATASET}}/${{RAW_LOG_ROOT}}"
in_zip="${{FULL_DATASET}}/${{INPUT_ROOT}}/${{DESIGN}}.zip"

mkdir -p "$out_tier_dir" "$log_dir"

if [[ "$INPUT_SOURCE" == "base_aigs" ]]; then
    if [[ ! -d "$in_dir" ]]; then
        echo "✗ Missing design input directory: $in_dir" >&2
        exit 1
    fi
else
    if [[ ! -f "$in_zip" && ! -d "$in_dir" ]]; then
        echo "✗ Missing design input zip/dir for input_source=$INPUT_SOURCE: zip=$in_zip dir=$in_dir" >&2
        exit 1
    fi
fi

# ENFORCE LOCAL SCRATCH NODE SPACE
# Slurm explicitly sets TMPDIR to /scratch-node/<user>.<jobid> on Snellius. 
# We fallback to /tmp (which is also local to the node) if it is somehow missing.
LOCAL_SCRATCH="${{TMPDIR:-/tmp}}"
mkdir -p "$LOCAL_SCRATCH"

# Create our working directory purely on the node's local NVMe scratch disk
tmp_extract_root="$(mktemp -d "$LOCAL_SCRATCH/opt_${{ALGORITHM}}_${{DESIGN}}_${{INPUT_LABEL}}_XXXXXX")"
trap 'rm -rf "$tmp_extract_root"' EXIT

WORKERS="${{OPT_SCRIPT_PARALLELISM:-${{SLURM_CPUS_PER_TASK:-192}}}}"

if ! [[ "$WORKERS" =~ ^[0-9]+$ ]] || [[ "$WORKERS" -le 0 ]]; then
    echo "✗ Invalid worker count: $WORKERS" >&2
    exit 1
fi

# Both input processing and output processing happen entirely on local scratch
INPUT_TMP_DIR="$tmp_extract_root/in"
OUT_DIR="$tmp_extract_root/out"
mkdir -p "$INPUT_TMP_DIR" "$OUT_DIR"

echo "Staging inputs to local scratch node space..."

# Pre-stage all inputs into local scratch to eliminate parallel I/O bottlenecks
if [[ "$INPUT_SOURCE" == "base_aigs" ]]; then
    find "$in_dir" -maxdepth 1 -name "*.aig" -exec ln -s {{}} "$INPUT_TMP_DIR/" \\;
    for z in "$in_dir"/syn*.zip; do
        [[ -f "$z" ]] || continue
        zname="$(basename "$z")"
        mkdir -p "$INPUT_TMP_DIR/$zname"
        unzip -q -j "$z" -d "$INPUT_TMP_DIR/$zname/"
        for f in "$INPUT_TMP_DIR/$zname"/*.aig; do
            [[ -f "$f" ]] || continue
            mv "$f" "$INPUT_TMP_DIR/${{zname}}__$(basename "$f")"
        done
        rm -rf "$INPUT_TMP_DIR/$zname"
    done
else
    if [[ -f "$in_zip" ]]; then
        zname="$(basename "$in_zip")"
        mkdir -p "$INPUT_TMP_DIR/$zname"
        unzip -q -j "$in_zip" -d "$INPUT_TMP_DIR/$zname/"
        for f in "$INPUT_TMP_DIR/$zname"/*.aig; do
            [[ -f "$f" ]] || continue
            mv "$f" "$INPUT_TMP_DIR/${{zname}}__$(basename "$f")"
        done
        rm -rf "$INPUT_TMP_DIR/$zname"
    fi
    if [[ -d "$in_dir" ]]; then
        find "$in_dir" -maxdepth 1 -name "*.aig" -exec ln -s {{}} "$INPUT_TMP_DIR/" \\;
        for z in "$in_dir"/*.zip; do
            [[ -f "$z" ]] || continue
            zname="$(basename "$z")"
            mkdir -p "$INPUT_TMP_DIR/$zname"
            unzip -q -j "$z" -d "$INPUT_TMP_DIR/$zname/"
            for f in "$INPUT_TMP_DIR/$zname"/*.aig; do
                [[ -f "$f" ]] || continue
                mv "$f" "$INPUT_TMP_DIR/${{zname}}__$(basename "$f")"
            done
            rm -rf "$INPUT_TMP_DIR/$zname"
        done
    fi
fi

discovered="$(find "$INPUT_TMP_DIR" -maxdepth 1 -name "*.aig" | wc -l | tr -d ' ')"

if [[ $discovered -eq 0 ]]; then
    echo "✗ No input AIGs discovered under: $in_dir (input_source=$INPUT_SOURCE)" >&2
    exit 1
fi

echo "Processing $discovered files with $WORKERS concurrent workers..."

export COMMAND_TEMPLATE
export ABC_RC
export RUNTIME_TIMEOUT_SECONDS
export OUT_DIR

run_one() {{
    local input_aig="$1"
    local output_member="$(basename "$input_aig")"
    if [[ "$output_member" != *".zip__"* ]]; then
        output_member="plain__${{output_member}}"
    fi
    
    local output_tmp="${{OUT_DIR}}/tmp_${{RANDOM}}_${{BASHPID}}.aig"
    local seed_value="42"

    local cmd="${{COMMAND_TEMPLATE//\\{{input_aig\\}}/$input_aig}}"
    cmd="${{cmd//\\{{output_aig\\}}/$output_tmp}}"
    cmd="${{cmd//\\{{abc_rc\\}}/$ABC_RC}}"
    cmd="${{cmd//\\{{seed\\}}/$seed_value}}"
    cmd="${{cmd//\\{{timeout_seconds\\}}/$RUNTIME_TIMEOUT_SECONDS}}"

    if eval "$cmd" >/dev/null 2>&1; then
        if [[ ! -f "$output_tmp" ]]; then
            return 1
        fi
        
        local arcname="$output_member"
        local stem="${{arcname%.*}}"
        local ext="${{arcname##*.}}"
        if [[ "$stem" == "$arcname" ]]; then ext="aig"; fi
        
        local idx=1
        local final_path="${{OUT_DIR}}/$arcname"
        
        # Atomic lock-free filesystem write check for instantaneous deduplication
        set -o noclobber
        while ! {{ > "$final_path" ; }} 2>/dev/null; do
            final_path="${{OUT_DIR}}/${{stem}}__dup${{idx}}.${{ext}}"
            ((idx++))
        done
        set +o noclobber
        
        mv -f "$output_tmp" "$final_path"
        return 0
    fi
    rm -f "$output_tmp"
    return 1
}}
export -f run_one

# Use xargs to absolutely blast through the queue with minimal overhead
find "$INPUT_TMP_DIR" -maxdepth 1 -name "*.aig" -print0 | \\
    xargs -0 -P "$WORKERS" -n 1 -I {{}} bash -c 'run_one "$1" || echo "FAIL"' _ {{}} > "$tmp_extract_root/failures.log"

failed=$(grep -c "FAIL" "$tmp_extract_root/failures.log" || true)
created=$(find "$OUT_DIR" -maxdepth 1 -type f -name "*.aig" | wc -l | tr -d ' ')
processed=$((created + failed))

echo "Zipping $created outputs to $output_zip_path..."
if [[ $created -gt 0 ]]; then
    mkdir -p "$(dirname "$output_zip_path")"
    rm -f "$output_zip_path"
    # Stream the zip directly from local scratch into the designated full dataset folder
    if ! (cd "$OUT_DIR" && zip -q -r -1 "$output_zip_path" .); then
        echo "✗ Failed to create final zip: $output_zip_path" >&2
        failed=$((failed + 1))
    fi
fi

summary_file="$log_dir/summary.json"
cat > "$summary_file" <<EOF
{{
    "algorithm": "$ALGORITHM",
    "design": "$DESIGN",
    "tier": "$OUTPUT_TIER",
    "discovered": $discovered,
    "processed": $processed,
    "created": $created,
    "failed": $failed,
    "completed_at": "$(date -Iseconds)"
}}
EOF

if [[ $failed -gt 0 ]]; then
    echo "✗ Completed with $failed failures." >&2
    exit 1
fi
"""


def write_design_zip(
    zip_path: Path,
    script_contents: Dict[str, str],
) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"opt_scripts_{zip_path.stem}_") as tmp_dir:
        tmp_root = Path(tmp_dir)
        for script_name, content in script_contents.items():
            script_path = tmp_root / script_name
            with script_path.open("w", encoding="utf-8") as handle:
                handle.write(content)
            script_path.chmod(0o755)

        if zip_path.exists():
            zip_path.unlink()

        with zipfile.ZipFile(
            zip_path, "w", compression=zipfile.ZIP_DEFLATED
        ) as archive:
            for script_path in sorted(tmp_root.glob("*.sh")):
                archive.write(script_path, arcname=script_path.name)


def main() -> None:
    args = parse_args()

    base_dir = Path(args.base_dir).expanduser().resolve()
    full_dataset = Path(args.full_dataset).expanduser().resolve()
    config_path = (
        Path(args.config).expanduser().resolve()
        if args.config
        else (base_dir / "dataset_tools" / "optimization_config.json")
    )

    base_aigs_dir = full_dataset / "base_aigs"
    opt_root = full_dataset / "optimized_aigs"
    synscripts_opt_dir = full_dataset / "synScripts" / "optimization"
    manifests_dir = opt_root / "manifests"
    runtime_config_dir = opt_root / "config"
    runtime_config_path = runtime_config_dir / "optimization_config.json"
    abc_rc = base_dir / "abc.rc"

    config = load_config(config_path)
    runtime_cfg = config.get("runtime", {})
    default_timeout_seconds = int(runtime_cfg.get("timeout_seconds", 10))
    if default_timeout_seconds <= 0:
        raise ValueError("runtime.timeout_seconds must be a positive integer")

    available_designs = discover_designs(base_aigs_dir)
    requested_designs = normalize_inputs(args.designs)
    requested_algorithms = normalize_inputs(args.algorithms)
    selected_algorithms = select_algorithms(requested_algorithms)
    selected_input_sources = select_input_sources(args.input_source)

    designs = select_designs(
        available_designs=available_designs,
        design_group=args.design_group,
        explicit_designs=requested_designs,
    )

    manifests_dir.mkdir(parents=True, exist_ok=True)
    runtime_config_dir.mkdir(parents=True, exist_ok=True)
    synscripts_opt_dir.mkdir(parents=True, exist_ok=True)

    shutil.copyfile(config_path, runtime_config_path)

    for algorithm in selected_algorithms:
        for input_source in selected_input_sources:
            out_tier = INPUT_SOURCE_TO_OUTPUT[input_source]
            (opt_root / algorithm / out_tier).mkdir(parents=True, exist_ok=True)

    generated_script_count = 0
    generated_zips: List[str] = []

    for algorithm in selected_algorithms:
        algo_zip_dir = synscripts_opt_dir / algorithm
        algo_zip_dir.mkdir(parents=True, exist_ok=True)
        for design in designs:
            scripts_for_design: Dict[str, str] = {}
            algo_cfg = config["algorithms"][algorithm]
            timeout_seconds: int | None = None
            if algorithm == "Deepsyn":
                timeout_seconds = int(
                    algo_cfg.get("timeout_seconds", default_timeout_seconds)
                )
            for input_source in selected_input_sources:
                label = INPUT_SOURCE_TO_LABEL[input_source]
                script_name = f"optimizeBulk_{algorithm}_{design}_{label}.sh"
                scripts_for_design[script_name] = render_shard_script(
                    algorithm=algorithm,
                    design=design,
                    full_dataset=str(full_dataset),
                    abc_rc=str(abc_rc),
                    input_source=input_source,
                    algo_cfg=algo_cfg,
                    timeout_seconds=timeout_seconds,
                )
                generated_script_count += 1

            zip_path = algo_zip_dir / f"{design}.zip"
            write_design_zip(zip_path, scripts_for_design)
            generated_zips.append(str(zip_path))

    manifest_path = manifests_dir / "bulk_scripts_manifest.json"
    manifest = {
        "generated_at": datetime.now().isoformat(),
        "full_dataset": str(full_dataset),
        "config_source": str(config_path),
        "runtime_config_copy": str(runtime_config_path),
        "script_zip_root": str(synscripts_opt_dir),
        "algorithms": selected_algorithms,
        "input_sources": selected_input_sources,
        "design_group": args.design_group,
        "requested_designs": requested_designs,
        "requested_algorithms": requested_algorithms,
        "available_design_count": len(available_designs),
        "design_count": len(designs),
        "designs": designs,
        "zip_count": len(generated_zips),
        "script_count": generated_script_count,
        "zip_layout": "synScripts/optimization/{algorithm}/{design}.zip",
        "script_layout": "optimizeBulk_{algorithm}_{design}_{base|tier1|tier2}.sh",
        "output_mapping": {
            "base_aigs": "tier1",
            "tier1": "tier2",
            "tier2": "tier3",
        },
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print("==========================================")
    print("Optimization shard scripts generated")
    print("==========================================")
    print(f"FULL_DATASET: {full_dataset}")
    print(f"Config source: {config_path}")
    print(f"Runtime config copy: {runtime_config_path}")
    print(f"Script zip root: {synscripts_opt_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"Design group: {args.design_group}")
    print(f"Requested designs: {requested_designs if requested_designs else '[]'}")
    print(f"Requested algorithms: {requested_algorithms}")
    print(f"Algorithms: {selected_algorithms}")
    print(f"Input sources: {selected_input_sources}")
    print(f"Design zips: {len(generated_zips)}")
    print(f"Scripts: {generated_script_count}")


if __name__ == "__main__":
    main()
