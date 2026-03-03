#!/usr/bin/env python3
"""
Generate per-design optimization runner scripts for FULL_DATASET.

Artifacts are generated per:
- algorithm
- design
- input tier (base_aigs, tier1, tier2)

Scripts are stored as zipped bundles per design under:
  FULL_DATASET/synScripts/optimization/{design}.zip
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

    return f"""#!/bin/bash
set -euo pipefail

ALGORITHM="{algorithm}"
DESIGN="{design}"
FULL_DATASET="{full_dataset}"
ABC_RC="{abc_rc}"
INPUT_SOURCE="{input_source}"
OUTPUT_TIER="{output_tier}"
INPUT_LABEL="{input_label}"

COMMAND_TEMPLATE='{command_template_escaped}'

INPUT_ROOT="{input_root}"
OUTPUT_ROOT="optimized_aigs/${{ALGORITHM}}/${{OUTPUT_TIER}}"
RAW_LOG_ROOT="metadata/raw_logs/${{DESIGN}}/${{OUTPUT_TIER}}/${{ALGORITHM}}"

in_dir="${{FULL_DATASET}}/${{INPUT_ROOT}}/${{DESIGN}}"
out_dir="${{FULL_DATASET}}/${{OUTPUT_ROOT}}/${{DESIGN}}"
log_dir="${{FULL_DATASET}}/${{RAW_LOG_ROOT}}"

mkdir -p "$out_dir" "$log_dir"

if [[ ! -d "$in_dir" ]]; then
    echo "✗ Missing design input directory: $in_dir" >&2
    exit 1
fi

tmp_extract_root="$(mktemp -d "${{TMPDIR:-/tmp}}/opt_${{ALGORITHM}}_${{DESIGN}}_${{INPUT_LABEL}}_XXXXXX")"
trap 'rm -rf "$tmp_extract_root"' EXIT

processed=0
created=0
failed=0
discovered=0

run_one() {{
    local input_aig="$1"
    local input_ref="$2"
    local filename
    local output_aig
    local log_file
    local cmd
    local seed_value

    filename="$(basename "$input_ref")"
    output_aig="$out_dir/$filename"
    log_file="$log_dir/${{filename}}.log"
    seed_value="$(printf '%s' "$input_ref" | cksum | awk '{{print $1}}')"

    cmd="$COMMAND_TEMPLATE"
    cmd="${{cmd//\\{{input_aig\\}}/$input_aig}}"
    cmd="${{cmd//\\{{output_aig\\}}/$output_aig}}"
    cmd="${{cmd//\\{{abc_rc\\}}/$ABC_RC}}"
    cmd="${{cmd//\\{{seed\\}}/$seed_value}}"

    echo "Running: $cmd" > "$log_file"
    if eval "$cmd" >> "$log_file" 2>&1; then
        created=$((created + 1))
    else
        failed=$((failed + 1))
    fi
    processed=$((processed + 1))
}}

if [[ "$INPUT_SOURCE" == "base_aigs" ]]; then
    while IFS= read -r -d '' input_aig; do
        discovered=$((discovered + 1))
        run_one "$input_aig" "$input_aig"
    done < <(find "$in_dir" -maxdepth 1 -type f -name "*.aig" -print0)

    while IFS= read -r -d '' zip_file; do
        while IFS= read -r member_path; do
            [[ -z "$member_path" ]] && continue
            discovered=$((discovered + 1))

            member_file="$(basename "$member_path")"
            extracted_input="$tmp_extract_root/${{member_file}}"

            if unzip -p "$zip_file" "$member_path" > "$extracted_input"; then
                run_one "$extracted_input" "$member_file"
            else
                echo "✗ Failed to extract $member_path from $zip_file" >&2
                failed=$((failed + 1))
            fi
        done < <(unzip -Z1 "$zip_file" '*.aig' 2>/dev/null | sort)
    done < <(find "$in_dir" -maxdepth 1 -type f -name "syn*.zip" -print0)
else
    while IFS= read -r -d '' input_aig; do
        discovered=$((discovered + 1))
        run_one "$input_aig" "$input_aig"
    done < <(find "$in_dir" -type f -name "*.aig" -print0)
fi

if [[ $discovered -eq 0 ]]; then
    echo "✗ No input AIGs discovered under: $in_dir (input_source=$INPUT_SOURCE)" >&2
    exit 1
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
        for tier in ("tier1", "tier2", "tier3"):
            for design in designs:
                (opt_root / algorithm / tier / design).mkdir(
                    parents=True, exist_ok=True
                )

    generated_script_count = 0
    generated_zips: List[str] = []

    for design in designs:
        scripts_for_design: Dict[str, str] = {}
        for algorithm in selected_algorithms:
            for input_source in selected_input_sources:
                label = INPUT_SOURCE_TO_LABEL[input_source]
                script_name = f"optimizeBulk_{algorithm}_{design}_{label}.sh"
                scripts_for_design[script_name] = render_shard_script(
                    algorithm=algorithm,
                    design=design,
                    full_dataset=str(full_dataset),
                    abc_rc=str(abc_rc),
                    input_source=input_source,
                    algo_cfg=config["algorithms"][algorithm],
                )
                generated_script_count += 1

        zip_path = synscripts_opt_dir / f"{design}.zip"
        write_design_zip(zip_path, scripts_for_design)
        generated_zips.append(str(zip_path))

    manifest_path = (
        manifests_dir
        / f"bulk_scripts_manifest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
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
        "zip_layout": "synScripts/optimization/{design}.zip",
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
