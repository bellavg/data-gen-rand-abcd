#!/usr/bin/env python3
"""
Generate tier-agnostic per-design optimization runner scripts.

Scripts are stored under:
    data/abc_scripts/optimization_scripts/{design}/{algorithm}.sh

These bash scripts dynamically accept an input source directory and an
output destination, allowing you to use ONE script for Tier 1, Tier 2, etc.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

# ==============================================================================
# EMBEDDED OPTIMIZATION CONFIGURATION
# ==============================================================================
CONFIG = {
    "algorithms": {
        "Orchestrate": {
            "command_template": "abc -c 'source {abc_rc}; read {input_aig}; strash; orchestrate; write {output_aig}'"
        },
        "Deepsyn": {
            "timeout_seconds": 20,
            "command_template": "abc -c 'source {abc_rc}; read {input_aig}; strash; &get; &deepsyn -S {seed} -T {timeout_seconds}; &put; write {output_aig}'",
        },
        "Syn4": {
            "command_template": "abc -c 'source {abc_rc}; read {input_aig}; strash; &get; &syn4; &put; write {output_aig}'"
        },
        "C2RS": {
            "command_template": "abc -c 'source {abc_rc}; read {input_aig}; strash; c2rs; write {output_aig}'"
        },
    }
}

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate tier-agnostic optimization scripts"
    )
    parser.add_argument(
        "--base-dir",
        default="YOUR_CONSTANT_PATH_HERE",
        help="Project base directory",
    )
    return parser.parse_args()


def shell_quote_single(value: str) -> str:
    return value.replace("'", "'\"'\"'")


def render_agnostic_script(
    algorithm: str,
    design: str,
    abc_rc: str,
    algo_cfg: Dict,
    timeout_seconds: int | None,
) -> str:
    command_template = str(algo_cfg.get("command_template", "")).strip()
    command_template_escaped = shell_quote_single(command_template)
    timeout_seconds_value = "" if timeout_seconds is None else str(timeout_seconds)

    return f"""#!/bin/bash
set -euo pipefail

# Configuration
ALGORITHM="{algorithm}"
DESIGN="{design}"
ABC_RC="{abc_rc}"
RUNTIME_TIMEOUT_SECONDS="{timeout_seconds_value}"
COMMAND_TEMPLATE='{command_template_escaped}'

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <input_aig_directory> <output_directory> [workers]"
    echo "Example: $0 data/designs/${{DESIGN}}/tier0 data/designs/${{DESIGN}}/tier1/${{ALGORITHM}}"
    exit 1
fi

INPUT_SRC="$(realpath "$1")"
OUT_DIR="$(realpath "$2")"
WORKERS="${{3:-${{OPT_SCRIPT_PARALLELISM:-${{SLURM_CPUS_PER_TASK:-128}}}}}}"

echo "=== Starting ${{ALGORITHM}} Optimization for ${{DESIGN}} ==="

if [[ ! -d "$INPUT_SRC" ]]; then
    echo "✗ ERROR: Input directory not found at $INPUT_SRC" >&2
    exit 1
fi

mkdir -p "$OUT_DIR"

# Setup high-speed local scratch space
LOCAL_SCRATCH="${{TMPDIR:-/tmp}}"
mkdir -p "$LOCAL_SCRATCH"

tmp_extract_root="$(mktemp -d "$LOCAL_SCRATCH/opt_${{ALGORITHM}}_${{DESIGN}}_XXXXXX")"
trap 'rm -rf "$tmp_extract_root"' EXIT

INPUT_TMP_DIR="$tmp_extract_root/in"
OUT_TMP_DIR="$tmp_extract_root/out"
mkdir -p "$INPUT_TMP_DIR" "$OUT_TMP_DIR"

echo "Staging inputs from $INPUT_SRC to local scratch..."
find "$INPUT_SRC" -maxdepth 1 -name "*.aig" -exec ln -s {{}} "$INPUT_TMP_DIR/" \\;

discovered=$(find "$INPUT_TMP_DIR" -maxdepth 1 -name "*.aig" | wc -l | tr -d ' ')
if [[ $discovered -eq 0 ]]; then
    echo "✗ ERROR: No input AIGs discovered in: $INPUT_SRC" >&2
    exit 1
fi

echo "Processing $discovered files with $WORKERS concurrent workers..."

export COMMAND_TEMPLATE ABC_RC RUNTIME_TIMEOUT_SECONDS OUT_TMP_DIR OUT_DIR DESIGN ALGORITHM

run_one() {{
    local input_aig="$1"
    local input_member="$(basename "$input_aig")"
    
    # Extract 'tierX' (e.g., 'tier1', 'tier2') from the output directory path
    local tier_name="$(basename "$(dirname "$OUT_DIR")")"
    
    # Use Regex to dynamically extract the synX_stepY base regardless of prior tier tags
    if [[ "$input_member" =~ (syn[0-9]+(_step[0-9]+)?\.aig)$ ]]; then
        local suffix="${{BASH_REMATCH[1]}}"
        local output_member="${{DESIGN}}_${{ALGORITHM}}_${{tier_name}}_${{suffix}}"
    else
        # Fallback if filename format is completely unexpected
        local suffix="${{input_member#${{DESIGN}}_}}"
        local output_member="${{DESIGN}}_${{ALGORITHM}}_${{tier_name}}_${{suffix}}"
    fi
    
    local output_tmp="${{OUT_TMP_DIR}}/tmp_${{RANDOM}}_${{BASHPID}}.aig"
    local seed_value="$RANDOM"

    # Swap in the parameters matching the config
    local cmd="${{COMMAND_TEMPLATE//\\{{input_aig\\}}/$input_aig}}"
    cmd="${{cmd//\\{{output_aig\\}}/$output_tmp}}"
    cmd="${{cmd//\\{{abc_rc\\}}/$ABC_RC}}"
    cmd="${{cmd//\\{{seed\\}}/$seed_value}}"
    cmd="${{cmd//\\{{timeout_seconds\\}}/$RUNTIME_TIMEOUT_SECONDS}}"

    if eval "$cmd" >/dev/null 2>&1; then
        if [[ -f "$output_tmp" ]]; then
            # Move successfully created file to the output directory
            mv -f "$output_tmp" "${{OUT_TMP_DIR}}/${{output_member}}"
            return 0
        fi
    fi
    rm -f "$output_tmp"
    return 1
}}
export -f run_one

# Xargs multi-core processing
find "$INPUT_TMP_DIR" -maxdepth 1 -name "*.aig" -print0 | \\
    xargs -0 -P "$WORKERS" -n 1 -I {{}} bash -c 'run_one "$1" || echo "FAIL"' _ {{}} > "$tmp_extract_root/failures.log"

failed=$(grep -c "FAIL" "$tmp_extract_root/failures.log" || true)
created=$(find "$OUT_TMP_DIR" -maxdepth 1 -type f -name "*.aig" | wc -l | tr -d ' ')

echo "Transferring $created optimized AIGs to $OUT_DIR..."
if [[ $created -gt 0 ]]; then
    # Move them back to the permanent output directory
    mv "$OUT_TMP_DIR"/*.aig "$OUT_DIR/"
fi

if [[ $failed -gt 0 ]]; then
    echo "✗ Optimization completed with $failed failures." >&2
    exit 1
else
    echo "✓ Success. $created files created in $OUT_DIR."
fi
"""


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir).expanduser().resolve()

    designs = sorted(RANDOM_AIG_DESIGNS + OPENABC_DESIGNS)

    # Root folder for generated scripts: data/abc_scripts/optimization_scripts
    abc_scripts_root = base_dir / "data" / "abc_scripts" / "optimization_scripts"
    abc_rc = base_dir / "data" / "abc_scripts" / "abc.rc"

    generated_script_count = 0

    for design in designs:
        # Create data/abc_scripts/optimization_scripts/{design_name}/
        design_script_dir = abc_scripts_root / design
        design_script_dir.mkdir(parents=True, exist_ok=True)

        for algorithm in ALGORITHMS:
            algo_cfg = CONFIG["algorithms"].get(algorithm, {})
            # Only DeepSyn explicitly uses the timeout flag in its template
            timeout_seconds = (
                int(algo_cfg.get("timeout_seconds", 20))
                if algorithm == "Deepsyn"
                else None
            )

            # data/abc_scripts/optimization_scripts/{design_name}/{algorithm}.sh
            script_path = design_script_dir / f"{algorithm}.sh"

            script_content = render_agnostic_script(
                algorithm=algorithm,
                design=design,
                abc_rc=str(abc_rc),
                algo_cfg=algo_cfg,
                timeout_seconds=timeout_seconds,
            )

            with open(script_path, "w", encoding="utf-8") as f:
                f.write(script_content)

            # Make executable (Wrapped in try/except for virtualized/shared filesystems or AV delays)
            try:
                script_path.chmod(0o755)
            except Exception:
                pass

            generated_script_count += 1

    print("==================================================")
    print(" Tier-Agnostic Optimization Scripts Generated")
    print("==================================================")
    print(f" Script Root: {abc_scripts_root}")
    print(f" Algorithms : {', '.join(ALGORITHMS)}")
    print(f" Designs    : {len(designs)} selected")
    print(f" Total files: {generated_script_count} (.sh executable scripts)")
    print("\n File Structure:")
    print("  data/abc_scripts/optimization_scripts/")
    print("  ├── aes/")
    print("  │   ├── Orchestrate.sh")
    print("  │   ├── Deepsyn.sh")
    print("  │   ├── Syn4.sh")
    print("  │   └── C2RS.sh")
    print("  └── ...")
    print("\nExample usage for Tier 1:")
    print("  bash data/abc_scripts/optimization_scripts/aes/Syn4.sh \\")
    print("       data/designs/aes/tier0 \\")
    print("       data/designs/aes/tier1/Syn4")
    print("\nExample usage for Tier 2:")
    print("  bash data/abc_scripts/optimization_scripts/aes/Syn4.sh \\")
    print("       data/designs/aes/tier1/Syn4 \\")
    print("       data/designs/aes/tier2/Syn4")


if __name__ == "__main__":
    main()
