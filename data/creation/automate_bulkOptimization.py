#!/usr/bin/env python3
"""
Generate tier-agnostic per-design optimization runner scripts and ZIP them.
"""

from __future__ import annotations
import argparse
import zipfile
from pathlib import Path
from typing import Dict

CONFIG = {
    "algorithms": {
        "Orchestrate": {
            "command_template": "abc -c 'source {abc_rc}; read {input_aig}; print_stats; strash; orchestrate; print_stats; write {output_aig}'"
        },
        "Deepsyn": {
            "timeout_seconds": 20,
            "command_template": "abc -c 'source {abc_rc}; read {input_aig}; print_stats; strash; &get; &deepsyn -S {seed} -T {timeout_seconds}; &put; print_stats; write {output_aig}'",
        },
        "Syn4": {
            "command_template": "abc -c 'source {abc_rc}; read {input_aig}; print_stats; strash; &get; &syn4; &put; print_stats; write {output_aig}'"
        },
        "C2RS": {
            "command_template": "abc -c 'source {abc_rc}; read {input_aig}; print_stats; strash; c2rs; print_stats; write {output_aig}'"
        },
    }
}

ALGORITHMS = ["Orchestrate", "Deepsyn", "Syn4", "C2RS"]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--home", required=True, help="Root path of the project")
    parser.add_argument("--design", required=True, help="Specific design to generate scripts for")
    return parser.parse_args()

def shell_quote_single(value: str) -> str:
    return value.replace("'", "'\"'\"'")

def render_agnostic_script(algorithm: str, design: str, abc_rc: str, algo_cfg: Dict, timeout_seconds: int | None) -> str:
    command_template = str(algo_cfg.get("command_template", "")).strip()
    command_template_escaped = shell_quote_single(command_template)
    timeout_seconds_value = "" if timeout_seconds is None else str(timeout_seconds)

    return f"""#!/bin/bash
set -euo pipefail

ALGORITHM="{algorithm}"
DESIGN="{design}"
ABC_RC="{abc_rc}"
RUNTIME_TIMEOUT_SECONDS="{timeout_seconds_value}"
COMMAND_TEMPLATE='{command_template_escaped}'

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <input_aig_directory> <output_directory> <log_directory> [workers]"
    exit 1
fi

INPUT_SRC="$1"
OUT_DIR="$2"
LOG_DIR="$3"
WORKERS="${{4:-192}}"

export COMMAND_TEMPLATE ABC_RC RUNTIME_TIMEOUT_SECONDS OUT_DIR LOG_DIR DESIGN ALGORITHM

run_one() {{
    local input_aig="$1"
    local input_member="$(basename "$input_aig")"
    
    # 1. Extract the raw "syn[recipe]_step[step].aig" suffix
    local suffix=""
    if [[ "$input_member" =~ (syn[0-9X]+(_step[0-9]+)?\\.aig)$ ]]; then
        suffix="${{BASH_REMATCH[1]}}"
    else
        suffix="${{input_member##*_}}"
    fi

    # 2. Dynamically construct clean output names
    if [[ "$input_member" == *"_tier1_"* ]]; then
        # TIER-2 LOGIC: Strip out the messy mktemp string and grab the Tier-1 Algo
        local without_design="${{input_member#${{DESIGN}}_}}"
        local tier1_algo="${{without_design%%_tier1_*}}"
        
        # Format: [design]_[tier 1 algo]_[tier 2 algo]_tier2_syn[recipe]_step[step].aig
        local output_member="${{DESIGN}}_${{tier1_algo}}_${{ALGORITHM}}_tier2_${{suffix}}"
    else
        # TIER-1 LOGIC: Standard clean format
        # Format: [design]_[tier 1 algo]_tier1_syn[recipe]_step[step].aig
        local output_member="${{DESIGN}}_${{ALGORITHM}}_tier1_${{suffix}}"
    fi
    
    local output_final="${{OUT_DIR}}/${{output_member}}"
    local log_final="${{LOG_DIR}}/${{output_member%.aig}}.log"
    local seed_value="$RANDOM"

    local cmd="${{COMMAND_TEMPLATE//\\{{input_aig\\}}/$input_aig}}"
    cmd="${{cmd//\\{{output_aig\\}}/$output_final}}"
    cmd="${{cmd//\\{{abc_rc\\}}/$ABC_RC}}"
    cmd="${{cmd//\\{{seed\\}}/$seed_value}}"
    cmd="${{cmd//\\{{timeout_seconds\\}}/$RUNTIME_TIMEOUT_SECONDS}}"

    # Execute ABC. Catch errors gracefully. Print warning immediately to SLURM output AND save to log.
    eval "$cmd" > "$log_final" 2>&1 || (echo "⚠️ WARN: ABC failed on $input_member" >&2 && echo "WARN: ABC failed on $input_member" >> "$log_final")
}}
export -f run_one

# Run xargs and force a success return code just in case xargs throws a warning
find "$INPUT_SRC" -maxdepth 1 -name "*.aig" -print0 | \\
    xargs -0 -P "$WORKERS" -I {{}} bash -c 'run_one "$1"' _ {{}} || true
"""

def generate_scripts_for_design(home_dir: str, design: str) -> None:
    base_dir = Path(home_dir).expanduser().resolve()
    abc_scripts_root = base_dir / "data" / "abc_scripts" / "optimization_scripts"
    abc_scripts_root.mkdir(parents=True, exist_ok=True)
    abc_rc = base_dir / "data" / "abc_scripts" / "abc.rc"
    
    zip_path = abc_scripts_root / f"{design}.zip"
    
    # Write the 4 scripts directly into a ZIP file!
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for algorithm in ALGORITHMS:
            algo_cfg = CONFIG["algorithms"].get(algorithm, {})
            timeout_seconds = int(algo_cfg.get("timeout_seconds", 20)) if algorithm == "Deepsyn" else None
            
            script_content = render_agnostic_script(
                algorithm=algorithm, design=design, abc_rc=str(abc_rc),
                algo_cfg=algo_cfg, timeout_seconds=timeout_seconds,
            )
            # Save inside the zip as: aes/Orchestrate.sh
            zf.writestr(f"{design}/{algorithm}.sh", script_content)

    print(f"✓ Generated and zipped scripts to {zip_path}")

if __name__ == "__main__":
    args = parse_args()
    generate_scripts_for_design(args.home, args.design)