#!/usr/bin/env python3
"""
Generate tier-agnostic per-design optimization runner scripts.
"""

from __future__ import annotations
import argparse
import os
from pathlib import Path
from typing import Dict

# ==============================================================================
# CHANGED: Added 'print_stats;' after read and before write for all algorithms!
# ==========================================
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

DESIGNS = [
    "128", "256", "512", "1024", "2048", "4096", "8192", "16384",
    "ac97_ctrl", "aes", "aes_secworks", "aes_xcrypt", "apex1", "bc0", 
    "bp_be", "c1355", "c5315", "c6288", "c7552", "dalu", "des3_area", 
    "dft", "div", "dynamic_node", "ethernet", "fir", "fpu", "hyp", 
    "i10", "i2c", "idft", "iir", "jpeg", "k2", "log2", "mainpla", 
    "max", "mem_ctrl", "multiplier", "pci", "picosoc", "sasc", 
    "sha256", "simple_spi", "sin", "spi", "sqrt", "square", "ss_pcm", 
    "tinyRocket", "tv80", "usb_phy", "vga_lcd", "wb_conmax", "wb_dma"
]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate xargs-based optimization scripts")
    parser.add_argument("--home", required=True, help="Root path of the project")
    return parser.parse_args()

def shell_quote_single(value: str) -> str:
    return value.replace("'", "'\"'\"'")

def render_agnostic_script(
    algorithm: str, design: str, abc_rc: str, algo_cfg: Dict, timeout_seconds: int | None,
) -> str:
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
    local tier_name="$(basename "$(dirname "$OUT_DIR")")"
    
    if [[ "$input_member" =~ (syn[0-9]+(_step[0-9]+)?\.aig)$ ]]; then
        local suffix="${{BASH_REMATCH[1]}}"
        local output_member="${{DESIGN}}_${{ALGORITHM}}_${{tier_name}}_${{suffix}}"
    else
        local suffix="${{input_member#${{DESIGN}}_}}"
        local output_member="${{DESIGN}}_${{ALGORITHM}}_${{tier_name}}_${{suffix}}"
    fi
    
    local output_final="${{OUT_DIR}}/${{output_member}}"
    local log_final="${{LOG_DIR}}/${{output_member%.aig}.log}"
    local seed_value="$RANDOM"

    local cmd="${{COMMAND_TEMPLATE//\\{{input_aig\\}}/$input_aig}}"
    cmd="${{cmd//\\{{output_aig\\}}/$output_final}}"
    cmd="${{cmd//\\{{abc_rc\\}}/$ABC_RC}}"
    cmd="${{cmd//\\{{seed\\}}/$seed_value}}"
    cmd="${{cmd//\\{{timeout_seconds\\}}/$RUNTIME_TIMEOUT_SECONDS}}"

    # Execute the synthesized command and push output to the specific log file
    eval "$cmd" > "$log_final" 2>&1
}}
export -f run_one

find "$INPUT_SRC" -maxdepth 1 -name "*.aig" -print0 | \\
    xargs -0 -P "$WORKERS" -n 1 -I {{}} bash -c 'run_one "$1"' _ {{}}
"""

def generate_all_scripts(home_dir: str) -> None:
    base_dir = Path(home_dir).expanduser().resolve()
    abc_scripts_root = base_dir / "data" / "abc_scripts" / "optimization_scripts"
    abc_rc = base_dir / "data" / "abc_scripts" / "abc.rc"
    
    generated_count = 0

    for design in DESIGNS:
        design_script_dir = abc_scripts_root / design
        design_script_dir.mkdir(parents=True, exist_ok=True)

        for algorithm in ALGORITHMS:
            algo_cfg = CONFIG["algorithms"].get(algorithm, {})
            timeout_seconds = int(algo_cfg.get("timeout_seconds", 20)) if algorithm == "Deepsyn" else None
            
            script_path = design_script_dir / f"{algorithm}.sh"
            script_content = render_agnostic_script(
                algorithm=algorithm, design=design, abc_rc=str(abc_rc),
                algo_cfg=algo_cfg, timeout_seconds=timeout_seconds,
            )

            with open(script_path, "w", encoding="utf-8") as f:
                f.write(script_content)

            try:
                script_path.chmod(0o755)
            except Exception:
                pass
            generated_count += 1

    print(f"✓ Generated {generated_count} highly-parallel optimization scripts.")

def main() -> None:
    args = parse_args()
    generate_all_scripts(args.home)

if __name__ == "__main__":
    main()