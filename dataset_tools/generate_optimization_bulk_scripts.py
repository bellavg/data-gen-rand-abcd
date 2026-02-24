#!/usr/bin/env python3
"""
Generate per-design-per-algorithm optimization runner scripts for FULL_DATASET.

Pattern mirrors synthesis flow while staying scalable:
- One generation step creates bulk shell scripts
- Separate per-algorithm SLURM jobs execute generated scripts
- Each generated script handles one (algorithm, design) shard
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List

ALGORITHMS = ["Orchestrate", "Deepsyn", "Syn4", "C2RS"]


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


def shell_quote_single(value: str) -> str:
    return value.replace("'", "'\"'\"'")


def render_shard_script(
    algorithm: str,
    design: str,
    full_dataset: str,
    abc_rc: str,
    runtime: Dict,
    algo_cfg: Dict,
) -> str:
    command_template = str(algo_cfg.get("command_template", "")).strip()
    if not command_template:
        raise ValueError(f"Missing command_template for algorithm {algorithm}")

    command_template_escaped = shell_quote_single(command_template)
    needs_abc_rc = algorithm in ("Syn4", "C2RS")
    abc_rc_check = (
        (
            'if [[ ! -f "$ABC_RC" ]]; then\n'
            '  echo "✗ Missing abc.rc: $ABC_RC"\n'
            "  exit 1\n"
            "fi\n"
        )
        if needs_abc_rc
        else ""
    )

    return f"""#!/bin/bash
set -euo pipefail

ALGORITHM=\"{algorithm}\"
DESIGN=\"{design}\"
FULL_DATASET=\"{full_dataset}\"
ABC_RC=\"{abc_rc}\"

TIER=\"${{TIER:-1}}\"
DRY_RUN=\"${{DRY_RUN:-{str(runtime.get("dry_run", True)).lower()}}}\"
TIMEOUT_SECONDS=\"${{TIMEOUT_SECONDS:-{int(runtime.get("timeout_seconds", 600))}}}\"
THREADS=\"${{THREADS:-{int(runtime.get("threads", 1))}}}\"
MAX_RETRIES=\"${{MAX_RETRIES:-{int(runtime.get("max_retries", 2))}}}\"

COMMAND_TEMPLATE='{command_template_escaped}'

if [[ \"$TIER\" != \"1\" && \"$TIER\" != \"2\" ]]; then
  echo \"✗ Invalid TIER=$TIER (must be 1 or 2)\"
  exit 1
fi

if [[ \"$TIER\" == \"1\" ]]; then
  INPUT_ROOT=\"base_aigs\"
else
  INPUT_ROOT=\"optimized_aigs/${{ALGORITHM}}/tier1\"
fi
OUTPUT_ROOT=\"optimized_aigs/${{ALGORITHM}}/tier${{TIER}}\"
LOG_ROOT=\"optimized_aigs/logs/${{ALGORITHM}}/tier${{TIER}}\"
DONE_ROOT=\"optimized_aigs/done/${{ALGORITHM}}/tier${{TIER}}\"

if [[ ! -d \"${{FULL_DATASET}}/${{INPUT_ROOT}}\" ]]; then
  echo \"✗ Missing input root: ${{FULL_DATASET}}/${{INPUT_ROOT}}\"
  exit 1
fi

{abc_rc_check}
in_dir=\"${{FULL_DATASET}}/${{INPUT_ROOT}}/${{DESIGN}}\"
out_dir=\"${{FULL_DATASET}}/${{OUTPUT_ROOT}}/${{DESIGN}}\"
mkdir -p \"$out_dir\" \"${{FULL_DATASET}}/${{LOG_ROOT}}\" \"${{FULL_DATASET}}/${{DONE_ROOT}}\"

if [[ ! -d \"$in_dir\" ]]; then
  echo \"✗ Missing design input directory: $in_dir\"
  exit 1
fi

log_file=\"${{FULL_DATASET}}/${{LOG_ROOT}}/${{DESIGN}}.log\"
done_file=\"${{FULL_DATASET}}/${{DONE_ROOT}}/${{DESIGN}}.done\"

if [[ -f \"$done_file\" ]]; then
  echo \"✓ Already completed: $done_file\"
  exit 0
fi

echo \"==========================================\" | tee -a \"$log_file\"
echo \"Optimization shard runner\" | tee -a \"$log_file\"
echo \"Algorithm: $ALGORITHM\" | tee -a \"$log_file\"
echo \"Design:    $DESIGN\" | tee -a \"$log_file\"
echo \"Tier:      $TIER\" | tee -a \"$log_file\"
echo \"Input:     $in_dir\" | tee -a \"$log_file\"
echo \"Output:    $out_dir\" | tee -a \"$log_file\"
echo \"DRY_RUN:   $DRY_RUN\" | tee -a \"$log_file\"
echo \"==========================================\" | tee -a \"$log_file\"

processed=0
created=0
skipped=0
failed=0

while IFS= read -r -d '' input_aig; do
  filename=\"$(basename \"$input_aig\")\"
  output_aig=\"${{out_dir}}/${{filename}}\"

  if [[ -f \"$output_aig\" ]]; then
    skipped=$((skipped + 1))
    continue
  fi

  seed_hex=$(printf \"%s\" \"$input_aig\" | shasum | awk '{{print $1}}' | cut -c1-8)
  seed=$((16#$seed_hex))

  cmd=\"$COMMAND_TEMPLATE\"
  cmd=\"${{cmd//\{{input_aig\}}/$input_aig}}\"
  cmd=\"${{cmd//\{{output_aig\}}/$output_aig}}\"
  cmd=\"${{cmd//\{{timeout_seconds\}}/$TIMEOUT_SECONDS}}\"
  cmd=\"${{cmd//\{{threads\}}/$THREADS}}\"
  cmd=\"${{cmd//\{{seed\}}/$seed}}\"
  cmd=\"${{cmd//\{{abc_rc\}}/$ABC_RC}}\"

  processed=$((processed + 1))

  if [[ \"$DRY_RUN\" == \"true\" ]]; then
    echo \"[DRY_RUN] $cmd\" | tee -a \"$log_file\"
    created=$((created + 1))
    if [[ \"$ALGORITHM\" == \"Deepsyn\" ]]; then
      echo \"$seed\" > \"${{output_aig%.aig}}.seed.txt\"
    fi
    continue
  fi

  ok=false
  attempt=0
  while [[ \"$ok\" == \"false\" && $attempt -le $MAX_RETRIES ]]; do
    if eval \"$cmd\"; then
      ok=true
    else
      attempt=$((attempt + 1))
    fi
  done

  if [[ \"$ok\" == \"true\" ]]; then
    created=$((created + 1))
    if [[ \"$ALGORITHM\" == \"Deepsyn\" ]]; then
      echo \"$seed\" > \"${{output_aig%.aig}}.seed.txt\"
    fi
  else
    failed=$((failed + 1))
    echo \"✗ Failed: $input_aig\" | tee -a \"$log_file\"
  fi
done < <(find \"$in_dir\" -type f -name \"*.aig\" -print0 | sort -z)

echo \"\" | tee -a \"$log_file\"
echo \"Summary ($ALGORITHM, $DESIGN, tier $TIER):\" | tee -a \"$log_file\"
echo \"  processed=$processed\" | tee -a \"$log_file\"
echo \"  created=$created\" | tee -a \"$log_file\"
echo \"  skipped=$skipped\" | tee -a \"$log_file\"
echo \"  failed=$failed\" | tee -a \"$log_file\"

if [[ $failed -gt 0 ]]; then
  exit 1
fi

echo \"algorithm=$ALGORITHM\" > \"$done_file\"
echo \"design=$DESIGN\" >> \"$done_file\"
echo \"tier=$TIER\" >> \"$done_file\"
echo \"processed=$processed\" >> \"$done_file\"
echo \"created=$created\" >> \"$done_file\"
echo \"skipped=$skipped\" >> \"$done_file\"
echo \"failed=$failed\" >> \"$done_file\"
echo \"completed_at=$(date -Iseconds)\" >> \"$done_file\"
"""


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
    scripts_dir = opt_root / "scripts"
    manifests_dir = opt_root / "manifests"
    runtime_config_dir = opt_root / "config"
    runtime_config_path = runtime_config_dir / "optimization_config.json"
    abc_rc = base_dir / "abc.rc"

    config = load_config(config_path)
    designs = discover_designs(base_aigs_dir)

    scripts_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)
    runtime_config_dir.mkdir(parents=True, exist_ok=True)

    shutil.copyfile(config_path, runtime_config_path)

    for algorithm in ALGORITHMS:
        for tier in ("tier1", "tier2"):
            for design in designs:
                (opt_root / algorithm / tier / design).mkdir(
                    parents=True, exist_ok=True
                )

    runtime = config.get("runtime", {})
    generated_scripts: List[str] = []
    for algorithm in ALGORITHMS:
        algorithm_script_dir = scripts_dir / algorithm
        algorithm_script_dir.mkdir(parents=True, exist_ok=True)
        for design in designs:
            script_path = algorithm_script_dir / f"optimizeBulk_{algorithm}_{design}.sh"
            content = render_shard_script(
                algorithm=algorithm,
                design=design,
                full_dataset=str(full_dataset),
                abc_rc=str(abc_rc),
                runtime=runtime,
                algo_cfg=config["algorithms"][algorithm],
            )
            with script_path.open("w", encoding="utf-8") as handle:
                handle.write(content)
            script_path.chmod(0o755)
            generated_scripts.append(str(script_path))

    manifest_path = (
        manifests_dir
        / f"bulk_scripts_manifest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    manifest = {
        "generated_at": datetime.now().isoformat(),
        "full_dataset": str(full_dataset),
        "config_source": str(config_path),
        "runtime_config_copy": str(runtime_config_path),
        "scripts_dir": str(scripts_dir),
        "algorithms": ALGORITHMS,
        "design_count": len(designs),
        "designs": designs,
        "script_count": len(generated_scripts),
        "script_layout": "optimized_aigs/scripts/{algorithm}/optimizeBulk_{algorithm}_{design}.sh",
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print("==========================================")
    print("Optimization shard scripts generated")
    print("==========================================")
    print(f"FULL_DATASET: {full_dataset}")
    print(f"Config source: {config_path}")
    print(f"Runtime config copy: {runtime_config_path}")
    print(f"Scripts dir: {scripts_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"Designs: {len(designs)}")
    print(f"Scripts: {len(generated_scripts)}")


if __name__ == "__main__":
    main()
