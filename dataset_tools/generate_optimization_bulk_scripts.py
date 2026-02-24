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
INPUT_SOURCES = ["base_aigs", "tier1", "tier2"]
INPUT_SOURCE_TO_OUTPUT = {
    "base_aigs": "tier1",
    "tier1": "tier2",
    "tier2": "final",
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

    if design_group == "random":
        selected = available.intersection(RANDOM_AIG_DESIGNS)
    elif design_group == "openabc":
        selected = available.difference(RANDOM_AIG_DESIGNS)
    else:
        selected = set(available_designs)

    if explicit_designs:
        requested = set(explicit_designs)
        missing = sorted(requested.difference(available))
        if missing:
            raise ValueError(
                "Requested designs not found in base_aigs: " + ", ".join(missing)
            )
        selected = selected.intersection(requested)

    selected_list = sorted(selected)
    if not selected_list:
        raise ValueError(
            "No designs selected after applying filters. "
            f"design_group={design_group}, explicit_designs={explicit_designs or '[]'}"
        )
    return selected_list


def shell_quote_single(value: str) -> str:
    return value.replace("'", "'\"'\"'")


def render_shard_script(
    algorithm: str,
    design: str,
    full_dataset: str,
    abc_rc: str,
    input_source: str,
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

DRY_RUN="${{DRY_RUN:-{str(runtime.get("dry_run", True)).lower()}}}"
TIMEOUT_SECONDS="${{TIMEOUT_SECONDS:-{int(runtime.get("timeout_seconds", 600))}}}"
THREADS="${{THREADS:-{int(runtime.get("threads", 1))}}}"
MAX_RETRIES="${{MAX_RETRIES:-{int(runtime.get("max_retries", 2))}}}"

COMMAND_TEMPLATE='{command_template_escaped}'

INPUT_ROOT="{input_root}"
OUTPUT_ROOT="optimized_aigs/${{ALGORITHM}}/${{OUTPUT_TIER}}"
LOG_ROOT="optimized_aigs/logs/${{ALGORITHM}}/${{OUTPUT_TIER}}"
DONE_ROOT="optimized_aigs/done/${{ALGORITHM}}/${{OUTPUT_TIER}}"

if [[ ! -d "${{FULL_DATASET}}/${{INPUT_ROOT}}" ]]; then
  echo "✗ Missing input root: ${{FULL_DATASET}}/${{INPUT_ROOT}}"
  exit 1
fi

{abc_rc_check}in_dir="${{FULL_DATASET}}/${{INPUT_ROOT}}/${{DESIGN}}"
out_dir="${{FULL_DATASET}}/${{OUTPUT_ROOT}}/${{DESIGN}}"
mkdir -p "$out_dir" "${{FULL_DATASET}}/${{LOG_ROOT}}" "${{FULL_DATASET}}/${{DONE_ROOT}}"

if [[ ! -d "$in_dir" ]]; then
  echo "✗ Missing design input directory: $in_dir"
  exit 1
fi

log_file="${{FULL_DATASET}}/${{LOG_ROOT}}/${{DESIGN}}.log"
done_file="${{FULL_DATASET}}/${{DONE_ROOT}}/${{DESIGN}}.done"

if [[ -f "$done_file" ]]; then
  echo "✓ Already completed: $done_file"
  exit 0
fi

echo "==========================================" | tee -a "$log_file"
echo "Optimization shard runner" | tee -a "$log_file"
echo "Algorithm: $ALGORITHM" | tee -a "$log_file"
echo "Design:    $DESIGN" | tee -a "$log_file"
echo "Input src: $INPUT_SOURCE" | tee -a "$log_file"
echo "Output:    $OUTPUT_TIER" | tee -a "$log_file"
echo "Input:     $in_dir" | tee -a "$log_file"
echo "Output:    $out_dir" | tee -a "$log_file"
echo "DRY_RUN:   $DRY_RUN" | tee -a "$log_file"
echo "==========================================" | tee -a "$log_file"

processed=0
created=0
skipped=0
failed=0

run_one_input() {{
  local input_for_cmd="$1"
  local logical_input_id="$2"
  local filename="$3"
  local output_aig="${{out_dir}}/${{filename}}"

  if [[ -f "$output_aig" ]]; then
    skipped=$((skipped + 1))
    return 0
  fi

  local seed_hex
  local seed
  seed_hex=$(printf "%s" "$logical_input_id" | shasum | awk '{{print $1}}' | cut -c1-8)
  seed=$((16#$seed_hex))

  local cmd
  cmd="$COMMAND_TEMPLATE"
  cmd="${{cmd//\{{input_aig\}}/$input_for_cmd}}"
  cmd="${{cmd//\{{output_aig\}}/$output_aig}}"
  cmd="${{cmd//\{{timeout_seconds\}}/$TIMEOUT_SECONDS}}"
  cmd="${{cmd//\{{threads\}}/$THREADS}}"
  cmd="${{cmd//\{{seed\}}/$seed}}"
  cmd="${{cmd//\{{abc_rc\}}/$ABC_RC}}"

  processed=$((processed + 1))

  if [[ "$DRY_RUN" == "true" ]]; then
    echo "[DRY_RUN] $cmd" | tee -a "$log_file"
    created=$((created + 1))
    if [[ "$ALGORITHM" == "Deepsyn" ]]; then
      echo "$seed" > "${{output_aig%.aig}}.seed.txt"
    fi
    return 0
  fi

  local ok=false
  local attempt=0
  while [[ "$ok" == "false" && $attempt -le $MAX_RETRIES ]]; do
    if eval "$cmd"; then
      ok=true
    else
      attempt=$((attempt + 1))
    fi
  done

  if [[ "$ok" == "true" ]]; then
    created=$((created + 1))
    if [[ "$ALGORITHM" == "Deepsyn" ]]; then
      echo "$seed" > "${{output_aig%.aig}}.seed.txt"
    fi
    return 0
  fi

  failed=$((failed + 1))
  echo "✗ Failed: $logical_input_id" | tee -a "$log_file"
  return 1
}}

if [[ "$INPUT_SOURCE" == "base_aigs" ]]; then
  if ! command -v unzip >/dev/null 2>&1; then
    echo "✗ Missing required tool: unzip" | tee -a "$log_file"
    exit 1
  fi

  while IFS= read -r -d '' input_aig; do
    filename="$(basename "$input_aig")"
    run_one_input "$input_aig" "$input_aig" "$filename"
  done < <(find "$in_dir" -maxdepth 1 -type f -name "*_orig.aig" -print0 | sort -z)

  while IFS= read -r -d '' zip_path; do
    while IFS= read -r member; do
      [[ -z "$member" ]] && continue
      filename="$(basename "$member")"
      logical_input="$zip_path::$member"
      input_for_cmd="$logical_input"
      tmp_input=""

      if [[ "$DRY_RUN" != "true" ]]; then
        tmp_input="$(mktemp "${{TMPDIR:-/tmp}}/opt_input_${{DESIGN}}_XXXXXX.aig")"
        if ! unzip -p "$zip_path" "$member" > "$tmp_input"; then
          rm -f "$tmp_input"
          failed=$((failed + 1))
          echo "✗ Failed to extract: $zip_path::$member" | tee -a "$log_file"
          continue
        fi
        input_for_cmd="$tmp_input"
      fi

      run_one_input "$input_for_cmd" "$logical_input" "$filename"

      if [[ -n "$tmp_input" ]]; then
        rm -f "$tmp_input"
      fi
    done < <(unzip -Z1 "$zip_path" '*.aig' | sort)
  done < <(find "$in_dir" -maxdepth 1 -type f -name 'syn*.zip' -print0 | sort -z)
else
  while IFS= read -r -d '' input_aig; do
    filename="$(basename "$input_aig")"
    run_one_input "$input_aig" "$input_aig" "$filename"
  done < <(find "$in_dir" -type f -name "*.aig" -print0 | sort -z)
fi

echo "" | tee -a "$log_file"
echo "Summary ($ALGORITHM, $DESIGN, input=$INPUT_SOURCE, output=$OUTPUT_TIER):" | tee -a "$log_file"
echo "  processed=$processed" | tee -a "$log_file"
echo "  created=$created" | tee -a "$log_file"
echo "  skipped=$skipped" | tee -a "$log_file"
echo "  failed=$failed" | tee -a "$log_file"

if [[ $failed -gt 0 ]]; then
  exit 1
fi

echo "algorithm=$ALGORITHM" > "$done_file"
echo "design=$DESIGN" >> "$done_file"
echo "input_source=$INPUT_SOURCE" >> "$done_file"
echo "output_tier=$OUTPUT_TIER" >> "$done_file"
echo "processed=$processed" >> "$done_file"
echo "created=$created" >> "$done_file"
echo "skipped=$skipped" >> "$done_file"
echo "failed=$failed" >> "$done_file"
echo "completed_at=$(date -Iseconds)" >> "$done_file"
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

        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
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
        for tier in ("tier1", "tier2", "final"):
            for design in designs:
                (opt_root / algorithm / tier / design).mkdir(parents=True, exist_ok=True)

    runtime = config.get("runtime", {})
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
                    runtime=runtime,
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
            "tier2": "final",
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
