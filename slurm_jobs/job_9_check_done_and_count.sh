#!/bin/bash
#SBATCH --job-name=job_9_check_done
#SBATCH --time=00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=genoa
#SBATCH --output=logs/check_done_%j.out

set -euo pipefail

### Job 9: Check done markers and count .aig outputs
### Usage: sbatch slurm_jobs/job_9_check_done_and_count.sh [--full-dataset PATH] [--algorithm ALG[,ALG2]] --tier N [--list-missing]

FULL_DATASET="${FULL_DATASET:-/scratch-shared/${USER:-$LOGNAME}/FULL_DATASET}"
ALGORITHMS="${1:-all}"
TIER=1
LIST_MISSING=false

while [ "$#" -gt 0 ]; do
  case "$1" in
    --full-dataset) FULL_DATASET="$2"; shift 2;;
    --algorithm|-a) ALGORITHMS="$2"; shift 2;;
    --tier|-t) TIER="$2"; shift 2;;
    --list-missing) LIST_MISSING=true; shift;;
    --help|-h) echo "Usage: $0 [--full-dataset PATH] [--algorithm ALG[,ALG2]] --tier N [--list-missing]"; exit 0;;
    *) echo "Unknown arg: $1" >&2; exit 2;;
  esac
done

if [ ! -d "$FULL_DATASET" ]; then
  echo "✗ FULL_DATASET not found: $FULL_DATASET" >&2
  exit 2
fi

# Determine expected designs from base_aigs directory names
BASE_DIR="$FULL_DATASET/base_aigs"
if [ ! -d "$BASE_DIR" ]; then
  echo "✗ Missing base_aigs: $BASE_DIR" >&2
  exit 2
fi

mapfile -t EXPECTED < <(find "$BASE_DIR" -maxdepth 1 -mindepth 1 -type d -printf '%f\n' | sort)
expected_count=${#EXPECTED[@]}

if [ "$ALGORITHMS" = "all" ]; then
  # discover algorithms from optimized_aigs/done if available, fallback to folder listing
  algo_root="$FULL_DATASET/optimized_aigs/done"
  if [ -d "$algo_root" ]; then
    mapfile -t ALGO_LIST < <(find "$algo_root" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)
  else
    echo "✗ No optimized_aigs/done directory; please provide --algorithm" >&2
    exit 2
  fi
else
  IFS=',' read -r -a ALGO_LIST <<< "$ALGORITHMS"
fi

echo "Checking done markers and AIG counts in: $FULL_DATASET"
echo "Tier: $TIER    Expected designs: $expected_count"
echo "Algorithms: ${ALGO_LIST[*]}"
echo ""

overall_missing=0
overall_aigs=0
printf "%-16s %-8s %-8s %-8s\n" "Algorithm" "Done" "Missing" "AIGs"
printf "%s\n" "-------------------------------------------------"
for alg in "${ALGO_LIST[@]}"; do
  # done markers
  done_dir="$FULL_DATASET/optimized_aigs/done/$alg/tier$TIER"
  if [ ! -d "$done_dir" ]; then
    present=0
    present_list=()
  else
    mapfile -t present_files < <(find "$done_dir" -maxdepth 1 -type f -name '*.done' -printf '%f\n' | sort)
    present_list=()
    for f in "${present_files[@]}"; do
      present_list+=("${f%.done}")
    done
    present=${#present_list[@]}
  fi

  missing_count=$((expected_count - present))
  if [ "$missing_count" -lt 0 ]; then missing_count=0; fi

  # count .aig files under optimized_aigs/<alg>/tierN
  aig_dir="$FULL_DATASET/optimized_aigs/$alg/tier$TIER"
  if [ -d "$aig_dir" ]; then
    aig_count=$(find "$aig_dir" -type f -name '*.aig' | wc -l | tr -d ' ')
  else
    aig_count=0
  fi

  overall_aigs=$((overall_aigs + aig_count))

  printf "%-16s %-8d %-8d %-8d\n" "$alg" "$present" "$missing_count" "$aig_count"

  if [ "$LIST_MISSING" = true ]; then
    if [ "$present" -eq 0 ]; then
      printf "  All designs missing for %s (no .done files found)\n" "$alg"
    else
      for d in "${EXPECTED[@]}"; do
        skip=false
        for p in "${present_list[@]}"; do
          if [ "$d" = "$p" ]; then skip=true; break; fi
        done
        if [ "$skip" = false ]; then printf "  MISSING: %s\n" "$d"; overall_missing=$((overall_missing+1)); fi
      done
    fi
  else
    overall_missing=$((overall_missing + missing_count))
  fi
done

echo ""
printf "Total AIGs across algorithms (tier %s): %d\n" "$TIER" "$overall_aigs"

# Report on done-markers summary but continue to metadata checks
if [ "$overall_missing" -eq 0 ]; then
  echo "All done markers present for requested algorithms/tiers."
else
  echo "Total missing designs: $overall_missing" >&2
fi

# ---- Per-design metadata CSV validation ----
echo "\nPerforming per-design metadata CSV validation (metadata/stats/*.csv)"
missing_rows=0
missing_files=0
mismatch_count=0

for design in "${EXPECTED[@]}"; do
  csv_path="$FULL_DATASET/metadata/stats/${design}.csv"
  if [ ! -f "$csv_path" ]; then
    echo "MISSING CSV: $csv_path" >&2
    missing_files=$((missing_files+1))
    continue
  fi

  header=$(head -n1 "$csv_path" | tr -d '\r')
  IFS=',' read -r -a hdr_cols <<< "$header"
  # find index of algorithm and output_tier columns
  alg_idx=-1; tier_idx=-1
  for i in "${!hdr_cols[@]}"; do
    col=${hdr_cols[$i]}
    if [ "$col" = "algorithm" ]; then alg_idx=$((i+1)); fi
    if [ "$col" = "output_tier" ] || [ "$col" = "outputTier" ]; then tier_idx=$((i+1)); fi
  done
  if [ $alg_idx -eq -1 ]; then
    echo "CSV $csv_path missing 'algorithm' column" >&2
    missing_rows=$((missing_rows+1))
    continue
  fi

  # For each algorithm, count rows in CSV matching algorithm and tier (if tier column exists)
  for alg in "${ALGO_LIST[@]}"; do
    if [ $tier_idx -ne -1 ]; then
      csv_count=$(awk -F',' -v aidx=$alg_idx -v tidx=$tier_idx -v alg="$alg" -v t="tier$TIER" 'NR>1 && $aidx==alg && $tidx==t {count++} END{print (count+0)}' "$csv_path")
    else
      csv_count=$(awk -F',' -v aidx=$alg_idx -v alg="$alg" 'NR>1 && $aidx==alg {count++} END{print (count+0)}' "$csv_path")
    fi

    # count AIGs for this design & alg
    aigd="${FULL_DATASET}/optimized_aigs/${alg}/tier${TIER}/${design}"
    if [ -d "$aigd" ]; then
      aig_count=$(find "$aigd" -type f -name '*.aig' | wc -l | tr -d ' ')
    else
      aig_count=0
    fi

    if [ "$csv_count" -lt "$aig_count" ]; then
      echo "MISMATCH: design=$design alg=$alg CSV_rows=$csv_count AIGs=$aig_count" >&2
      mismatch_count=$((mismatch_count+1))
    fi
  done
done

echo "\nPer-design CSV summary: missing_files=$missing_files mismatches=$mismatch_count missing_headers=$missing_rows"

if [ $missing_files -ne 0 ] || [ $mismatch_count -ne 0 ] || [ $missing_rows -ne 0 ]; then
  echo "Check failed: some designs have missing CSVs or mismatches." >&2
  exit 6
else
  echo "All per-design metadata CSVs look consistent with created AIGs."
fi
