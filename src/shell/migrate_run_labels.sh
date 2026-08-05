#!/bin/bash
# Rename checkpoint and log directories onto config.run_label_for's scheme, and
# split the one directory that two different runs shared.
#
#   bash src/shell/migrate_run_labels.sh          # dry run (default)
#   APPLY=1 bash src/shell/migrate_run_labels.sh  # actually move
#
# Runs on a login node. It only moves; it never deletes or overwrites.
#
# WHY
# ---
# train.py, test.py and benchmark.py each carried their own copy of the
# directory-naming rule and they drifted. train.py labelled by method alone, so
#
#     --partition random   ->  Orchestrate_random
#     --split_by  random   ->  Orchestrate_random
#
# were the same directory. Both runs wrote save_top_k=3 checkpoints into it and
# test.py's "best" selection took the lowest val_loss across both, which was the
# partitioning run's epoch 3 (0.0031) rather than the splitting run's epoch 16
# (0.0048). RQ1a's random-split row was measured on the partitioning model.
#
# The rule now lives once, in config.run_label_for:
#
#     <algorithm>_<reduction_type>_<method>, and <algorithm>_none_<split_by>
#     for a non-default split. All defaults stays bare <algorithm>.
#
# NEW LAYOUT
# ----------
#     Orchestrate                                 (unchanged: baseline, design split)
#     Orchestrate_sparsification_<method>         was Orchestrate_<method>
#     Orchestrate_partition_<method>              was Orchestrate_<method>
#     Orchestrate_none_recipe                    was Orchestrate_recipe
#     Orchestrate_none_random                    was inside Orchestrate_random
#
# Orchestrate_random is the only directory holding two runs. Its files are
# separated by mtime: the partitioning run is 2026-07-03/04, the splitting run
# is 2026-08-01/04, and they never overlapped. Every checkpoint whose filename
# carries an epoch and val_loss is additionally checked against what each run's
# W&B history says it kept; anything matching neither is left behind in
# Orchestrate_random rather than guessed at.
#
# ORDER
# -----
# Run this only together with the code change (config.run_label_for plus the
# train/test/benchmark callers). Directories and code have to move at the same
# time or eval will not find its checkpoints.

set -euo pipefail

APPLY="${APPLY:-0}"
TRAIN_ROOT="${TRAIN_ROOT:-/scratch-shared/$USER/aig_train_run}"
ALGORITHM="${ALGORITHM:-Orchestrate}"

CKPT_ROOT="$TRAIN_ROOT/$ALGORITHM/checkpoints"
LOG_ROOT="$TRAIN_ROOT/$ALGORITHM"

# Everything written before this belongs to the partitioning run; everything
# after, to the splitting-protocol run.
CUTOFF="${CUTOFF:-2026-07-20 00:00:00}"

# W&B run ids, so the log directories move by identity rather than by date.
PARTITION_WANDB_ID="${PARTITION_WANDB_ID:-6wcya9z3}"
SPLIT_WANDB_ID="${SPLIT_WANDB_ID:-zpf7d6yi}"

# Checkpoints each run is known to have kept, "epoch:val_loss" at the 4dp the
# filename carries, read off the W&B histories.
PARTITION_KEPT="00:0.0045 02:0.0049 03:0.0031"
SPLIT_KEPT="13:0.0054 15:0.0052 16:0.0048"

# Straight renames: "<old dir> <new dir>".
RENAMES=(
    "${ALGORITHM}_and_gate_only        ${ALGORITHM}_sparsification_and_gate_only"
    "${ALGORITHM}_random_edge_dropout  ${ALGORITHM}_sparsification_random_edge_dropout"
    "${ALGORITHM}_spanning_forest      ${ALGORITHM}_sparsification_spanning_forest"
    "${ALGORITHM}_pagerank             ${ALGORITHM}_sparsification_pagerank"
    "${ALGORITHM}_metis                ${ALGORITHM}_partition_metis"
    "${ALGORITHM}_level_slicing        ${ALGORITHM}_partition_level_slicing"
    "${ALGORITHM}_span_weighted_metis  ${ALGORITHM}_partition_span_weighted_metis"
    "${ALGORITHM}_recipe               ${ALGORITHM}_none_recipe"
)

declare -a MOVE_SRC=() MOVE_DST=()

echo "=========================================="
echo "MIGRATE RUN LABELS"
echo "=========================================="
echo "Checkpoints: $CKPT_ROOT"
echo "Logs       : $LOG_ROOT"
echo "Cutoff     : $CUTOFF"
[[ "$APPLY" == "1" ]] && echo "Mode       : APPLY" || echo "Mode       : DRY RUN (set APPLY=1 to move)"
echo

[[ -d "$CKPT_ROOT" ]] || { echo "ERROR: $CKPT_ROOT does not exist."; exit 1; }

# --- 1. straight renames ------------------------------------------------------
echo "--- Directory renames ---"
for entry in "${RENAMES[@]}"; do
    read -r old new <<< "$entry"
    for root in "$CKPT_ROOT/%s" "$LOG_ROOT/logs_%s"; do
        # shellcheck disable=SC2059
        src="$(printf "$root" "$old")"
        # shellcheck disable=SC2059
        dst="$(printf "$root" "$new")"
        [[ -d "$src" ]] || continue
        if [[ -e "$dst" ]]; then
            echo "  SKIP  $(basename "$src") -> $(basename "$dst") (destination exists)"
            continue
        fi
        printf '  %-52s -> %s\n' "$(basename "$src")" "$(basename "$dst")"
        MOVE_SRC+=("$src"); MOVE_DST+=("$dst")
    done
done
echo

# --- 2. the shared directory --------------------------------------------------
SRC_CKPT="$CKPT_ROOT/${ALGORITHM}_random"
DST_PARTITION="$CKPT_ROOT/${ALGORITHM}_partition_random"
DST_SPLIT="$CKPT_ROOT/${ALGORITHM}_none_random"

kept_by() {
    local list epoch="$2" val="$3"
    case "$1" in
        partition) list="$PARTITION_KEPT" ;;
        split)     list="$SPLIT_KEPT" ;;
        *)         return 1 ;;
    esac
    [[ " $list " == *" ${epoch}:${val} "* ]]
}

left_behind=0
if [[ -d "$SRC_CKPT" ]]; then
    echo "--- Splitting $(basename "$SRC_CKPT") (two runs shared it) ---"
    while IFS= read -r -d '' path; do
        name="$(basename "$path")"
        mtime="$(date -r "$path" '+%Y-%m-%d %H:%M')"

        if [[ -n "$(find "$path" -maxdepth 0 -newermt "$CUTOFF" -print -quit)" ]]; then
            bucket="split"; dest="$DST_SPLIT"
        else
            bucket="partition"; dest="$DST_PARTITION"
        fi

        if [[ "$name" =~ ^epoch=([0-9]+)-val_loss=val_loss=([0-9.]+)\.ckpt$ ]]; then
            if kept_by "$bucket" "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}"; then
                note="confirmed against W&B history"
            else
                # Not a checkpoint either run reports keeping. Three last*.ckpt
                # files mean three runs wrote here, so a third source exists.
                bucket="LEFT IN PLACE"; dest=""
                note="matches neither run; inspect before moving"
                left_behind=$((left_behind + 1))
            fi
        elif [[ "$bucket" != "split" ]]; then
            # last*.ckpt carries no epoch or val_loss, so mtime is the only
            # evidence, and mtime cannot separate the partitioning run from the
            # unidentified third writer: both are July. Only the August one is
            # unambiguous. The rest stay put, which costs nothing --
            # resolve_checkpoint_path ignores any filename without a val_loss
            # in it, so a last*.ckpt is only ever needed to resume training.
            bucket="LEFT IN PLACE"; dest=""
            note="no epoch in name and two July writers; cannot attribute"
            left_behind=$((left_behind + 1))
        else
            note="only August writer is the splitting run; mtime is decisive"
        fi

        printf '  %-42s %s  -> %-14s (%s)\n' "$name" "$mtime" "$bucket" "$note"
        [[ -n "$dest" ]] && { MOVE_SRC+=("$path"); MOVE_DST+=("$dest/$name"); }
    done < <(find "$SRC_CKPT" -maxdepth 1 -name '*.ckpt' -print0 | sort -z)
    echo
fi

SRC_LOGS="$LOG_ROOT/logs_${ALGORITHM}_random"
if [[ -d "$SRC_LOGS/wandb" ]]; then
    echo "--- Splitting $(basename "$SRC_LOGS")/wandb (by W&B run id) ---"
    while IFS= read -r -d '' run_dir; do
        name="$(basename "$run_dir")"
        case "$name" in
            *"$PARTITION_WANDB_ID") dest="$LOG_ROOT/logs_${ALGORITHM}_partition_random/wandb"; bucket="partition" ;;
            *"$SPLIT_WANDB_ID")     dest="$LOG_ROOT/logs_${ALGORITHM}_none_random/wandb";     bucket="split" ;;
            *)                      dest="";                                                   bucket="LEFT IN PLACE (unknown run id)" ;;
        esac
        printf '  %-46s -> %s\n' "$name" "$bucket"
        [[ -n "$dest" ]] && { MOVE_SRC+=("$run_dir"); MOVE_DST+=("$dest/$name"); }
    done < <(find "$SRC_LOGS/wandb" -maxdepth 1 -mindepth 1 -type d -name 'run-*' -print0 | sort -z)
    echo "  (latest-run symlink and debug*.log stay put: they point at whichever"
    echo "   run finished last and carry no per-run identity.)"
    echo
fi

# --- 3. apply -----------------------------------------------------------------
if [[ "$left_behind" -gt 0 ]]; then
    echo "NOTE: $left_behind checkpoint(s) belong to neither known run and stay in"
    echo "      $(basename "$SRC_CKPT"). Nothing is deleted. Expect at least one:"
    echo "      the three last*.ckpt files mean three runs wrote to that directory."
    echo
fi

# Anything the rename table does not name is reported rather than ignored: a
# directory nobody recognises is how a stale or hand-made checkpoint dir gets
# read as a result later.
echo "--- Directories left untouched ---"
KNOWN="$ALGORITHM ${ALGORITHM}_random ${ALGORITHM}_partition_random ${ALGORITHM}_none_random"
for entry in "${RENAMES[@]}"; do
    read -r old new <<< "$entry"
    KNOWN="$KNOWN $old $new"
done
while IFS= read -r -d '' dir; do
    name="$(basename "$dir")"
    if [[ " $KNOWN " == *" $name "* ]]; then
        [[ "$name" == "$ALGORITHM" ]] && echo "  $name (baseline, design split: correct as is)"
    else
        echo "  $name  <- NOT RECOGNISED. Belongs to no known configuration; check what wrote it."
    fi
done < <(find "$CKPT_ROOT" -maxdepth 1 -mindepth 1 -type d -print0 | sort -z)
echo

echo "${#MOVE_SRC[@]} item(s) to move."
if [[ "$APPLY" != "1" ]]; then
    echo
    echo "DRY RUN. Re-run with APPLY=1 once the tables above read correctly."
    exit 0
fi

for i in "${!MOVE_SRC[@]}"; do
    src="${MOVE_SRC[$i]}"; dst="${MOVE_DST[$i]}"
    if [[ -e "$dst" ]]; then
        echo "REFUSING to overwrite $dst"
        exit 1
    fi
    mkdir -p "$(dirname "$dst")"
    mv -n "$src" "$dst"
    echo "moved $(basename "$src") -> $dst"
done

echo
echo "=========================================="
echo "DONE"
echo "=========================================="
echo "Verify the new layout, then re-run the RQ1a random eval. The directory"
echo "now holds only that run's checkpoints, so plain --checkpoint_filename"
echo "best resolves to epoch 16 (val_loss 0.0048) with no pinning needed."
echo
echo "Do NOT pass SPLIT_BY on the sbatch line: no env-var form propagates on"
echo "this cluster, --export=ALL,VAR=value included. The job runs with the"
echo "script's default and still reports success. Key it off"
echo "SLURM_ARRAY_TASK_ID the way train_no_sparsification.sh:70 does."
echo
echo "Check the resolved path in the job log before trusting the result:"
echo "  [test] device=cuda checkpoint=.../${ALGORITHM}_none_random/epoch=16-...0.0048.ckpt"
