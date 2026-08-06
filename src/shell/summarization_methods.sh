#!/bin/bash
# Shared method list for the summarization jobs.  Sourced, never executed.
#
# Both precompute_summarization.sh and train_summarization.sh turn a Slurm
# array index into a method name through this list, so it has to be one
# source of truth: if the two ever disagreed, submitting a range that means
# "cone" to one script would mean something else to the other, and the run
# would look healthy while measuring the wrong method.
#
# APPEND ONLY — never reorder or remove.  The array ranges people submit by
# hand (and any --dependency chains already queued) are positional.  The list
# was re-cut once, on 2026-08-02, when identity, spectral and lsh were deleted
# from the study: every index below moved, so a range noted down before that
# date means a different method now.
#
# Ordered as in the thesis: domain-specific, adapted SOTA, SOTA — then
# wl_exact, appended afterwards because the list is append-only, so its slot is
# wherever it lands rather than where it belongs thematically (next to wl).
#
# wl_exact runs the SAME clustering as wl but through the exact rewrite
# (data/exact_graph.py) instead of the lossy apply_merge_map, and is the only
# method trained with `--model exact`.  data.summarization.EXACT_METHODS is
# the single source of truth for which methods those are;
# train_summarization.sh queries it directly rather than restating it here.
METHODS=(cone wl convmatch wl_exact)

# Shards per method in precompute_summarization.sh.  Its array index packs
# both coordinates: method = index / SHARDS_PER_METHOD, shard = index %
# SHARDS_PER_METHOD.  train_summarization.sh uses one task per method and
# ignores this.
SHARDS_PER_METHOD=32

# Print "<method> <first-index> <last-index>" per method, for the range
# tables the two scripts show in their headers and error messages.
summarization_ranges() {
    local index=0
    for method in "${METHODS[@]}"; do
        echo "$method $(( index * SHARDS_PER_METHOD )) $(( (index + 1) * SHARDS_PER_METHOD - 1 ))"
        index=$(( index + 1 ))
    done
}
