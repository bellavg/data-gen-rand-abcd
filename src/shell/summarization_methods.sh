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
# hand (and any --dependency chains already queued) are positional.
#
# Ordered as in the thesis: control, domain-specific, adapted SOTA, SOTA,
# classic control, cheap control.
METHODS=(identity cone wl convmatch spectral lsh)

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
