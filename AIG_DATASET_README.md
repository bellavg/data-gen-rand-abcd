# Thesis Project Data Documentation

## TODOs
- Finish downloading original OPENABCD 
- Run job 4s to apply synthesis scripts to all random 


## Thesis Project Overview

### Directory Layout


### Models 

X - [node features (types), edge indices, edge features(types)]


Y - [node optimizability, depth optimizability]

## Thesis Project - Data Summary
This project combines AIGs from two ingestion paths (synthetic random AIGs, and designs converted from BENCH netlists) into a unified dataset for downstream ML experiments and algorithm comparison. By provenance the corpus spans four benchmark collections plus the synthetic designs: 29 OpenABC-D IP designs, 8 EPFL arithmetic circuits, 4 ISCAS-85 circuits, and 6 MCNC/LGSynth circuits (all 47 obtained as BENCH netlists from the OpenABC-D distribution), plus 8 synthetic random AIGs.



### Per-Algorithm Parameters

The canonical per-algorithm defaults used by the dataset are summarized below. The automation pipeline provides command templates (see `creation/automate_bulkOptimization.py`) that invoke `abc`; ABC's internal defaults apply unless overridden by the template or `abc.rc`.

- `Orchestrate`:
- `Orchestrate`:
  - defaults: `deterministic=true` (default), `max_passes=20` (default), `score_metric=area_delay` (default)
  - command template: `abc -c 'source {abc_rc}; read {input_aig}; print_stats; strash; orchestrate; print_stats; write {output_aig}'` (inputted)

- `Deepsyn`:
- `Deepsyn`:
  - defaults: `record_seed=true` (default), `seed_mode=deterministic_by_input_path` (default), `max_passes=20` (default)
  - runtime timeout (automation): `timeout_seconds=20` (inputted)
  - command template: `abc -c 'source {abc_rc}; read {input_aig}; print_stats; strash; &get; &deepsyn -S {seed} -T {timeout_seconds}; &put; print_stats; write {output_aig}'` (inputted)

- `Syn4`:
- `Syn4`:
  - defaults: `deterministic=true` (default), `max_passes=4` (default), `flow_name=syn4_default` (default)
  - command template: `abc -c 'source {abc_rc}; read {input_aig}; print_stats; strash; &get; &syn4; &put; print_stats; write {output_aig}'` (inputted)

- `C2RS`:
- `C2RS`:
  - defaults: `deterministic=true` (default), `use_abc_l_flag=true` (via alias) (default)
  - note: invoked via alias `c2rs` from `abc.rc`
  - command template: `abc -c 'source {abc_rc}; read {input_aig}; print_stats; strash; c2rs; print_stats; write {output_aig}'` (inputted)

  - `c2rs` alias (exact definition in `abc.rc`):

    ```text
    alias c2rs        "b -l; rs -K 6 -l; rw -l; rs -K 6 -N 2 -l; rf -l; rs -K 8 -l; b -l; rs -K 8 -N 2 -l; rw -l; rs -K 10 -l; rwz -l; rs -K 10 -N 2 -l; b -l; rs -K 12 -l; rfz -l; rs -K 12 -N 2 -l; rwz -l; b -l"
    ```

    Defined in [abc.rc].

Notes:
- The automation `CONFIG` supplies the command templates and the Deepsyn `timeout_seconds`; other per-algorithm parameters are set by ABC or by algorithm options in `{abc_rc}`.
- If you want, I can link to or embed the exact `CONFIG` snippet from `creation/automate_bulkOptimization.py` here for clarity.

## Directory Layout


