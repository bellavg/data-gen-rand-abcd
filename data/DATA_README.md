

# Thesis Project Data — Summary
This project combines AIGs from two ingestion paths (synthetic random AIGs, and designs converted from BENCH netlists) into a unified dataset for downstream ML experiments and algorithm comparison. By provenance the corpus spans four benchmark collections plus the synthetic designs: 29 OpenABC-D IP designs, 8 EPFL arithmetic circuits, 4 ISCAS-85 circuits, and 6 MCNC/LGSynth circuits (all 47 obtained as BENCH netlists from the OpenABC-D distribution), plus 8 synthetic random AIGs.

# Random AIG dataset: 8 synthetic random start designs (sizes 128–16384) — 33,608 AIGs (per design: 4,201 AIGs — 1 original + 200 × 21 = 4,200 synthesized; total 4,201)
# Converted from BENCH: 47 real designs converted from BENCH to AIG — 197,447 AIGs (per design: 4,201 AIGs — 1 original + 200 × 21 = 4,200 synthesized; total 4,201). Only 29 of the 47 are OpenABC-D's own IP designs; the other 18 are classical benchmarks redistributed in OpenABC-D's bench directory (8 EPFL, 4 ISCAS-85, 6 MCNC/LGSynth).

Total base AIGs (exact): 231,055 (≈0.23M)

Note: The random AIGs and the BENCH-converted designs are the two ingestion paths. The "Full Dataset" described in this repository is the combined (union) dataset created by merging them and is the primary dataset for this project.

Planned experiment pipeline (high level):

  1) For every base AIG (Tier-0), apply four AIG optimization algorithms (Tier‑1):
    - Orchestrate, Deepsyn (with random seed recorded), Syn4, C2RS
    - This produces 4 × base_count Tier‑1 outputs: 924,220 outputs (≈0.92M files)

  2) For every Tier‑1 AIG, re-apply only the other three algorithms the graph has not yet been optimized with (i.e., do not re-apply the same algorithm that produced the Tier‑1 AIG), using the same timing and hyperparameters as the first pass:
    - This produces 3 × Tier1_count Tier‑2 outputs: 2,772,660 outputs (≈2.77M files)

Visual pipeline (ASCII):

  [Base AIGs (Tier-0) 231,055 (≈0.23M)]
      |
      |-- apply 4 algos --> [Tier-1 AIGs 924,220 (≈0.92M)]
            |
            |-- apply 3 algos --> [Tier-2 AIGs 2,772,660 (≈2.77M)]

All algorithms use the same timing constraints and hyperparameters; Deepsyn record the RNG seed used for reproducibility.

# Data folder structure

This file maps the current contents of the `data/` folder so you can quickly find scripts,
designs, and job orchestration used to create and optimize the dataset.

```
data/
├─ DATA_README.md                  # this file
├─ abc_scripts/                    # ABC config, scripts and recipe sets
│  ├─ abc.rc
│  ├─ optimization_scripts/        # per-design optimization runner scripts
│  │  └─ {design_name}/
│  │     └─ {algorithm}.sh         # e.g. data/abc_scripts/optimization_scripts/i2c/Orchestrate.sh
│  ├─ reference_scripts/           # reference synthesis scripts 
│  │  ├─ abc0.script
│  │  └─ ... (abc{recipe_id}.script up to abc199.script)
│  └─ synthesis_scripts/           # synthesis recipe scripts (per-design)
│     └─ {design_name}/
│        └─ abc{recipe_id}.script
├─ creation/                       # creation pipeline scripts
│  ├─ 1_bench_to_aig.sh
│  ├─ 2_generate_scripts.sh
│  ├─ 3_run_synthesis.sh
│  ├─ 4_optimize_aigs.sh
│  ├─ 5_make_csv.sh
│  ├─ automate_bulkOptimization.py
│  └─ automate_bulkSynthesis.py
└─ designs/                        # per-design AIG folders (examples)
   ├─ {design_name}/                # e.g. 128, 256, i2c, ac97_ctrl, aes (one folder per design)
   │  ├─ tier0/                     # original/base artifacts
   │  │  ├─ {design_name}_synX_step0.aig  # original AIG moved here (X = placeholder for "no recipe")
   │  │  └─{design_name}_syn{recipe_id}_step{step_id}.aig  
   │  ├─ tier1/                     # algorithm outputs (first pass)
   │  │  └─ {algorithm}/            # e.g. Orchestrate, Deepsyn, Syn4, C2RS
   │  │     └─ {design_name}_{algorithm}_tier1_syn{recipe_id}_step{step_id}.aig  # e.g. i2c_Orchestrate_tier1_syn42_step7.aig
   │  └─ design metadata/                     
   │     ├─ raw_logs/                     # raw log outputs
   │     │  ├─ synthesis_logs/            # ABC synthesis run logs (zipped as {design}_synthesis_logs.zip)
   │     │  │  └─ log_{design}_syn{recipe_id}.log
   │     │  └─ optimization_logs/         # optimizer / orchestrator logs
   │     │     ├─ tier1/
   │     │     │  └─ {algorithm}/
   │     │     │     └─ optimize_{algorithm}_{design}.zip  # contains {design}_{algorithm}_tier1_syn{recipe_id}_step{step}.log
   │     │     └─ tier2/
   │     │        └─ {algorithm}/
   │     │           └─ opt_t2_{algorithm}_{design}.zip   # contains {design}_{tier1_algorithm}_{tier2_algorithm}_tier2_syn{recipe_id}_step{step}.log
   │     └─ {design}.csv                  # per-design stats CSV (one row per AIG)
   └─ ...                          # repeated per-design folders


```
Variable ranges / notes (for `{...}` values used above):
- `design_name`: `128, 256, 512, 1024, 2048, 4096, 8192, 16384, ac97_ctrl, aes, aes_secworks, aes_xcrypt, apex1, bc0, bp_be, c1355, c5315, c6288, c7552, dalu, des3_area, dft, div, dynamic_node, ethernet, fir, fpu, hyp, i10, i2c, idft, iir, jpeg, k2, log2, mainpla, max, mem_ctrl, multiplier, pci, picosoc, sasc, sha256, simple_spi, sin, spi, sqrt, square, ss_pcm, tinyRocket, tv80, usb_phy, vga_lcd, wb_conmax, wb_dma` (random-size names plus the OpenABC designs present in data/abc_scripts/synthesis_scripts).
- `recipe_id`: synthesis recipe identifier. Range: `0..199`.
- `step_id`: per-recipe step index. Range: `1..21` (synthesized steps). Base AIGs use `{design}_orig.aig`.
- `tier_id`: generation tier for algorithm outputs. Values: `1` = first-pass, `2` = second-pass. Base AIGs should have an empty `tier_id` in per-design CSV rows.
- `algorithm`:  `Orchestrate`, `Deepsyn`, `Syn4`, `C2RS`


