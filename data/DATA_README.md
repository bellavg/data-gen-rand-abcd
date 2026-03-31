

# Thesis Project Data — Summary
This project combines two AIG sources into a unified dataset for downstream ML experiments and algorithm comparison.

# Random AIG dataset: 8 synthetic random start designs (sizes 128–16384) — 33,608 AIGs (per design: 4,201 AIGs — 1 original + 200 × 21 = 4,200 synthesized; total 4,201)
# Converted OpenABC-D: 47 real IP designs converted from BENCH to AIG — 197,447 AIGs (per design: 4,201 AIGs — 1 original + 200 × 21 = 4,200 synthesized; total 4,201)

Total base AIGs (exact): 231,055 (≈0.23M)

Note: The Random and Converted OpenABC-D datasets are the two source datasets. The "Full Dataset" described in this repository is the combined (union) dataset created by merging those two sources and is the primary dataset for this project.

Planned experiment pipeline (high level):

  1) For every base AIG(Tier-0), apply four AIG optimization algorithms (Tier‑1):
    - Orchestrate, Deepsyn (with random seed recorded), Syn4, C2RS
    - This produces 4 × base_count Tier‑1 outputs: 924,220 outputs (≈0.92M files)

  2) For every Tier‑1 AIG, re-apply the same four algorithms (Tier‑2), using the same timing and hyperparameters as the first pass:
    - This produces 4 × Tier1_count Tier‑2 outputs: 3,696,880 outputs (≈3.70M files)

Storage/scale note: the two-tier expansion is very large (tens of millions of files). Plan storage, I/O and compute accordingly.

Visual pipeline (ASCII):

  [Base AIGs (Tier-0) 231,055 (≈0.23M)]
      |
      |-- apply 4 algos --> [Tier-1 AIGs 924,220 (≈0.92M)]
                    |
                    |-- apply 4 algos --> [Tier-2 AIGs 3,696,880 (≈3.70M)]

All algorithms use the same timing constraints and hyperparameters per your plan; Deepsyn runs will record the RNG seed used for reproducibility.

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
   │  │     └─ {design_name}_{algorithm}_tier1_syn{recipe_id}_step{step_id}.{ext}  # e.g. i2c_Orchestrate_tier1_syn42_step7.aig
   │  ├─ tier2/                     # algorithm outputs (second pass)
   │  │  └─ {algorithm}/
   │  │     └─ {design_name}_{algorithm}_tier2_syn{recipe_id}_step{step_id}.{ext}  # e.g. i2c_Orchestrate_tier2_syn42_step7.aig
   │  └─ design metadata/                     
  │     ├─ raw_logs/                     # raw log outputs
  │     │  ├─ synthesis_logs/            # ABC synthesis run logs
  │     │  │  └─ log_{design}_syn{recipe_id}.log
  │     │  └─ optimization_logs/         # optimizer / orchestrator logs
  │     │     └─ optimize_{algorithm}_{design}_run{run_id}.log
   │     └─ {design}.csv                  # per-design stats CSV (one row per AIG)
   └─ ...                          # repeated per-design folders

```
Variable ranges / notes (for `{...}` values used above):
- `design_name`: `128, 256, 512, 1024, 2048, 4096, 8192, 16384, ac97_ctrl, aes, aes_secworks, aes_xcrypt, apex1, bc0, bp_be, c1355, c5315, c6288, c7552, dalu, des3_area, dft, div, dynamic_node, ethernet, fir, fpu, hyp, i10, i2c, idft, iir, jpeg, k2, log2, mainpla, max, mem_ctrl, multiplier, pci, picosoc, sasc, sha256, simple_spi, sin, spi, sqrt, square, ss_pcm, tinyRocket, tv80, usb_phy, vga_lcd, wb_conmax, wb_dma` (random-size names plus the OpenABC designs present in data/abc_scripts/synthesis_scripts).
- `recipe_id`: synthesis recipe identifier. Range: `0..199`.
- `step_id`: per-recipe step index. Range: `1..21` (synthesized steps). Base AIGs use `{design}_orig.aig`.
- `tier_id`: generation tier for algorithm outputs. Values: `1` = first-pass, `2` = second-pass. Base AIGs should have an empty `tier_id` in per-design CSV rows.
- `algorithm`:  `Orchestrate`, `Deepsyn`, `Syn4`, `C2RS`


