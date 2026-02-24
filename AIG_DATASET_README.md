# Thesis Project Data Documentation

## TODOs
- Finish downloading original OPENABCD 
- Run job 4s to apply synthesis scripts to all random 

## Thesis Project Data — Summary
This project combines two AIG sources into a unified dataset for downstream ML experiments and algorithm comparison.

- Random AIG dataset: 8 synthetic random start designs (sizes 128–16384) — 252,008 AIGs (per design: 31,501 AIGs — 1 original + 1,500 × 21 = 31,500 synthesized; total 31,501)
- Converted OpenABC-D: 29 real IP designs converted from BENCH to AIG — 913,529 AIGs (per design: 31,501 AIGs — 1 original + 1,500 × 21 = 31,500 synthesized; total 31,501)

Total base AIGs (exact): 1,165,537 (≈1.17M)

Note: The Random and Converted OpenABC-D datasets are the two source datasets. The "Full Dataset" described in this repository is the combined (union) dataset created by merging those two sources and is the primary dataset for this project.

Planned experiment pipeline (high level):

  1) For every base AIG, apply four AIG optimization algorithms (Tier‑1):
    - Orchestrate, Deepsyn (with random seed recorded), Syn4, C2RS
    - This produces 4 × base_count Tier‑1 outputs: 4,662,148 outputs (≈4.66M files)

  2) For every Tier‑1 AIG, re-apply the same four algorithms (Tier‑2), using the same timing and hyperparameters as the first pass:
    - This produces 4 × Tier1_count Tier‑2 outputs: 18,648,592 outputs (≈18.65M files)

Storage/scale note: the two-tier expansion is very large (tens of millions of files). Plan storage, I/O and compute accordingly.

Visual pipeline (ASCII):

  [Base AIGs 1,165,537 (≈1.17M)]
      |
      |-- apply 4 algos --> [Tier-1 AIGs 4,662,148 (≈4.66M)]
                    |
                    |-- apply 4 algos --> [Tier-2 AIGs 18,648,592 (≈18.65M)]

All algorithms use the same timing constraints and hyperparameters per your plan; Deepsyn runs will record the RNG seed used for reproducibility.

## Optimization Parameters / Tier Configuration

- **Same parameter policy for Tier‑1 and Tier‑2:** Yes
- **Algorithms:** `Orchestrate`, `Deepsyn`, `Syn4`, `C2RS`
- **Tier‑1 input set:** `FULL_DATASET/base_aigs/{design}/{design}_orig.aig` and/or AIGs extracted from `FULL_DATASET/base_aigs/{design}/syn*.zip`
- **Tier‑2 input set:** `FULL_DATASET/optimized_aigs/{algorithm}/tier1/**.aig`
- **Tier outputs:** `FULL_DATASET/optimized_aigs/{algorithm}/tier{1|2}/{design}/...`

### Per-Algorithm Parameters
#TODO check if true and replace with real values
Per-algorithm defaults:

- `Orchestrate`:
  - `engine=orchestrate`
  - `deterministic=true`
  - `max_passes=20`
  - `score_metric=area_delay`
- `Deepsyn`:
  - `engine=deepsyn`
  - `record_seed=true`
  - `seed_mode=deterministic_by_input_path`
  - `max_passes=20`
- `Syn4`:
  - `engine=syn4`
  - `flow_name=syn4_default`
  - `max_passes=4`
  - `deterministic=true`
- `C2RS`:
  - `engine=abc_alias`
  - `alias=c2rs` (from `abc.rc`)
  - `use_abc_l_flag=true`
  - `deterministic=true`

## Full Dataset Naming Schema and Directory Structure

Directory layout (Full Dataset):

```
FULL_DATASET/
├─ base_aigs/                             
│  ├─ ac97_ctrl/                      # example design folder
│  │  ├─ ac97_ctrl_orig.aig
│  │  ├─ syn0.zip                     # contains ac97_ctrl_syn0_step{1..21}.aig
│  │  ├─ syn1.zip                     # contains ac97_ctrl_syn1_step{1..21}.aig
│  │  ├─ ...                  
│  │  └─ syn1499.zip
│  ├─ ...
│  └─ {design}/                   # repeated for each design 
│     ├─ {design}_orig.aig        # original AIG
│     ├─ ...                     
│     └─ syn{recipe_id}.zip       # each zip stores step AIGs for that recipe
├─ synScripts/                    # zipped synthesis scripts per design 
|  ├─ ...
│  └─ {design}.zip                # inside: abc{recipe_id}.script #todo add line 
├─ optimized_aigs/                # algorithm outputs (tiered)
|  ├─ ...
│  └─ {algorithm}/               
|     ├─ ...
│     └─ tier{tier_id}/                 # per tier 
|        ├─ ...
|        └─ {design}/        # save as zip     
|           ├─ ...
|           └─ {design}_syn{recipe_id}_step{step_id}.aig
└─ metadata/                     
  ├─ stats/
  |  ├─ ...                     
  │  └─ {design}.csv             # per-design CSV (one row per AIG)
  └─ library/
    └─ nangate45.lib.zip

```


Variable ranges / notes (for `{...}` values used above):
- `design`: `128, 256, 512, 1024, 2048, 4096, 8192, 16384, i2c, spi, des3_area, ss_pcm, usb_phy, sasc, wb_dma, simple_spi, dynamic_node, aes, pci, ac97_ctrl, mem_ctrl, tv80, fpu, wb_conmax, tinyRocket, aes_xcrypt, aes_secworks, jpeg, bp_be, ethernet, vga_lcd, picosoc, dft, idft, fir, iir, sha256` (random-size names plus the 29 OpenABC‑D designs in one list).
- `recipe_id`: synthesis recipe identifier. Range: `0..1499`.
- `step_id`: per-recipe step index. Range: `1..21` (synthesized steps). Base AIGs use `{design}_orig.aig`.
- `tier_id`: generation tier for algorithm outputs. Values: `1` = first-pass, `2` = second-pass. Base AIGs should have an empty `tier_id` in per-design CSV rows.
- `algorithm`:  `Orchestrate`, `Deepsyn`, `Syn4`, `C2RS`



## AIG Statistics in `metadata/stats/{design}.csv`

Canonical CSV header (exact — tools should emit this header, comma-separated, no extra spaces):

```
file_path,design,recipe_id,step_id,tier_id,algorithm,nodes,edges,num_PI,num_PO,depth,avg_fanout,max_fanout
```

Canonical column definitions

| Column name | Type | Description |
|---|---:|---|
| `file_path` | string | Canonical relative AIG path (e.g. `base_aigs/ac97_ctrl/ac97_ctrl_syn0_step1.aig`). In zip-preserving storage this logical path resolves inside `base_aigs/ac97_ctrl/syn0.zip`. |
| `design` | string | Design identifier (e.g. `128`, `ac97_ctrl`). |
| `recipe_id` | integer | Synthesis recipe identifier (0..1499). |
| `step_id` | integer | Per-recipe step index. Parse as integer — some datasets use 0-based (0..20) while converted OpenABC‑D uses 1-based (1..21); accept both when ingesting. |
| `tier_id` | integer | Generation tier for algorithm outputs: `0` = base aigs. `1` = first-pass, `2` = second-pass. `3` = final aig, the graphs made from tier 2 second pass. |
| `algorithm` | string or empty | Optimization algorithm used to generate the graph: `Orchestrate`, `Deepsyn`, `Syn4`, `C2RS`. Base AIG rows should have an empty value. |
| `nodes` | integer | Number of internal AIG nodes (integer count). |
| `edges` | integer | Number of edges in the AIG (integer count). |
| `num_PI` | integer | Number of primary inputs. |
| `num_PO` | integer | Number of primary outputs. |
| `depth` | integer | Estimated combinational depth (length of the longest path, measured in AIG nodes). |
| `avg_fanout` | float | Average fanout per node (floating point). |
| `max_fanout` | integer | Maximum fanout observed (integer). |


## Random AIG Dataset Documentation
This dataset contains synthesized AIG (And-Inverter Graph) files generated using ABC for 8 different circuit designs of varying sizes.

### Dataset Structure

#### Random Designs
The dataset includes 8 random designs, named by their size:
- `128` 
- `256`  
- `512` 
- `1024` 
- `2048` 
- `4096` 
- `8192` 
- `16384` 

#### Directory Structure for the Original Random AIG 
```
OPENABC_DATASET/
├── bench/
│   ├── 128/
│   │   ├── 128_orig.aig              # Original AIG file
│   │   ├── syn0.zip                  # Synthesis recipe 0 results (21 AIG files)
│   │   ├── syn1.zip                  # Synthesis recipe 1 results (21 AIG files)
│   │   ├── ...
│   │   ├── syn1499.zip               # Synthesis recipe 1499 results (21 AIG files)
│   │   ├── metadata/                 # Metadata CSV files
│   │   │   └── 128.csv               # CSV with logical statistics (~31,500 rows)
│   │   └── log_128/                  # ABC synthesis logs
│   │       ├── log_128_syn0.log      # ABC log for recipe 0
│   │       ├── log_128_syn1.log      # ABC log for recipe 1
│   │       └── ... (1500 log files)
│   ├── 256/
│   │   └── [same structure]
│   └── ... [remaining 6 designs]
├── synScripts/
│   ├── 128/
│   │   ├── abc0.script               # ABC synthesis script 0 with metadata capture
│   │   ├── abc1.script               # ABC synthesis script 1 with metadata capture
│   │   └── ... (1500 script files)
│   └── ... [remaining 7 designs]
└── lib/
    └── (empty - library stored separately)
```

**Note to self:** The standard cell library is stored separately at:
`/scratch-shared/$USER/openabc_full/OPENABC_DATASET/lib/nangate45.lib`

### File Naming Convention

#### Original AIG Files
- **Format:** `{design}_orig.aig`
- **Examples:** `128_orig.aig`, `256_orig.aig`, `16384_orig.aig`
- **Location:** `OPENABC_DATASET/bench/{design}/`

-#### Synthesized AIG Files
- **Format:** `{design}_syn{recipe}_step{step_id}.aig`
- **Examples:** 
  - `128_syn0_step1.aig` - Design 128, recipe 0, step 1
  - `256_syn42_step15.aig` - Design 256, recipe 42, step 15
  - `16384_syn1499_step21.aig` - Design 16384, recipe 1499, step 21
- **Location:** Inside `syn{recipe}.zip` files

#### Compressed Archives
- **Format:** `syn{recipe}.zip`
- **Range:** `syn0.zip` to `syn1499.zip`
- **Location:** `OPENABC_DATASET/bench/{design}/`

### Dataset Size

#### Per Design
- **Original AIG files:** 1
- **Synthesis recipes:** 1,500
- **Steps per recipe:** 21 (step1 through step21)
- **Total AIGs per design:** 31,501 (1 original + 1,500 × 21 = 31,500 synthesized; total 31,501)
- **Zip archives:** 1,500
- **Log files:** 1,500
- **Statistics files:** 0 (metadata captured directly to CSV)
- **Metadata CSV:** 1 per design with canonical format

#### Total Dataset (All 8 Designs)
- **Original AIG files:** 8
- **Total synthesis recipes:** 12,000 (8 × 1,500)
- **Total base AIGs:** 252,008 (8 × 31,501)
- **Total synthesized AIGs:** 252,000 (8 × 31,500)
- **Total zip archives:** 12,000
- **Total log files:** 12,000

### Numbering Scheme

#### Recipe Numbers
- **Range:** 0 to 1499
- **Total:** 1,500 recipes per design
- All designs use the same recipe numbers for consistency

#### Step Numbers
- **Range:** 1 to 21
- **Total:** ~21 steps per recipe
- **step1:** Result after initial `strash` (structural hashing)
- **step2-20:** Intermediate optimization steps
- **step21:** Final optimized circuit (before mapping)
 - **orig:** Original AIG (stored as `{design}_orig.aig` in `base_aigs/{design}/`).
 - **step1:** Result after initial `strash` (structural hashing) — this is the first synthesized step.
 - **step2-20:** Intermediate optimization steps.
 - **step21:** Final optimized circuit (before mapping).

### Source Data: OpenABC-D Converted to AIG

## Original OpenABC-D Dataset
The source dataset contains **29 open-source hardware IP designs** from various sources (MIT-CEP, IWLS, OpenROAD, OpenPiton):

**Design Set 1 (8 designs):**
- `i2c`, `spi`, `des3_area`, `ss_pcm`, `usb_phy`, `sasc`, `wb_dma`, `simple_spi`

**Design Set 2 (7 designs):**
- `dynamic_node`, `aes`, `pci`, `ac97_ctrl`, `mem_ctrl`, `tv80`, `fpu`

**Design Set 3 (4 designs):**
- `wb_conmax`, `tinyRocket`, `aes_xcrypt`, `aes_secworks`

**Design Set 4 (5 designs):**
- `jpeg`, `bp_be`, `ethernet`, `vga_lcd`, `picosoc`

**Design Set 5 (5 designs):**
- `dft`, `idft`, `fir`, `iir`, `sha256`

### OpenABC-D Statistics (Converted to AIG)
- **Total Designs:** 29 open-source hardware IPs
- **Synthesis Recipes per Design:** 1,500
- **Total Synthesis Runs:** 43,500 (29 × 1,500)
- **Steps per Recipe:** 21 (step1 through step21)
- **Total synthesized AIG files:** 913,500 (29 × 1,500 × 21)
- **Total base AIG files including originals:** 913,529 (913,500 synthesized + 29 originals)
- **Original Format:** BENCH files (converted to AIG using ABC)
- **Conversion Command:** `read_bench {file}.bench; strash; write {file}.aig`

### Original OpenABC-D layout (as downloaded)


```
/scratch-shared/igardner1/openabc_full/
│
├── OPENABC_DATASET/                 <-- THE PRODUCTION ROOT
│   ├── lib/                         <-- PHYSICAL LIBRARIES (.lib, .v)
│   ├── statistics/                  <-- THE "LABELS" (CSV, PKL)
│   ├── synScripts.zip               <-- THE "RECIPES" (ABC commands)
│   └── bench/                       <-- THE CORE DATA (AIGs)
│       ├── ac97_ctrl/               <-- 31,501 AIG files
│       ├── aes_secworks/            <-- 31,501 AIG files
│       ├── aes_xcrypt/              <-- 31,501 AIG files
│       └── ... (26 more designs)
│
├── OPENABC_DATASET.zip              <-- MASTER INSTALLER (Can be deleted)
└── OPENABC_DATASET.z01...z13        <-- MASTER PARTS (Can be deleted)
```

Detailed Content Breakdown

| Item | What is inside EXACTLY? | Why do you need it? |
|---|---|---|
| `bench/<design>/*.aig` | Binary AIG files (extracted/converted from BENCH). No .bench or .zip needed here. | These are the core graph inputs for your GNN / ML models. |
| `synScripts.zip` | 1,500 recipe files (ABC command sequences like `rewrite; refactor; resub;`) | If you predict a recipe is best, this archive shows the exact ABC commands executed for that recipe. |
| `statistics/` | Large CSVs and Python pickle files mapping filenames to Area, Delay, Power and other labels. | Ground truth labels for supervised learning — required to train models to predict quality metrics. |
| `lib/` | Technology libraries (e.g. Nangate 45nm `.lib` files). | Required to map AIGs to real timing/area during mapping runs with ABC. |




