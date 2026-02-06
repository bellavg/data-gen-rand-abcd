# Thesis Project Data Documentation

## Thesis Project Data — Summary
This project combines two AIG sources into a unified dataset for downstream ML experiments and algorithm comparison:

 - Random AIG dataset: 8 synthetic random start designs (sizes 128–16384) — ~252,000 AIGs (per design ≈ 1 + 1,500×21 ≈31.5k AIGs)
 - Converted OpenABC-D: 29 real IP designs converted from BENCH to AIG — ~913,500 AIGs (per design ≈ 1 + 1,500×21 ≈31.5k AIGs)

Total base AIGs (approx): 1,165,537 (~1.17M)

Planned experiment pipeline (high level):

  1) For every base AIG, apply four AIG optimization algorithms (Tier‑1):
    - Orchestrate, Deepsyn (with random seed recorded), Syn4, C2RS
    - This produces 4 × base_count Tier‑1 outputs (~4.66M files)

  2) For every Tier‑1 AIG, re-apply the same four algorithms (Tier‑2), using the same timing and hyperparameters as the first pass:
    - This produces 4 × Tier1_count Tier‑2 outputs (~18.65M files)

Storage/scale note: the two-tier expansion is very large (tens of millions of files). Plan storage, I/O and compute accordingly.

Visual pipeline (ASCII):

  [Base AIGs ~1.17M]
      |
      |-- apply 4 algos --> [Tier-1 AIGs ~4.66M]
                    |
                    |-- apply 4 algos --> [Tier-2 AIGs ~18.65M]

All algorithms use the same timing constraints and hyperparameters per your plan; Deepsyn runs will record the RNG seed used for reproducibility.

### Random AIG Dataset Documentation
This dataset contains synthesized AIG (And-Inverter Graph) files generated using ABC (Berkeley Logic Synthesis and Verification Tool) for 8 different circuit designs of varying sizes.

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

#### Directory Structure
```
OPENABC_DATASET/
├── bench/
│   ├── 128/
│   │   ├── 128_orig.aig              # Original AIG file
│   │   ├── syn0.zip                  # Synthesis recipe 0 results
│   │   ├── syn1.zip                  # Synthesis recipe 1 results
│   │   ├── ...
│   │   ├── syn1499.zip               # Synthesis recipe 1499 results
│   │   └── log_128/
│   │       ├── log_128_syn0.log      # ABC log for recipe 0
│   │       ├── log_128_syn1.log      # ABC log for recipe 1
│   │       └── ...
│   ├── 256/
│   │   └── [same structure]
│   └── ... [remaining 6 designs]
├── synScripts/
│   ├── 128/
│   │   ├── abc0.script               # ABC synthesis script 0
│   │   ├── abc1.script               # ABC synthesis script 1
│   │   └── ...
│   └── ... [remaining 7 designs]
└── lib/
    └── (empty - library stored separately)
```

**Note:** The standard cell library is stored separately at:
`/scratch-shared/$USER/openabc_full/OPENABC_DATASET/lib/nangate45.lib`

### File Naming Convention

#### Original AIG Files
- **Format:** `{design}_orig.aig`
- **Examples:** `128_orig.aig`, `256_orig.aig`, `16384_orig.aig`
- **Location:** `OPENABC_DATASET/bench/{design}/`

#### Synthesized AIG Files
- **Format:** `{design}_syn{recipe}_step{step}.aig`
- **Examples:** 
  - `128_syn0_step0.aig` - Design 128, recipe 0, step 0
  - `256_syn42_step15.aig` - Design 256, recipe 42, step 15
  - `16384_syn1499_step20.aig` - Design 16384, recipe 1499, step 20
- **Location:** Inside `syn{recipe}.zip` files

#### Compressed Archives
- **Format:** `syn{recipe}.zip`
- **Range:** `syn0.zip` to `syn1499.zip`
- **Location:** `OPENABC_DATASET/bench/{design}/`

### Dataset Size

#### Per Design
- **Original AIG files:** 1
- **Synthesis recipes:** 1,500
- **Steps per recipe:** ~21 (step0 through step20)
- **Total AIGs per design:** ~31,500 (1 original + 1,500 × 21 synthesized)
- **Zip archives:** 1,500
- **Log files:** 1,500

#### Total Dataset (All 8 Designs)
- **Original AIG files:** 8
- **Total synthesis recipes:** 12,000 (8 × 1,500)
- **Total synthesized AIGs:** ~252,000 (8 × 31,500)
- **Total zip archives:** 12,000
- **Total log files:** 12,000

### Numbering Scheme

#### Recipe Numbers
- **Range:** 0 to 1499
- **Total:** 1,500 recipes per design
- All designs use the same recipe numbers for consistency

#### Step Numbers
- **Range:** 0 to 20
- **Total:** ~21 steps per recipe
- **step0:** Result after initial `strash` (structural hashing)
- **step1-19:** Intermediate optimization steps
- **step20:** Final optimized circuit (before mapping)

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
- **Steps per Recipe:** ~21 (step0 through step20)
- **Total AIG Files:** ~913,500 (29 × 1,500 × 21)
- **Original Format:** BENCH files (converted to AIG using ABC)
- **Conversion Command:** `read_bench {file}.bench; strash; write {file}.aig`


