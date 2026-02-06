# Random AIG Dataset Documentation

## Overview
This dataset contains synthesized AIG (And-Inverter Graph) files generated using ABC (Berkeley Logic Synthesis and Verification Tool) for 8 different circuit designs of varying sizes.

## Dataset Structure

### Random Designs
The dataset includes 8 random designs, named by their size:
- `128` 
- `256`  
- `512` 
- `1024` 
- `2048` 
- `4096` 
- `8192` 
- `16384` 

### Directory Structure
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

## File Naming Convention

### Original AIG Files
- **Format:** `{design}_orig.aig`
- **Examples:** `128_orig.aig`, `256_orig.aig`, `16384_orig.aig`
- **Location:** `OPENABC_DATASET/bench/{design}/`

### Synthesized AIG Files
- **Format:** `{design}_syn{recipe}_step{step}.aig`
- **Examples:** 
  - `128_syn0_step0.aig` - Design 128, recipe 0, step 0
  - `256_syn42_step15.aig` - Design 256, recipe 42, step 15
  - `16384_syn1499_step20.aig` - Design 16384, recipe 1499, step 20
- **Location:** Inside `syn{recipe}.zip` files

### Compressed Archives
- **Format:** `syn{recipe}.zip`
- **Range:** `syn0.zip` to `syn1499.zip`
- **Location:** `OPENABC_DATASET/bench/{design}/`

## Dataset Size

### Per Design
- **Original AIG files:** 1
- **Synthesis recipes:** 1,500
- **Steps per recipe:** ~21 (step0 through step20)
- **Total AIGs per design:** ~31,500 (1 original + 1,500 × 21 synthesized)
- **Zip archives:** 1,500
- **Log files:** 1,500

### Total Dataset (All 8 Designs)
- **Original AIG files:** 8
- **Total synthesis recipes:** 12,000 (8 × 1,500)
- **Total synthesized AIGs:** ~252,000 (8 × 31,500)
- **Total zip archives:** 12,000
- **Total log files:** 12,000

## Numbering Scheme

### Recipe Numbers
- **Range:** 0 to 1499
- **Total:** 1,500 recipes per design
- All designs use the same recipe numbers for consistency

### Step Numbers
- **Range:** 0 to 20
- **Total:** ~21 steps per recipe
- **step0:** Result after initial `strash` (structural hashing)
- **step1-19:** Intermediate optimization steps
- **step20:** Final optimized circuit (before mapping)