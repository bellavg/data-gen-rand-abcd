#!/bin/bash
#SBATCH --job-name=create_full_dataset
#SBATCH --time=06:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=genoa
#SBATCH --output=logs/create_full_dataset_%j.out

# Step 5: Create Full Dataset
# Combines Random AIG and OpenABC-D datasets into unified FULL_DATASET structure
# Collects and organizes existing metadata into canonical CSV format

set -e  # Exit on error

echo "=========================================="
echo "STEP 5: Creating Full Dataset"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "Start time: $(date)"
echo ""

# Load required modules for Snellius
module purge
module load 2025
module load Python/3.13.1-GCCcore-14.2.0

echo "Loaded modules:"
module list
echo ""

# Define paths
BASE_DIR="$HOME/data-gen-rand-abcd"
DATASET_TOOLS_DIR="${BASE_DIR}/dataset_tools"
RANDOM_DATASET="${BASE_DIR}/OPENABC_DATASET"
OPENABC_DATASET="/scratch-shared/igardner1/openabc_full/OPENABC_DATASET"  # Full OpenABC-D dataset
OUTPUT_DATASET="/scratch-shared/$USER/FULL_DATASET"
BACKUP_DIR="/scratch-shared/$USER/dataset_backups"

echo "Base directory: $BASE_DIR"
echo "Random dataset: $RANDOM_DATASET"
echo "OpenABC-D dataset: $OPENABC_DATASET"
echo "Output dataset: $OUTPUT_DATASET"
echo "Backup directory: $BACKUP_DIR"
echo "Using scratch-shared for output to handle large dataset size"
echo ""

# Check if required directories exist
if [ ! -d "$RANDOM_DATASET" ]; then
    echo "ERROR: Random dataset directory not found: $RANDOM_DATASET"
    echo "Please ensure job 4 synthesis steps have completed successfully."
    exit 1
fi

if [ ! -d "$DATASET_TOOLS_DIR" ]; then
    echo "ERROR: Dataset tools directory not found: $DATASET_TOOLS_DIR"
    exit 1
fi

if [ ! -d "$OPENABC_DATASET" ]; then
    echo "WARNING: OpenABC-D dataset directory not found: $OPENABC_DATASET"
    echo "Will proceed with Random dataset only."
    OPENABC_AVAILABLE=false
else
    OPENABC_AVAILABLE=true
fi

echo "Directory validation passed."
echo ""

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Install required Python packages
echo "Installing required Python packages..."
pip install --user pandas tqdm
echo ""

# Change to dataset tools directory
cd "$DATASET_TOOLS_DIR"

# Phase 1: Process OpenABC-D dataset first (if available)
if [ "$OPENABC_AVAILABLE" = true ]; then
    echo "=========================================="
    echo "PHASE 1: Processing OpenABC-D Dataset"
    echo "=========================================="
    echo "OpenABC-D dataset size before processing:"
    du -sh "$OPENABC_DATASET"
    echo ""
    
    echo "Creating initial dataset structure with OpenABC-D only..."
    python3 run_dataset_creation.py \
        --output-dir "/scratch-shared/$USER/FULL_DATASET" \
        --workers 4 || {
        echo "ERROR: Failed to process OpenABC-D dataset"
        exit 1
    }
    
    echo ""
    echo "=========================================="
    echo "PHASE 2: Backing up OpenABC-D Dataset"
    echo "=========================================="
    echo "Creating compressed backup of OpenABC-D dataset..."
    echo "This may take a while due to large size..."
    
    # Create compressed backup
    BACKUP_FILE="$BACKUP_DIR/OPENABC_DATASET_backup_$(date +%Y%m%d_%H%M%S).tar.gz"
    echo "Backup location: $BACKUP_FILE"
    
    tar -czf "$BACKUP_FILE" -C "$(dirname "$OPENABC_DATASET")" "$(basename "$OPENABC_DATASET")" || {
        echo "ERROR: Failed to create backup of OpenABC-D dataset"
        exit 1
    }
    
    # Check backup size
    BACKUP_SIZE=$(du -sh "$BACKUP_FILE" | cut -f1)
    BACKUP_SIZE_BYTES=$(du -sb "$BACKUP_FILE" | cut -f1)
    echo ""
    echo "✓ OpenABC-D backup created successfully!"
    echo "Backup size: $BACKUP_SIZE (${BACKUP_SIZE_BYTES} bytes)"
    echo "Backup location: $BACKUP_FILE"
    echo ""
    
    # Verify backup integrity
    echo "Verifying backup integrity..."
    tar -tzf "$BACKUP_FILE" > /dev/null && echo "✓ Backup integrity verified" || {
        echo "ERROR: Backup integrity check failed"
        exit 1
    }
    echo ""
fi

# Phase 3: Process Random dataset
echo "=========================================="
echo "PHASE 3: Adding Random AIG Dataset"
echo "=========================================="
echo "Random dataset size before processing:"
du -sh "$RANDOM_DATASET"
echo ""

echo "Adding Random AIG dataset to existing FULL_DATASET..."
echo "This will:"
echo "  1. Add Random AIG files to existing unified structure"
echo "  2. Collect metadata from Random dataset"  
echo "  3. Update dataset manifest and summary"
echo ""

# Build command for combined dataset
CMD_ARGS="--output-dir /scratch-shared/$USER/FULL_DATASET --workers 4"
if [ "$OPENABC_AVAILABLE" = true ]; then
    # Add OpenABC dataset path if available
    python3 create_full_dataset.py \
        --random-path "$RANDOM_DATASET" \
        --openabc-path "$(dirname "$OPENABC_DATASET")" \
        $CMD_ARGS
else
    # Random dataset only
    python3 create_full_dataset.py \
        --random-path "$RANDOM_DATASET" \
        $CMD_ARGS
fi

# Verify the output
if [ -d "$OUTPUT_DATASET" ]; then
    echo ""
    echo "=========================================="
    echo "FINAL DATASET SUMMARY"
    echo "=========================================="
    
    # Calculate total dataset size
    echo "Calculating final dataset size..."
    TOTAL_SIZE=$(du -sh "$OUTPUT_DATASET" | cut -f1)
    TOTAL_SIZE_BYTES=$(du -sb "$OUTPUT_DATASET" | cut -f1)
    
    echo "Total FULL_DATASET size: $TOTAL_SIZE (${TOTAL_SIZE_BYTES} bytes)"
    echo ""
    
    # Count files in different directories
    if [ -d "$OUTPUT_DATASET/base_aigs" ]; then
        AIG_COUNT=$(find "$OUTPUT_DATASET/base_aigs" -name "*.aig" | wc -l)
        AIG_SIZE=$(du -sh "$OUTPUT_DATASET/base_aigs" | cut -f1)
        echo "AIG files created: $AIG_COUNT (Size: $AIG_SIZE)"
    fi
    
    if [ -d "$OUTPUT_DATASET/metadata/stats" ]; then
        CSV_COUNT=$(find "$OUTPUT_DATASET/metadata/stats" -name "*.csv" | wc -l)
        METADATA_SIZE=$(du -sh "$OUTPUT_DATASET/metadata" | cut -f1)
        echo "Metadata CSV files: $CSV_COUNT (Size: $METADATA_SIZE)"
    fi
    
    if [ -d "$OUTPUT_DATASET/synScripts" ]; then
        SCRIPT_COUNT=$(find "$OUTPUT_DATASET/synScripts" -name "*.zip" | wc -l)
        SCRIPTS_SIZE=$(du -sh "$OUTPUT_DATASET/synScripts" | cut -f1)
        echo "Synthesis script archives: $SCRIPT_COUNT (Size: $SCRIPTS_SIZE)"
    fi
    
    echo ""
    echo "Dataset directory structure:"
    ls -la "$OUTPUT_DATASET"
    
    echo ""
    echo "Directory sizes breakdown:"
    du -sh "$OUTPUT_DATASET"/*
    
    echo ""
    echo "=========================================="
    echo "SPACE USAGE SUMMARY"
    echo "=========================================="
    
    # Show backup information if created
    if [ "$OPENABC_AVAILABLE" = true ] && [ -f "$BACKUP_FILE" ]; then
        echo "OpenABC-D backup: $BACKUP_SIZE"
        echo "Backup location: $BACKUP_FILE"
        echo ""
    fi
    
    echo "Final dataset size: $TOTAL_SIZE"
    echo "Dataset location: $OUTPUT_DATASET"
    echo ""
    
    # Check available scratch space
    echo "Scratch space usage:"
    df -h "/scratch-shared/$USER"
    
    echo ""
    echo "✓ Full dataset created successfully!"
    echo ""
    echo "Files created:"
    echo "  - FULL_DATASET: $OUTPUT_DATASET ($TOTAL_SIZE)"
    if [ "$OPENABC_AVAILABLE" = true ] && [ -f "$BACKUP_FILE" ]; then
        echo "  - OpenABC-D backup: $BACKUP_FILE ($BACKUP_SIZE)"
    fi
    
    echo ""
    echo "Next steps:"
    echo "  1. Verify dataset integrity"
    echo "  2. Run algorithm optimization pipelines (Orchestrate, Deepsyn, Syn4, C2RS)"
    echo "  3. Begin ML experiments"
    echo "  4. Consider removing original datasets if space is needed"
    
else
    echo ""
    echo "ERROR: Dataset creation failed - output directory not found"
    exit 1
fi

echo ""
echo "End time: $(date)"
echo "Job 5 completed successfully!"