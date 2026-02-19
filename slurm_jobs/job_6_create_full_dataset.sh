#!/bin/bash
#SBATCH --job-name=create_full_dataset
#SBATCH --time=06:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=genoa
#SBATCH --output=logs/create_full_dataset_%j.out

# Job 6: Create Full Dataset
# Combines Random AIG and OpenABC-D datasets into unified FULL_DATASET structure
# Collects and organizes existing metadata into canonical CSV format

set -e  # Exit on error

echo "=========================================="
echo "JOB 6: Creating Full Dataset"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "Start time: $(date)"
echo ""

# Load required modules for Snellius
module purge
module load 2025
module load foss/2025a
module load Python/3.13.1-GCCcore-14.2.0
module load SciPy-bundle/2025.06-gfbf-2025a

echo "Loaded modules:"
echo "✓ Modules loaded: 2025, foss/2025a, Python/3.13.1, SciPy-bundle/2025.06"
echo "  - Provides: pandas, numpy, tqdm and other scientific Python packages"
echo ""

# Define paths
BASE_DIR="$HOME/data-gen-rand-abcd"
DATASET_TOOLS_DIR="${BASE_DIR}/dataset_tools"
RANDOM_DATASET="${BASE_DIR}/OPENABC_DATASET"
OPENABC_DATASET="/scratch-shared/igardner1/openabc_full/OPENABC_DATASET"  # Full OpenABC-D dataset
OUTPUT_DATASET="/scratch-shared/$USER/FULL_DATASET"
BACKUP_DIR="/scratch-shared/$USER/dataset_backups"
HOME_ARCHIVE_DIR="${BASE_DIR}"
CLEANUP_ORIGINALS=false

RANDOM_BACKUP_FILE=""
RANDOM_BACKUP_SIZE=""
OPENABC_BACKUP_FILE=""
OPENABC_BACKUP_SIZE=""

echo "Base directory: $BASE_DIR"
echo "Random dataset: $RANDOM_DATASET"
echo "OpenABC-D dataset: $OPENABC_DATASET"
echo "Output dataset: $OUTPUT_DATASET"
echo "Backup directory: $BACKUP_DIR"
echo "Home archive directory: $HOME_ARCHIVE_DIR"
echo "Cleanup originals after success: $CLEANUP_ORIGINALS"
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

# Verify required Python packages are available
echo "Verifying Python packages are available..."
python3 -c "import pandas, numpy; print('✓ pandas and numpy available')" || {
    echo "ERROR: Required Python packages not available"
    echo "Make sure SciPy-bundle module is loaded correctly"
    exit 1
}

# Check if tqdm is available (may need separate install)
python3 -c "import tqdm; print('✓ tqdm available')" 2>/dev/null || {
    echo "Installing tqdm (not in SciPy-bundle)..."
    pip install --user tqdm || {
        echo "ERROR: Could not install tqdm"
        exit 1
    }
    echo "✓ tqdm installed"
}
echo ""

# Change to dataset tools directory
cd "$DATASET_TOOLS_DIR"

# Phase: Build full dataset in one pass
echo "=========================================="
echo "PHASE: Building Full Dataset"
echo "=========================================="
echo "Random dataset size before processing:"
du -sh "$RANDOM_DATASET"
echo ""

if [ "$OPENABC_AVAILABLE" = true ]; then
    echo "OpenABC-D dataset size before processing:"
    du -sh "$OPENABC_DATASET"
    echo ""
fi

echo "Creating FULL_DATASET from available sources..."
echo "This will:"
echo "  1. Reorganize Random and OpenABC-D AIG files into unified structure"
echo "  2. Collect metadata into canonical CSV files"
echo "  3. Write manifest and dataset summary"
echo ""

# Build command for combined dataset
CMD_ARGS="--output /scratch-shared/$USER/FULL_DATASET --workers 4"
if [ "$OPENABC_AVAILABLE" = true ]; then
    # Combined run: random + OpenABC-D
    python3 create_full_dataset.py \
        --random-path "$RANDOM_DATASET" \
        --openabc-path "$OPENABC_DATASET" \
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
    echo "DATASET VALIDATION"
    echo "=========================================="
    
    # Validate that the full dataset was created successfully
    echo "Validating FULL_DATASET creation..."
    
    # Check for expected directory structure
    validation_failed=false
    
    if [ ! -d "$OUTPUT_DATASET/base_aigs" ]; then
        echo "✗ Missing base_aigs directory"
        validation_failed=true
    else
        echo "✓ base_aigs directory exists"
    fi
    
    if [ ! -d "$OUTPUT_DATASET/metadata" ]; then
        echo "✗ Missing metadata directory"
        validation_failed=true
    else
        echo "✓ metadata directory exists"
    fi
    
    # Check for some actual content
    aig_count=$(find "$OUTPUT_DATASET/base_aigs" -name "*.aig" 2>/dev/null | wc -l)
    if [ "$aig_count" -eq 0 ]; then
        echo "✗ No AIG files found in base_aigs directory"
        validation_failed=true
    else
        echo "✓ Found $aig_count AIG files"
    fi
    
    csv_count=$(find "$OUTPUT_DATASET/metadata" -name "*.csv" 2>/dev/null | wc -l)
    if [ "$csv_count" -eq 0 ]; then
        echo "✗ No CSV metadata files found"
        validation_failed=true
    else
        echo "✓ Found $csv_count CSV metadata files"
    fi
    
    # If validation failed, abort cleanup
    if [ "$validation_failed" = true ]; then
        echo ""
        echo "⚠️  DATASET VALIDATION FAILED"
        echo "✗ FULL_DATASET creation appears incomplete or failed"
        echo "✓ ORIGINALS PRESERVED - No cleanup will be performed"
        echo ""
        echo "Please check the dataset creation process and try again."
        echo "Original datasets remain untouched:"
        echo "  - Random dataset: $RANDOM_DATASET"
        if [ "$OPENABC_AVAILABLE" = true ]; then
            echo "  - OpenABC-D dataset: $OPENABC_DATASET"
        fi
        exit 1
    fi
    
    echo ""
    echo "✓ Dataset validation passed - FULL_DATASET creation successful"

    if [ "$CLEANUP_ORIGINALS" = true ]; then
        echo "✓ Safe to proceed with cleanup of originals"
        echo ""
        echo "=========================================="
        echo "CLEANUP PHASE: Archiving Original Datasets"
        echo "=========================================="
        echo "The original datasets are very large. Creating compressed archives and removing originals to save space..."
        echo ""

        # Archive and remove Random dataset
        echo "Archiving Random AIG dataset..."
        RANDOM_BACKUP_FILE="$BACKUP_DIR/RANDOM_DATASET_backup_$(date +%Y%m%d_%H%M%S).tar.gz"
        echo "Creating: $RANDOM_BACKUP_FILE"

        tar -czf "$RANDOM_BACKUP_FILE" -C "$(dirname "$RANDOM_DATASET")" "$(basename "$RANDOM_DATASET")" || {
            echo "ERROR: Failed to create Random dataset backup"
            exit 1
        }

        # Verify Random backup
        echo "Verifying Random dataset backup..."
        tar -tzf "$RANDOM_BACKUP_FILE" > /dev/null && echo "✓ Random backup integrity verified" || {
            echo "ERROR: Random backup integrity check failed"
            exit 1
        }

        RANDOM_BACKUP_SIZE=$(du -sh "$RANDOM_BACKUP_FILE" | cut -f1)
        echo "✓ Random dataset archived: $RANDOM_BACKUP_SIZE"
        echo "Backup location: $RANDOM_BACKUP_FILE"
        echo ""

        # Archive and remove OpenABC-D dataset (if available and not already backed up)
        if [ "$OPENABC_AVAILABLE" = true ]; then
            echo "Archiving OpenABC-D dataset..."
            OPENABC_BACKUP_FILE="$BACKUP_DIR/OPENABC_DATASET_backup_$(date +%Y%m%d_%H%M%S).tar.gz"
            echo "Creating: $OPENABC_BACKUP_FILE"

            tar -czf "$OPENABC_BACKUP_FILE" -C "$(dirname "$OPENABC_DATASET")" "$(basename "$OPENABC_DATASET")" || {
                echo "ERROR: Failed to create OpenABC-D dataset backup"
                exit 1
            }

            # Verify OpenABC backup
            echo "Verifying OpenABC-D dataset backup..."
            tar -tzf "$OPENABC_BACKUP_FILE" > /dev/null && echo "✓ OpenABC-D backup integrity verified" || {
                echo "ERROR: OpenABC-D backup integrity check failed"
                exit 1
            }

            OPENABC_BACKUP_SIZE=$(du -sh "$OPENABC_BACKUP_FILE" | cut -f1)
            echo "✓ OpenABC-D dataset archived: $OPENABC_BACKUP_SIZE"
            echo "Backup location: $OPENABC_BACKUP_FILE"
            echo ""
        fi

        echo "=========================================="
        echo "REMOVING ORIGINAL DATASETS"
        echo "=========================================="
        echo "⚠️  FINAL SAFETY CHECK"
        echo "   - FULL_DATASET validation: PASSED ✓"
        echo "   - Backups created and verified: ✓"
        echo "   - About to delete original datasets!"
        echo ""
        echo "Original datasets to be removed:"
        echo "  - Random dataset: $RANDOM_DATASET"
        if [ "$OPENABC_AVAILABLE" = true ]; then
            echo "  - OpenABC-D dataset: $OPENABC_DATASET"
        fi
        echo ""
        echo "Proceeding with cleanup in 15 seconds..."
        echo "Press Ctrl+C to abort if you want to keep originals"
        echo ""

        sleep 15

        # Remove Random dataset
        echo "Removing Random dataset: $RANDOM_DATASET"
        rm -rf "$RANDOM_DATASET" && echo "✓ Random dataset removed" || echo "⚠️  Warning: Failed to remove Random dataset"

        # Remove OpenABC-D dataset (if we have access)
        if [ "$OPENABC_AVAILABLE" = true ]; then
            echo "Removing OpenABC-D dataset: $OPENABC_DATASET"
            rm -rf "$OPENABC_DATASET" && echo "✓ OpenABC-D dataset removed" || echo "⚠️  Warning: Failed to remove OpenABC-D dataset"
        fi

        echo ""
        echo "✓ Cleanup completed - original datasets archived and removed"
    else
        echo "✓ Non-destructive mode: cleanup disabled"
        echo "  Original datasets are preserved:"
        echo "  - Random dataset: $RANDOM_DATASET"
        if [ "$OPENABC_AVAILABLE" = true ]; then
            echo "  - OpenABC-D dataset: $OPENABC_DATASET"
        fi
    fi
    
    echo ""
    echo "=========================================="
    echo "FINAL DATASET SUMMARY"
    echo "=========================================="
    
    # Calculate total dataset size
    echo "Calculating final dataset size..."
    TOTAL_SIZE=$(du -sh "$OUTPUT_DATASET" | cut -f1)
    TOTAL_SIZE_BYTES=$(du -sb "$OUTPUT_DATASET" | cut -f1)
    TOTAL_SIZE_GB=$(echo "scale=2; $TOTAL_SIZE_BYTES / 1024 / 1024 / 1024" | bc)
    
    echo ""
    echo "🗂️  FULL DATASET SIZE ANALYSIS"
    echo "================================"
    echo "Total size: $TOTAL_SIZE (${TOTAL_SIZE_BYTES} bytes / ${TOTAL_SIZE_GB} GB)"
    echo "Location: $OUTPUT_DATASET"
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
    echo "📁 Directory sizes breakdown:"
    du -sh "$OUTPUT_DATASET"/*
    
    echo ""
    echo "💾 STORAGE LOCATION ANALYSIS"
    echo "============================"
    echo "Current location: $OUTPUT_DATASET (scratch-shared)"
    echo ""
    echo "Storage options analysis:"
    
    # Check available space in different locations
    echo "  📊 Scratch space remaining:"
    df -h "/scratch-shared/$USER" | tail -1 | awk '{print "      Available: " $4 " (" $5 " used)"}'
    
    echo ""
    echo "  📊 Home directory space:"
    df -h "$HOME" | tail -1 | awk '{print "      Available: " $4 " (" $5 " used)"}'
    echo "      Location: $HOME"
    
    echo ""
    echo "  🔄 Dataset portability:"
    if (( $(echo "$TOTAL_SIZE_GB < 50" | bc -l) )); then
        echo "      ✅ SMALL dataset (${TOTAL_SIZE_GB} GB) - Easy to move to home directory"
        echo "      💡 Recommended: Move to home for permanent storage"
        echo "      📝 Command: mv $OUTPUT_DATASET $HOME/"
    elif (( $(echo "$TOTAL_SIZE_GB < 200" | bc -l) )); then
        echo "      ⚠️  MEDIUM dataset (${TOTAL_SIZE_GB} GB) - Consider if home has space"
        echo "      💡 Check home directory space before moving"
        echo "      📝 Command: mv $OUTPUT_DATASET $HOME/ (if space permits)"
    else
        echo "      ⚠️  LARGE dataset (${TOTAL_SIZE_GB} GB) - May need to stay in scratch"
        echo "      💡 Consider keeping in scratch-shared for performance"
        echo "      📝 Archive older datasets if scratch space is needed"
    fi

    echo ""
    echo "=========================================="
    echo "SECOND COPY: HOME ARCHIVE (IF SPACE ALLOWS)"
    echo "=========================================="
    mkdir -p "$HOME_ARCHIVE_DIR"

    HOME_ARCHIVE_FILE="$HOME_ARCHIVE_DIR/FULL_DATASET_$(date +%Y%m%d_%H%M%S).tar.gz"
    HOME_AVAILABLE_BYTES=$(df -B1 "$HOME" | awk 'NR==2 {print $4}')
    REQUIRED_BYTES=$((TOTAL_SIZE_BYTES + TOTAL_SIZE_BYTES / 20))  # 5% buffer
    HOME_AVAILABLE_GB=$(echo "scale=2; $HOME_AVAILABLE_BYTES / 1024 / 1024 / 1024" | bc)
    REQUIRED_GB=$(echo "scale=2; $REQUIRED_BYTES / 1024 / 1024 / 1024" | bc)

    echo "Home free space: ${HOME_AVAILABLE_GB} GB"
    echo "Required for archive (size + 5% buffer): ${REQUIRED_GB} GB"

    if [ "$HOME_AVAILABLE_BYTES" -ge "$REQUIRED_BYTES" ]; then
        echo "✓ Sufficient home space detected. Creating second copy archive..."
        echo "Archive target: $HOME_ARCHIVE_FILE"

        tar -czf "$HOME_ARCHIVE_FILE" -C "$(dirname "$OUTPUT_DATASET")" "$(basename "$OUTPUT_DATASET")" || {
            echo "⚠️  Warning: Failed to create home archive copy"
            HOME_ARCHIVE_FILE=""
        }

        if [ -n "$HOME_ARCHIVE_FILE" ] && [ -f "$HOME_ARCHIVE_FILE" ]; then
            tar -tzf "$HOME_ARCHIVE_FILE" > /dev/null && echo "✓ Home archive integrity verified" || {
                echo "⚠️  Warning: Home archive integrity check failed"
                rm -f "$HOME_ARCHIVE_FILE"
                HOME_ARCHIVE_FILE=""
            }
        fi

        if [ -n "$HOME_ARCHIVE_FILE" ] && [ -f "$HOME_ARCHIVE_FILE" ]; then
            HOME_ARCHIVE_SIZE=$(du -sh "$HOME_ARCHIVE_FILE" | cut -f1)
            echo "✓ Second copy created in home: $HOME_ARCHIVE_FILE ($HOME_ARCHIVE_SIZE)"
        fi
    else
        echo "⚠️  Skipping home archive copy: insufficient space"
        echo "   Needed: ${REQUIRED_GB} GB, Available: ${HOME_AVAILABLE_GB} GB"
        HOME_ARCHIVE_FILE=""
    fi
    
    echo ""
    echo "=========================================="
    echo "SPACE USAGE SUMMARY"
    echo "=========================================="
    
    # Show backup information
    echo "Dataset Backups Created:"
    if [ "$CLEANUP_ORIGINALS" = true ]; then
        echo "  - Random dataset: $RANDOM_BACKUP_SIZE ($RANDOM_BACKUP_FILE)"
        if [ "$OPENABC_AVAILABLE" = true ]; then
            echo "  - OpenABC-D dataset: $OPENABC_BACKUP_SIZE ($OPENABC_BACKUP_FILE)"
        fi
    else
        echo "  - Not created (cleanup disabled)"
    fi
    echo ""
    
    echo "Final dataset size: $TOTAL_SIZE (${TOTAL_SIZE_GB} GB)"
    echo "Dataset location: $OUTPUT_DATASET (scratch-shared)"
    echo ""
    
    # Calculate space savings
    echo "Space Savings:"
    if [ "$CLEANUP_ORIGINALS" = true ]; then
        echo "  - Original datasets: REMOVED (archived as compressed backups)"
        echo "  - Storage efficiency: Significant space saved by removing uncompressed originals"
    else
        echo "  - Original datasets: PRESERVED (non-destructive mode)"
        echo "  - No cleanup-related space reclaimed"
    fi
    echo ""
    
    # Check available scratch space
    echo "Scratch space usage after cleanup:"
    df -h "/scratch-shared/$USER"
    
    echo ""
    echo "✓ Full dataset created successfully with space optimization!"
    echo ""
    echo "Files created:"
    echo "  - FULL_DATASET: $OUTPUT_DATASET ($TOTAL_SIZE / ${TOTAL_SIZE_GB} GB)"
    if [ "$CLEANUP_ORIGINALS" = true ]; then
        echo "  - Random backup: $RANDOM_BACKUP_FILE ($RANDOM_BACKUP_SIZE)"
        if [ "$OPENABC_AVAILABLE" = true ]; then
            echo "  - OpenABC-D backup: $OPENABC_BACKUP_FILE ($OPENABC_BACKUP_SIZE)"
        fi
    fi
    if [ -n "$HOME_ARCHIVE_FILE" ] && [ -f "$HOME_ARCHIVE_FILE" ]; then
        echo "  - Home archive copy: $HOME_ARCHIVE_FILE ($HOME_ARCHIVE_SIZE)"
    else
        echo "  - Home archive copy: Not created (insufficient space or archive failure)"
    fi
    echo ""
    echo "Files removed:"
    if [ "$CLEANUP_ORIGINALS" = true ]; then
        echo "  - Original Random dataset: $RANDOM_DATASET (archived)"
        if [ "$OPENABC_AVAILABLE" = true ]; then
            echo "  - Original OpenABC-D dataset: $OPENABC_DATASET (archived)"
        fi
    else
        echo "  - None (cleanup disabled)"
    fi
    echo ""
    echo "🚀 Next steps:"
    echo "  1. Verify dataset integrity"
    echo "  2. Consider moving to home directory if size permits (see analysis above)"
    echo "  3. Run algorithm optimization pipelines (Orchestrate, Deepsyn, Syn4, C2RS)"
    echo "  4. Begin ML experiments"
    echo "  5. Backups can be extracted if originals are needed: tar -xzf backup_file.tar.gz"
    
else
    echo ""
    echo "✗ ERROR: Dataset creation failed - output directory not found"
    echo "✓ ORIGINALS PRESERVED - No cleanup performed"
    echo ""
    echo "Original datasets remain untouched:"
    echo "  - Random dataset: $RANDOM_DATASET"
    if [ "$OPENABC_AVAILABLE" = true ]; then
        echo "  - OpenABC-D dataset: $OPENABC_DATASET"
    fi
    echo ""
    echo "Please investigate the dataset creation failure and try again."
    exit 1
fi

echo ""
echo "End time: $(date)"
echo "Job 6 completed successfully!"