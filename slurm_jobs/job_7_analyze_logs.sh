#!/bin/bash
#SBATCH --job-name=log_analysis
#SBATCH --time=00:20:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=genoa
#SBATCH --output=logs/log_analysis_%j.out

# Job 7: Quick ABC Log Error Analysis
# Samples a few ABC synthesis logs and prints common errors directly to SLURM output
# No files created - all analysis printed to console for immediate review

set -e  # Exit on error

echo "=========================================="
echo "QUICK ABC LOG ERROR ANALYSIS"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "Start time: $(date)"
echo ""

# Define paths
BASE_DIR="$HOME/data-gen-rand-abcd"
SYNTHESIS_BASE="${BASE_DIR}/OPENABC_DATASET/bench"

echo "Sampling ABC synthesis logs from: $SYNTHESIS_BASE"
echo ""

# Check if synthesis base directory exists
if [ ! -d "$SYNTHESIS_BASE" ]; then
    echo "❌ ERROR: Synthesis base directory not found: $SYNTHESIS_BASE"
    exit 1
fi

# Sample and analyze a few log files from different designs
echo "🔍 SAMPLING LOG FILES FOR ERROR PATTERNS:"
echo "=========================================="

sample_count=0
max_samples=8

for design in 128 256 512 1024 2048 4096 8192 16384; do
    log_dir="${SYNTHESIS_BASE}/${design}/log_${design}"
    
    if [ ! -d "$log_dir" ]; then
        echo "⚠️  Design $design: No log directory found"
        continue
    fi
    
    # Get first few log files from this design
    sample_files=$(find "$log_dir" -name "*.log" | head -2)
    
    if [ -z "$sample_files" ]; then
        echo "⚠️  Design $design: No .log files found in $log_dir"
        continue
    fi
    
    echo ""
    echo "📁 DESIGN $design samples:"
    echo "─────────────────────────────"
    
    while IFS= read -r log_file && [ $sample_count -lt $max_samples ]; do
        basename=$(basename "$log_file")
        echo "File: $basename"
        
        # Check for ABC command errors
        abc_errors=$(grep "\*\* cmd error:" "$log_file" 2>/dev/null | wc -l)
        if [ "$abc_errors" -gt 0 ]; then
            echo "  🔴 ABC Command Errors ($abc_errors):"
            grep "\*\* cmd error:" "$log_file" 2>/dev/null | head -3 | sed 's/^/    /'
        fi
        
        # Check for warnings
        warnings=$(grep -i "warning" "$log_file" 2>/dev/null | wc -l)
        if [ "$warnings" -gt 0 ]; then
            echo "  🟡 Warnings ($warnings):"
            grep -i "warning" "$log_file" 2>/dev/null | head -3 | sed 's/^/    /'
        fi
        
        # Check for general errors
        general_errors=$(grep -i "error:" "$log_file" 2>/dev/null | wc -l)
        if [ "$general_errors" -gt 0 ]; then
            echo "  🔴 General Errors ($general_errors):"
            grep -i "error:" "$log_file" 2>/dev/null | head -3 | sed 's/^/    /'
        fi
        
        # Show last few lines for context
        echo "  📄 Last 3 lines:"
        tail -3 "$log_file" 2>/dev/null | sed 's/^/    /'
        echo ""
        
        sample_count=$((sample_count + 1))
    done <<< "$sample_files"
    
    if [ $sample_count -ge $max_samples ]; then
        break
    fi
done

echo ""
echo "🔍 MOST COMMON ERROR PATTERNS ACROSS ALL DESIGNS:"
echo "================================================"

for design in 128 256 512 1024 2048; do
    log_dir="${SYNTHESIS_BASE}/${design}/log_${design}"
    if [ -d "$log_dir" ]; then
        echo ""
        echo "Design $design top errors:"
        find "$log_dir" -name "*.log" | head -10 | xargs grep -h -i "error\|warning" 2>/dev/null | \
            sort | uniq -c | sort -nr | head -5 | while read count pattern; do
            echo "  [$count times] $pattern"
        done
    fi
done

echo ""
echo "=========================================="
echo "ANALYSIS COMPLETE"
echo "=========================================="
echo "End time: $(date)"
echo ""
echo "📊 SUMMARY:"
echo "- Analyzed $sample_count sample log files"
echo "- Check the patterns above to identify the main issues"
echo "- Most frequent errors are likely the root cause"

