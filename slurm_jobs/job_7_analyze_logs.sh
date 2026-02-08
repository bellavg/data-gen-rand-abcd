#!/bin/bash
#SBATCH --job-name=log_analysis
#SBATCH --time=00:20:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=genoa
#SBATCH --output=logs/log_analysis_%j.out

# Job 7: Complete ABC Log Verification
# Checks ALL ABC synthesis logs to verify only the benign FA_X1 warning occurs
# Reports any unexpected errors or warnings

set -e  # Exit on error

echo "=========================================="
echo "COMPLETE ABC LOG VERIFICATION"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "Start time: $(date)"
echo ""

# Define paths
BASE_DIR="$HOME/data-gen-rand-abcd"
SYNTHESIS_BASE="${BASE_DIR}/OPENABC_DATASET/bench"

echo "Checking ALL ABC synthesis logs in: $SYNTHESIS_BASE"
echo "Looking for any errors/warnings other than the expected FA_X1 multi-output warning..."
echo ""

# Check if synthesis base directory exists
if [ ! -d "$SYNTHESIS_BASE" ]; then
    echo "❌ ERROR: Synthesis base directory not found: $SYNTHESIS_BASE"
    exit 1
fi

# Counters for comprehensive analysis
total_logs=0
logs_with_fa_warning=0
logs_with_other_issues=0
problematic_logs=()

echo "🔍 COMPREHENSIVE LOG VERIFICATION:"
echo "================================="

for design in 128 256 512 1024 2048 4096 8192 16384; do
    log_dir="${SYNTHESIS_BASE}/${design}/log_${design}"
    
    if [ ! -d "$log_dir" ]; then
        echo "⚠️  Design $design: No log directory found"
        continue
    fi
    
    echo ""
    echo "📁 Checking Design $design..."
    
    # Debug: show what we're looking for
    echo "   Log directory: $log_dir"
    if [ -d "$log_dir" ]; then
        log_file_count=$(find "$log_dir" -name "*.log" | wc -l | tr -d ' ')
        echo "   Found $log_file_count .log files"
        if [ "$log_file_count" -eq 0 ]; then
            echo "   📄 Directory contents:"
            ls -la "$log_dir/" | head -10 | sed 's/^/     /'
        fi
    fi
    
    design_logs=0
    design_fa_warnings=0
    design_other_issues=0
    
    # Process ALL .log files in this design
    while IFS= read -r log_file; do
        if [ -f "$log_file" ]; then
            total_logs=$((total_logs + 1))
            design_logs=$((design_logs + 1))
            basename=$(basename "$log_file")
            
            # Check for the expected FA_X1 warning
            fa_warnings=$(grep -c "Warning: Detected .* multi-output cells.*FA_X1" "$log_file" 2>/dev/null | head -1 || echo "0")
            
            # Check for ANY other warnings or errors
            other_warnings=$(grep -c -i "warning" "$log_file" 2>/dev/null | head -1 || echo "0")
            
            # Clean values and ensure they're integers
            fa_warnings=${fa_warnings//[^0-9]/}; fa_warnings=${fa_warnings:-0}
            other_warnings=${other_warnings//[^0-9]/}; other_warnings=${other_warnings:-0}
            
            # Subtract the FA_X1 warnings from total warnings
            other_warnings=$((other_warnings - fa_warnings))
            
            abc_errors=$(grep -c "\*\* cmd error:" "$log_file" 2>/dev/null | head -1 || echo "0")
            general_errors=$(grep -c -i "error:" "$log_file" 2>/dev/null | head -1 || echo "0")
            system_errors=$(grep -c -i "ERROR:" "$log_file" 2>/dev/null | head -1 || echo "0")
            
            # Clean error counts
            abc_errors=${abc_errors//[^0-9]/}; abc_errors=${abc_errors:-0}
            general_errors=${general_errors//[^0-9]/}; general_errors=${general_errors:-0}
            system_errors=${system_errors//[^0-9]/}; system_errors=${system_errors:-0}
            
            # Track FA_X1 warnings
            if [ "$fa_warnings" -gt 0 ]; then
                logs_with_fa_warning=$((logs_with_fa_warning + 1))
                design_fa_warnings=$((design_fa_warnings + 1))
            fi
            
            # Check for any problematic issues
            total_other_issues=$((other_warnings + abc_errors + general_errors + system_errors))
            
            if [ "$total_other_issues" -gt 0 ]; then
                logs_with_other_issues=$((logs_with_other_issues + 1))
                design_other_issues=$((design_other_issues + 1))
                problematic_logs+=("$log_file")
                
                echo "  🚨 UNEXPECTED ISSUE in $basename:"
                if [ "$other_warnings" -gt 0 ]; then
                    echo "    • Other warnings ($other_warnings):"
                    grep -i "warning" "$log_file" 2>/dev/null | grep -v "FA_X1" | head -3 | sed 's/^/      /'
                fi
                if [ "$abc_errors" -gt 0 ]; then
                    echo "    • ABC errors ($abc_errors):"
                    grep "\*\* cmd error:" "$log_file" 2>/dev/null | head -3 | sed 's/^/      /'
                fi
                if [ "$general_errors" -gt 0 ]; then
                    echo "    • General errors ($general_errors):"
                    grep -i "error:" "$log_file" 2>/dev/null | head -3 | sed 's/^/      /'
                fi
                if [ "$system_errors" -gt 0 ]; then
                    echo "    • System errors ($system_errors):"
                    grep -i "ERROR:" "$log_file" 2>/dev/null | head -3 | sed 's/^/      /'
                fi
            fi
            
            # Progress indicator for large numbers of files
            if [ $((total_logs % 500)) -eq 0 ]; then
                echo "    Progress: Checked $total_logs files total..."
            fi
        fi
    done < <(find "$log_dir" -name "*.log" | sort)
    
    echo "  📊 Design $design summary:"
    echo "    • Total logs: $design_logs"
    echo "    • With FA_X1 warnings: $design_fa_warnings"
    echo "    • With other issues: $design_other_issues"
done

echo ""
echo "🔍 CSV FILE COMPLETENESS CHECK:"
echo "==============================="
echo ""

# Check CSV files for completeness and synthesis progression
csv_issues_found=0

for design in 128 256 512 1024 2048 4096 8192 16384; do
    csv_file="${SYNTHESIS_BASE}/${design}/metadata/${design}.csv"
    
    if [ ! -f "$csv_file" ]; then
        echo "⚠️  Design $design: CSV file not found at $csv_file"
        csv_issues_found=$((csv_issues_found + 1))
        continue
    fi
    
    echo "📊 Design $design CSV analysis:"
    
    # Count total lines (minus header)
    total_lines=$(tail -n +2 "$csv_file" 2>/dev/null | wc -l | tr -d ' ')
    expected_lines=$((1500 * 21))  # 1500 recipes × 21 steps each
    
    echo "   • Total entries: $total_lines (expected: $expected_lines)"
    
    if [ "$total_lines" -lt $((expected_lines - 100)) ]; then
        echo "   ⚠️  Significantly fewer entries than expected!"
        csv_issues_found=$((csv_issues_found + 1))
    fi
    
    # Sample a few recipes to check for synthesis progression
    echo "   • Checking synthesis progression for sample recipes..."
    
    # Get a few random recipe IDs from the CSV
    sample_recipes=$(tail -n +2 "$csv_file" 2>/dev/null | cut -d',' -f1 | sort -u | head -3)
    
    for recipe_id in $sample_recipes; do
        if [ -n "$recipe_id" ]; then
            echo "     Recipe $recipe_id progression:"
            
            # Get node counts for this recipe across steps
            recipe_data=$(grep "^${recipe_id}," "$csv_file" 2>/dev/null | head -10)
            
            if [ -z "$recipe_data" ]; then
                echo "       ⚠️  No data found for recipe $recipe_id"
                continue
            fi
            
            # Extract step IDs and node counts
            prev_nodes=""
            identical_count=0
            total_steps=0
            
            while IFS=',' read -r rec_id step_id nodes edges pi po depth avg_fanout max_fanout rest; do
                if [ -n "$nodes" ] && [ "$nodes" != "nodes" ]; then  # Skip header
                    total_steps=$((total_steps + 1))
                    
                    if [ "$prev_nodes" = "$nodes" ]; then
                        identical_count=$((identical_count + 1))
                    fi
                    
                    if [ $total_steps -le 5 ]; then  # Show first 5 steps
                        echo "       Step $step_id: $nodes nodes, $edges edges, depth $depth"
                    fi
                    
                    prev_nodes="$nodes"
                fi
            done <<< "$recipe_data"
            
            # Check if too many identical node counts (indicating no synthesis progression)
            if [ $total_steps -gt 0 ] && [ $((identical_count * 100 / total_steps)) -gt 80 ]; then
                echo "       🚨 WARNING: $identical_count/$total_steps steps have identical node counts!"
                echo "       This suggests synthesis may not be progressing properly."
                csv_issues_found=$((csv_issues_found + 1))
            elif [ $total_steps -gt 0 ]; then
                percentage=$((identical_count * 100 / total_steps))
                echo "       ✅ Good variation: $identical_count/$total_steps identical (${percentage}%)"
            fi
        fi
    done
    
    # Quick sanity check: are all entries identical?
    unique_lines=$(tail -n +2 "$csv_file" 2>/dev/null | sort -u | wc -l | tr -d ' ')
    if [ "$unique_lines" -lt 10 ] && [ "$total_lines" -gt 100 ]; then
        echo "   🚨 CRITICAL: Only $unique_lines unique entries out of $total_lines total!"
        echo "   This indicates CSV generation failed - all entries are nearly identical."
        csv_issues_found=$((csv_issues_found + 1))
    else
        echo "   ✅ CSV diversity: $unique_lines unique entries out of $total_lines total"
    fi
    
    echo ""
done

echo "📊 CSV COMPLETENESS SUMMARY:"
echo "============================"
if [ "$csv_issues_found" -eq 0 ]; then
    echo "✅ All CSV files look complete and show proper synthesis progression"
    echo "   • Expected number of entries found"
    echo "   • Node counts vary properly across synthesis steps"
    echo "   • Good diversity in synthesis results"
else
    echo "⚠️  Found $csv_issues_found CSV-related issues"
    echo "   • Check the warnings above for specific problems"
    echo "   • May need to re-run metadata collection (job 5)"
fi

echo ""
echo "=========================================="
echo "COMPREHENSIVE VERIFICATION RESULTS"
echo "=========================================="
echo ""

echo "📊 OVERALL STATISTICS:"
echo "====================="
echo "Total log files checked: $total_logs"
echo "Files with expected FA_X1 warnings: $logs_with_fa_warning"
echo "Files with unexpected issues: $logs_with_other_issues"
echo "Files with no issues at all: $((total_logs - logs_with_fa_warning - logs_with_other_issues))"
echo ""

if [ "$logs_with_other_issues" -eq 0 ]; then
    echo "✅ VERIFICATION PASSED!"
    echo "======================================"
    echo "🎉 All log files are clean!"
    echo "   • Only the expected benign FA_X1 multi-output warnings found"
    echo "   • No actual errors or unexpected warnings detected"
    echo "   • Your synthesis pipeline is working perfectly"
    echo ""
    
    if [ "$logs_with_fa_warning" -gt 0 ]; then
        echo "ℹ️  The FA_X1 warnings are completely normal and expected when using"
        echo "   standard cell libraries with multi-output cells like full adders."
        echo "   These can be safely ignored."
    fi
else
    echo "⚠️  VERIFICATION FOUND ISSUES!"
    echo "======================================"
    echo "Found $logs_with_other_issues files with unexpected problems:"
    echo ""
    
    # List problematic files
    for log_file in "${problematic_logs[@]}"; do
        echo "  • $(basename "$log_file")"
    done
    
    echo ""
    echo "🔧 RECOMMENDATION: Investigate the files listed above for:"
    echo "   - Unexpected error messages"
    echo "   - Synthesis failures"
    echo "   - File I/O problems"
    echo "   - ABC command failures"
fi

echo ""
echo "=========================================="
echo "End time: $(date)"
echo ""

