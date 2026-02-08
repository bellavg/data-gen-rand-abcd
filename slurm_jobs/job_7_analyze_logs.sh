#!/bin/bash
#SBATCH --job-name=log_analysis
#SBATCH --time=00:20:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=genoa
#SBATCH --output=logs/log_analysis_%j.out

# Job 7: Comprehensive Log Analysis
# Analyzes all synthesis logs for warnings, errors, and common issues
# Run this after synthesis jobs (4a-4h) and metadata collection (job 5) complete

set -e  # Exit on error

echo "=========================================="
echo "COMPREHENSIVE SYNTHESIS LOG ANALYSIS"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "Start time: $(date)"
echo ""

# Define paths
BASE_DIR="$HOME/data-gen-rand-abcd"
LOGS_DIR="${BASE_DIR}/logs"
ANALYSIS_DIR="${BASE_DIR}/analysis_reports"

echo "Configuration:"
echo "  Base directory: ${BASE_DIR}"
echo "  Logs directory: ${LOGS_DIR}"
echo "  Analysis output: ${ANALYSIS_DIR}"
echo ""

# Create analysis directory
mkdir -p "${ANALYSIS_DIR}"

# Check if logs directory exists
if [ ! -d "$LOGS_DIR" ]; then
    echo "❌ ERROR: Logs directory not found: $LOGS_DIR"
    echo "Please ensure synthesis jobs have been completed."
    exit 1
fi

echo "✓ Found logs directory: $LOGS_DIR"
echo ""

# Function to analyze a single log file
analyze_log_file() {
    local log_file="$1"
    local basename=$(basename "$log_file" .out)
    
    # Count different types of issues
    local general_warnings=$(grep -c -i "warning:" "$log_file" 2>/dev/null || echo "0")
    local system_warnings=$(grep -c -i "WARNING:" "$log_file" 2>/dev/null || echo "0")
    local errors=$(grep -c -i "error:" "$log_file" 2>/dev/null || echo "0")
    local system_errors=$(grep -c -i "ERROR:" "$log_file" 2>/dev/null || echo "0")
    local abc_cmd_errors=$(grep -c "\*\* cmd error:" "$log_file" 2>/dev/null || echo "0")
    local zip_errors=$(grep -c "zip error:" "$log_file" 2>/dev/null || echo "0")
    
    # Output results in CSV format
    echo "$basename,$general_warnings,$system_warnings,$errors,$system_errors,$abc_cmd_errors,$zip_errors"
}

# Generate comprehensive analysis report
generate_analysis_report() {
    local report_file="${ANALYSIS_DIR}/synthesis_analysis_$(date +%Y%m%d_%H%M%S).txt"
    local csv_file="${ANALYSIS_DIR}/synthesis_issues_$(date +%Y%m%d_%H%M%S).csv"
    
    echo "📊 Generating comprehensive analysis report..."
    
    # Create CSV header
    echo "LogFile,GeneralWarnings,SystemWarnings,GeneralErrors,SystemErrors,ABCCommandErrors,ZipErrors" > "$csv_file"
    
    # Analyze each log file
    local total_logs=0
    local logs_with_issues=0
    local total_warnings=0
    local total_errors=0
    
    echo "🔍 Analyzing individual log files..."
    
    # Find all .out files and analyze them
    while IFS= read -r log_file; do
        if [ -f "$log_file" ]; then
            total_logs=$((total_logs + 1))
            local result=$(analyze_log_file "$log_file")
            echo "$result" >> "$csv_file"
            
            # Check if this file has any issues
            local issue_count=$(echo "$result" | cut -d',' -f2- | tr ',' '+' | bc 2>/dev/null || echo "0")
            if [ "$issue_count" -gt 0 ]; then
                logs_with_issues=$((logs_with_issues + 1))
                
                # Extract counts for totals
                local gw=$(echo "$result" | cut -d',' -f2)
                local sw=$(echo "$result" | cut -d',' -f3)
                local ge=$(echo "$result" | cut -d',' -f4)
                local se=$(echo "$result" | cut -d',' -f5)
                local ace=$(echo "$result" | cut -d',' -f6)
                local ze=$(echo "$result" | cut -d',' -f7)
                
                total_warnings=$((total_warnings + gw + sw))
                total_errors=$((total_errors + ge + se + ace + ze))
            fi
            
            # Progress indicator
            if [ $((total_logs % 50)) -eq 0 ]; then
                echo "   Processed $total_logs files..."
            fi
        fi
    done < <(find "$LOGS_DIR" -name "*.out" | sort)
    
    # Generate text report
    {
        echo "SYNTHESIS LOG ANALYSIS REPORT"
        echo "Generated: $(date)"
        echo "Base Directory: $BASE_DIR"
        echo "======================================"
        echo ""
        
        echo "SUMMARY STATISTICS:"
        echo "=================="
        echo "Total log files analyzed: $total_logs"
        echo "Files with issues: $logs_with_issues"
        echo "Files without issues: $((total_logs - logs_with_issues))"
        echo "Total warnings found: $total_warnings"
        echo "Total errors found: $total_errors"
        echo ""
        
        if [ "$total_logs" -gt 0 ]; then
            echo "Issue percentage: $(( (logs_with_issues * 100) / total_logs ))%"
        fi
        echo ""
        
        echo "BREAKDOWN BY JOB TYPE:"
        echo "====================="
        for job_pattern in "synthesis_128" "synthesis_256" "synthesis_512" "synthesis_1024" "synthesis_2048" "synthesis_4096" "synthesis_8192" "synthesis_16384" "collect_metadata"; do
            local pattern_files=$(find "$LOGS_DIR" -name "*${job_pattern}*" 2>/dev/null | wc -l | tr -d ' ')
            if [ "$pattern_files" -gt 0 ]; then
                local pattern_issues=0
                while IFS= read -r log_file; do
                    if [ -f "$log_file" ]; then
                        local issue_count=$(grep -c -i "warning\|error" "$log_file" 2>/dev/null || echo "0")
                        pattern_issues=$((pattern_issues + issue_count))
                    fi
                done < <(find "$LOGS_DIR" -name "*${job_pattern}*" 2>/dev/null)
                
                echo "$job_pattern: $pattern_files files, $pattern_issues issues"
            fi
        done
        echo ""
        
        echo "MOST FREQUENT WARNING TYPES:"
        echo "============================"
        find "$LOGS_DIR" -name "*.out" -exec grep -h -i "warning" {} \; 2>/dev/null | \
            head -100 | sort | uniq -c | sort -nr | head -10 | while read count msg; do
            echo "[$count occurrences] $msg"
        done
        echo ""
        
        echo "MOST FREQUENT ERROR TYPES:"
        echo "=========================="
        find "$LOGS_DIR" -name "*.out" -exec grep -h -i "error" {} \; 2>/dev/null | \
            head -100 | sort | uniq -c | sort -nr | head -10 | while read count msg; do
            echo "[$count occurrences] $msg"
        done
        echo ""
        
        echo "SAMPLE PROBLEMATIC FILES:"
        echo "========================"
        # Show top 10 files with most issues
        if [ -f "$csv_file" ]; then
            tail -n +2 "$csv_file" | while IFS=, read file gw sw ge se ace ze; do
                total=$((gw + sw + ge + se + ace + ze))
                echo "$total,$file"
            done | sort -nr | head -10 | while IFS=, read total file; do
                echo "$file: $total total issues"
            done
        fi
        
    } > "$report_file"
    
    echo "✅ Analysis complete!"
    echo "📄 CSV data saved to: $csv_file"
    echo "📊 Full report saved to: $report_file"
    echo ""
    
    # Display quick summary
    echo "QUICK SUMMARY:"
    echo "=============="
    echo "📁 Total log files: $total_logs"
    echo "⚠️  Files with issues: $logs_with_issues"
    echo "✅ Clean files: $((total_logs - logs_with_issues))"
    echo "🔶 Total warnings: $total_warnings"
    echo "🔴 Total errors: $total_errors"
    
    if [ "$total_logs" -gt 0 ]; then
        echo "📊 Issue rate: $(( (logs_with_issues * 100) / total_logs ))%"
    fi
}

# Main execution
echo "🔍 Starting comprehensive log analysis..."
generate_analysis_report

echo ""
echo "=========================================="
echo "Log Analysis Complete"
echo "=========================================="
echo "End time: $(date)"
echo ""
echo "Next steps:"
echo "1. Review the analysis reports in: ${ANALYSIS_DIR}"
echo "2. Check CSV file for detailed per-file breakdown"
echo "3. Investigate high-issue files if needed"