#!/usr/bin/env python3
"""
Master orchestration script for creating the Full AIG Dataset.

This script coordinates the entire process:
1. Reorganizes Random AIG and OpenABC-D datasets into FULL_DATASET structure
2. Generates metadata CSV files with AIG statistics
3. Validates the resulting dataset
4. Provides comprehensive reporting

Usage:
    python create_full_dataset.py --random-path /path/to/random --openabc-path /path/to/openabc --output /path/to/FULL_DATASET
"""

import os
import sys
import argparse
import json
import subprocess
import time
from datetime import datetime

def run_command(cmd, description, check=True):
    """
    Run a command and handle errors gracefully.
    """
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=check)
        
        end_time = time.time()
        duration = end_time - start_time
        
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
        
        if result.stderr and result.returncode == 0:
            print("STDERR (warnings):")
            print(result.stderr)
        
        print(f"\n✓ Completed in {duration:.1f} seconds")
        return result
        
    except subprocess.CalledProcessError as e:
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n✗ FAILED after {duration:.1f} seconds")
        print(f"Return code: {e.returncode}")
        
        if e.stdout:
            print("STDOUT:")
            print(e.stdout)
        
        if e.stderr:
            print("STDERR:")
            print(e.stderr)
        
        if check:
            print(f"\nError in step: {description}")
            sys.exit(1)
        
        return e

def check_prerequisites():
    """
    Check that required tools are available.
    """
    print("Checking prerequisites...")
    
    required_tools = ['abc', 'python3']
    missing_tools = []
    
    for tool in required_tools:
        try:
            result = subprocess.run(['which', tool], capture_output=True, text=True, check=False)
            if result.returncode == 0:
                print(f"  ✓ {tool}: {result.stdout.strip()}")
            else:
                missing_tools.append(tool)
        except (ImportError, subprocess.CalledProcessError):
            missing_tools.append(tool)
    
    if missing_tools:
        print(f"  ✗ Missing tools: {', '.join(missing_tools)}")
        print("\nPlease install the missing tools:")
        print("  - abc: ABC synthesis tool (install from Berkeley)")
        print("  - python3: Python 3.6+ with required packages")
        return False
    
    # Check Python packages
    required_packages = ['tqdm', 'pandas']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✓ Python package '{package}' available")
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"  ✗ Missing Python packages: {', '.join(missing_packages)}")
        print(f"    Install with: pip install {' '.join(missing_packages)}")
        return False
    
    return True

def validate_input_paths(args):
    """
    Validate that input paths exist and contain expected structure.
    """
    print("Validating input paths...")
    
    if not os.path.exists(args.random_path):
        print(f"  ✗ Random dataset path does not exist: {args.random_path}")
        return False
    print(f"  ✓ Random dataset path exists: {args.random_path}")
    
    # Check for expected Random dataset structure
    random_bench = os.path.join(args.random_path, 'OPENABC_DATASET', 'bench')
    if not os.path.exists(random_bench):
        # Try alternative structure
        random_bench = os.path.join(args.random_path, 'bench')
        if not os.path.exists(random_bench):
            print("  ✗ Random dataset bench directory not found")
            return False
    print(f"  ✓ Random dataset bench directory found: {random_bench}")
    
    # Check for some expected designs
    expected_random_designs = ['128', '256', '512', '1024']
    found_designs = []
    for design in expected_random_designs:
        design_path = os.path.join(random_bench, design)
        if os.path.exists(design_path):
            found_designs.append(design)
    
    if not found_designs:
        print(f"  ✗ No expected random designs found in {random_bench}")
        return False
    print(f"  ✓ Found random designs: {', '.join(found_designs)}")
    
    # Validate OpenABC path if provided
    if args.openabc_path:
        if not os.path.exists(args.openabc_path):
            print(f"  ✗ OpenABC dataset path does not exist: {args.openabc_path}")
            return False
        print(f"  ✓ OpenABC dataset path exists: {args.openabc_path}")
    
    # Check library files
    if args.lib_paths:
        for lib_path in args.lib_paths:
            if not os.path.exists(lib_path):
                print(f"  ✗ Library path does not exist: {lib_path}")
                return False
        print("  ✓ All library paths exist")
    
    return True

def create_progress_log(output_path):
    """
    Create a progress log file to track the process.
    """
    log_path = os.path.join(output_path, 'creation_progress.json')
    progress = {
        'start_time': datetime.now().isoformat(),
        'steps': [],
        'status': 'in_progress'
    }
    
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(progress, f, indent=2)
    
    return log_path

def update_progress_log(log_path, step_name, status, details=None):
    """
    Update the progress log with step completion.
    """
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            progress = json.load(f)
        
        step_info = {
            'name': step_name,
            'status': status,
            'timestamp': datetime.now().isoformat()
        }
        
        if details:
            step_info['details'] = details
        
        progress['steps'].append(step_info)
        
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(progress, f, indent=2)
    
    except (OSError, json.JSONDecodeError) as e:
        print(f"Warning: Could not update progress log: {e}")

def main():
    parser = argparse.ArgumentParser(description='Create Full AIG Dataset from Random and OpenABC-D datasets')
    
    # Input paths
    parser.add_argument('--random-path', required=True,
                       help='Path to Random AIG dataset')
    parser.add_argument('--openabc-path',
                       help='Path to OpenABC-D dataset (if separate)')
    parser.add_argument('--lib-paths', nargs='*',
                       help='Paths to library files (.lib)')
    
    # Output configuration
    parser.add_argument('--output', '-o', required=True,
                       help='Output directory for FULL_DATASET')
    
    # Processing options
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of parallel workers for metadata generation (default: 4)')
    parser.add_argument('--designs', nargs='*',
                       help='Process only specific designs (default: all)')
    
    # Execution control
    parser.add_argument('--skip-reorganize', action='store_true',
                       help='Skip dataset reorganization step')
    parser.add_argument('--skip-metadata', action='store_true',
                       help='Skip metadata generation step')
    parser.add_argument('--skip-validation', action='store_true',
                       help='Skip validation step')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be done without doing it')
    
    # Debugging
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    if args.dry_run:
        print("DRY RUN MODE - No changes will be made")
    
    print(f"""
Full AIG Dataset Creation
========================
Random dataset: {args.random_path}
OpenABC dataset: {args.openabc_path or 'Included in random dataset'}
Output: {args.output}
Workers: {args.workers}
Library paths: {args.lib_paths or 'None'}
""")
    
    # Check prerequisites
    if not check_prerequisites():
        print("Prerequisites check failed. Please install required tools.")
        sys.exit(1)
    
    # Validate input paths
    if not validate_input_paths(args):
        print("Input path validation failed.")
        sys.exit(1)
    
    # Create output directory and progress log
    if not args.dry_run:
        os.makedirs(args.output, exist_ok=True)
        log_path = create_progress_log(args.output)
    else:
        log_path = None
    
    start_time = time.time()
    
    try:
        # Step 1: Dataset Reorganization
        if not args.skip_reorganize:
            cmd = [
                'python3', 'reorganize_datasets.py',
                '--random-dataset', args.random_path,
                '--output', args.output
            ]
            
            if args.openabc_path:
                cmd.extend(['--openabc-dataset', args.openabc_path])
            
            if args.lib_paths:
                cmd.extend(['--lib-paths'] + args.lib_paths)
            
            if args.dry_run:
                cmd.append('--dry-run')
            
            run_command(cmd, "Dataset Reorganization", check=not args.dry_run)
            
            if log_path:
                update_progress_log(log_path, "reorganization", "completed" if not args.dry_run else "skipped")
        else:
            print("\nSkipping dataset reorganization step")
            if log_path:
                update_progress_log(log_path, "reorganization", "skipped")
        
        # Step 2: Metadata Generation
        if not args.skip_metadata and not args.dry_run:
            cmd = [
                'python3', 'generate_metadata.py',
                args.output,
                '--workers', str(args.workers),
                '--validate',
                '--summary'
            ]
            
            if args.lib_paths and len(args.lib_paths) > 0:
                cmd.extend(['--lib', args.lib_paths[0]])  # Use first library file
            
            if args.designs:
                for design in args.designs:
                    design_cmd = cmd + ['--design', design]
                    run_command(design_cmd, f"Metadata Generation for {design}")
            else:
                run_command(cmd, "Metadata Generation")
            
            if log_path:
                update_progress_log(log_path, "metadata_generation", "completed")
        else:
            if args.skip_metadata:
                print("\nSkipping metadata generation step")
            if log_path:
                update_progress_log(log_path, "metadata_generation", "skipped")
        
        # Final validation and reporting
        if not args.skip_validation and not args.dry_run:
            print(f"\n{'='*60}")
            print("FINAL VALIDATION AND REPORTING")
            print(f"{'='*60}")
            
            # Check if dataset manifest exists
            manifest_path = os.path.join(args.output, 'dataset_manifest.json')
            if os.path.exists(manifest_path):
                print(f"✓ Dataset manifest found: {manifest_path}")
                with open(manifest_path, 'r', encoding='utf-8') as f:
                    manifest = json.load(f)
                    print(f"  - Total designs: {manifest['statistics']['total_designs']}")
                    print(f"  - Random designs: {manifest['statistics']['random_designs']}")
                    print(f"  - OpenABC designs: {manifest['statistics']['openabc_designs']}")
            else:
                print(f"✗ Dataset manifest not found: {manifest_path}")
            
            # Check summary statistics
            summary_path = os.path.join(args.output, 'metadata', 'stats', 'dataset_summary.json')
            if os.path.exists(summary_path):
                print(f"✓ Dataset summary found: {summary_path}")
                with open(summary_path, 'r', encoding='utf-8') as f:
                    summary = json.load(f)
                    print(f"  - Total files processed: {summary['totals']['files']}")
                    print(f"  - Designs with data: {summary['totals']['designs']}")
            else:
                print(f"✗ Dataset summary not found: {summary_path}")
            
            if log_path:
                update_progress_log(log_path, "validation", "completed")
        
        # Complete the process
        end_time = time.time()
        total_duration = end_time - start_time
        
        if log_path:
            with open(log_path, 'r', encoding='utf-8') as f:
                progress = json.load(f)
            progress['status'] = 'completed'
            progress['end_time'] = datetime.now().isoformat()
            progress['total_duration_seconds'] = total_duration
            with open(log_path, 'w', encoding='utf-8') as f:
                json.dump(progress, f, indent=2)
        
        print(f"\n{'='*60}")
        print("FULL DATASET CREATION COMPLETED SUCCESSFULLY!")
        print(f"{'='*60}")
        print(f"Total time: {total_duration/60:.1f} minutes")
        print(f"Output location: {args.output}")
        
        if not args.dry_run:
            print("\nDataset structure:")
            print(f"  {args.output}/base_aigs/          - Original and synthesized AIG files")
            print(f"  {args.output}/synScripts/        - ABC synthesis scripts")
            print(f"  {args.output}/metadata/stats/    - Per-design CSV files with statistics")
            print(f"  {args.output}/metadata/library/  - Technology library files")
            print(f"  {args.output}/optimized_aigs/    - Ready for algorithm outputs")
            
            print("\nNext steps:")
            print("1. Run algorithm optimization pipelines (Orchestrate, Deepsyn, Syn4, C2RS)")
            print("2. Validate dataset integrity")
            print("3. Begin ML experiments")
    
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user")
        if log_path:
            update_progress_log(log_path, "process", "interrupted")
        sys.exit(1)
    
    except (subprocess.CalledProcessError, OSError) as e:
        print(f"\n\nUnexpected error: {e}")
        if log_path:
            update_progress_log(log_path, "process", "error", str(e))
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()