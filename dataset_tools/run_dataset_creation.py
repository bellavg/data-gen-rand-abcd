#!/usr/bin/env python3
"""
Quick script to run the full dataset creation process.

This script assumes you have:
1. Random AIG dataset in OPENABC_DATASET/ (current location)  
2. Potentially OpenABC-D dataset elsewhere (will be auto-detected)
3. Want to create FULL_DATASET in the current directory

Usage:
    python run_dataset_creation.py [--output-dir FULL_DATASET]
"""

import os
import sys
import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser(description='Run full dataset creation process')
    parser.add_argument('--output-dir', default='FULL_DATASET', 
                       help='Output directory name (default: FULL_DATASET)')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of parallel workers (default: 4)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be done without doing it')
    
    args = parser.parse_args()
    
    # Get current directory (parent of dataset_tools)
    current_dir = os.path.dirname(os.getcwd())
    print(f"Working directory: {current_dir}")
    
    # Handle absolute vs relative output paths
    if os.path.isabs(args.output_dir):
        output_path = args.output_dir
    else:
        output_path = os.path.join(current_dir, args.output_dir)
    
    # Expected paths based on your current structure
    random_dataset = os.path.join(current_dir, 'OPENABC_DATASET')
    
    if not os.path.exists(random_dataset):
        print(f"Error: Random dataset not found at {random_dataset}")
        print("Expected structure: ./OPENABC_DATASET/bench/{128,256,512,...}/")
        sys.exit(1)
    
    print(f"Input (Random AIG): {random_dataset}")
    print(f"Output (Full Dataset): {output_path}")
    
    # Build command
    cmd = [
        'python3', 'create_full_dataset.py',
        '--random-path', random_dataset,
        '--output', output_path,
        '--workers', str(args.workers)
    ]
    
    if args.dry_run:
        cmd.append('--dry-run')
        print("DRY RUN MODE - No files will be modified")
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
        print("Dataset creation completed successfully!")
        
        if not args.dry_run:
            print(f"\nYour unified dataset is ready at: {output_path}")
            print("\nNext steps:")
            print("1. Verify the dataset structure")
            print("2. Run algorithm optimization pipelines")
            print("3. Begin ML experiments")
            
    except subprocess.CalledProcessError as e:
        print(f"Error: Dataset creation failed with exit code {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
        sys.exit(1)

if __name__ == '__main__':
    main()