#!/usr/bin/env python3
"""
Reorganize Random AIG Dataset and OpenABC-D Dataset into Full Dataset structure.

This script:
1. Creates the FULL_DATASET directory structure
2. Moves/copies AIG files from both datasets into the unified structure
3. Organizes synthesis scripts
4. Prepares for metadata generation
"""

import os
import shutil
import argparse
import sys
import glob
import zipfile
from pathlib import Path
import json

# Design lists
RANDOM_DESIGNS = ['128', '256', '512', '1024', '2048', '4096', '8192', '16384']

OPENABC_DESIGNS = [
    'i2c', 'spi', 'des3_area', 'ss_pcm', 'usb_phy', 'sasc', 'wb_dma', 'simple_spi',
    'dynamic_node', 'aes', 'pci', 'ac97_ctrl', 'mem_ctrl', 'tv80', 'fpu',
    'wb_conmax', 'tinyRocket', 'aes_xcrypt', 'aes_secworks',
    'jpeg', 'bp_be', 'ethernet', 'vga_lcd', 'picosoc',
    'dft', 'idft', 'fir', 'iir', 'sha256'
]

ALL_DESIGNS = RANDOM_DESIGNS + OPENABC_DESIGNS

def create_full_dataset_structure(output_dir):
    """
    Create the FULL_DATASET directory structure.
    """
    structure = {
        'base_aigs': {},
        'synScripts': None,
        'optimized_aigs': {
            'Orchestrate': {'tier1': {}, 'tier2': {}},
            'Deepsyn': {'tier1': {}, 'tier2': {}},
            'Syn4': {'tier1': {}, 'tier2': {}},
            'C2RS': {'tier1': {}, 'tier2': {}}
        },
        'metadata': {
            'stats': None,
            'library': None
        }
    }
    
    print(f"Creating FULL_DATASET structure in {output_dir}...")
    
    # Create base directories
    os.makedirs(os.path.join(output_dir, 'base_aigs'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'synScripts'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'metadata', 'stats'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'metadata', 'library'), exist_ok=True)
    
    # Create design-specific directories under base_aigs
    for design in ALL_DESIGNS:
        os.makedirs(os.path.join(output_dir, 'base_aigs', design), exist_ok=True)
    
    # Create algorithm and tier directories
    for algorithm in ['Orchestrate', 'Deepsyn', 'Syn4', 'C2RS']:
        for tier in ['tier1', 'tier2']:
            tier_path = os.path.join(output_dir, 'optimized_aigs', algorithm, tier)
            os.makedirs(tier_path, exist_ok=True)
            # Create design subdirectories
            for design in ALL_DESIGNS:
                os.makedirs(os.path.join(tier_path, design), exist_ok=True)
    
    print("Directory structure created successfully.")
    return True

def process_random_dataset(random_dataset_path, full_dataset_path):
    """
    Process the Random AIG dataset and move files to FULL_DATASET structure.
    """
    print("Processing Random AIG dataset...")
    
    bench_path = os.path.join(random_dataset_path, 'bench')
    if not os.path.exists(bench_path):
        print(f"Random dataset bench path not found: {bench_path}")
        return False
    
    processed_count = 0
    
    for design in RANDOM_DESIGNS:
        design_path = os.path.join(bench_path, design)
        if not os.path.exists(design_path):
            print(f"Design path not found: {design_path}")
            continue
            
        target_path = os.path.join(full_dataset_path, 'base_aigs', design)
        
        # Process original AIG file
        orig_file = os.path.join(design_path, f'{design}_orig.aig')
        if os.path.exists(orig_file):
            target_orig = os.path.join(target_path, f'{design}_orig.aig')
            shutil.copy2(orig_file, target_orig)
            processed_count += 1
            print(f"  Copied {design}_orig.aig")
        
        # Process synthesized AIG files (if they exist in zip files)
        for recipe_id in range(1500):  # 0 to 1499
            zip_file = os.path.join(design_path, f'syn{recipe_id}.zip')
            if os.path.exists(zip_file):
                with zipfile.ZipFile(zip_file, 'r') as zf:
                    for step_id in range(1, 22):  # steps 1-21
                        aig_name = f'{design}_syn{recipe_id}_step{step_id}.aig'
                        if aig_name in zf.namelist():
                            target_file = os.path.join(target_path, aig_name)
                            with zf.open(aig_name) as source, open(target_file, 'wb') as target:
                                shutil.copyfileobj(source, target)
                            processed_count += 1
                
                if recipe_id % 100 == 0:
                    print(f"  Processed {design} up to recipe {recipe_id}")
    
    # Copy synthesis scripts
    synscripts_src = os.path.join(random_dataset_path, 'synScripts')
    if os.path.exists(synscripts_src):
        for design in RANDOM_DESIGNS:
            design_scripts_src = os.path.join(synscripts_src, design)
            if os.path.exists(design_scripts_src):
                # Create zip archive for this design's scripts
                design_zip = os.path.join(full_dataset_path, 'synScripts', f'{design}.zip')
                with zipfile.ZipFile(design_zip, 'w') as zf:
                    for script_file in glob.glob(os.path.join(design_scripts_src, '*.script')):
                        zf.write(script_file, os.path.basename(script_file))
                print(f"  Created synthesis scripts archive for {design}")
    
    print(f"Random dataset processing complete. Processed {processed_count} files.")
    return True

def process_openabc_dataset(openabc_dataset_path, full_dataset_path):
    """
    Process the OpenABC-D dataset and move files to FULL_DATASET structure.
    
    Note: This assumes the OpenABC-D dataset has a similar structure to Random dataset
    but with different design names.
    """
    print("Processing OpenABC-D dataset...")
    
    # The OpenABC-D dataset might be structured differently
    # Check for common patterns
    possible_paths = [
        os.path.join(openabc_dataset_path, 'bench'),
        os.path.join(openabc_dataset_path, 'OPENABC_DATASET', 'bench'),
        openabc_dataset_path  # In case designs are directly in the root
    ]
    
    bench_path = None
    for path in possible_paths:
        if os.path.exists(path):
            bench_path = path
            break
    
    if not bench_path:
        print(f"OpenABC-D dataset bench path not found in {openabc_dataset_path}")
        return False
    
    processed_count = 0
    
    for design in OPENABC_DESIGNS:
        design_path = os.path.join(bench_path, design)
        if not os.path.exists(design_path):
            print(f"Design path not found: {design_path}")
            continue
            
        target_path = os.path.join(full_dataset_path, 'base_aigs', design)
        
        # Process original AIG file
        orig_file = os.path.join(design_path, f'{design}_orig.aig')
        if os.path.exists(orig_file):
            target_orig = os.path.join(target_path, f'{design}_orig.aig')
            shutil.copy2(orig_file, target_orig)
            processed_count += 1
            print(f"  Copied {design}_orig.aig")
        
        # Process synthesized AIG files
        # Check both zip files and direct AIG files
        for recipe_id in range(1500):
            zip_file = os.path.join(design_path, f'syn{recipe_id}.zip')
            if os.path.exists(zip_file):
                with zipfile.ZipFile(zip_file, 'r') as zf:
                    for step_id in range(1, 22):
                        aig_name = f'{design}_syn{recipe_id}_step{step_id}.aig'
                        if aig_name in zf.namelist():
                            target_file = os.path.join(target_path, aig_name)
                            with zf.open(aig_name) as source, open(target_file, 'wb') as target:
                                shutil.copyfileobj(source, target)
                            processed_count += 1
            else:
                # Check for direct AIG files
                for step_id in range(1, 22):
                    aig_file = os.path.join(design_path, f'{design}_syn{recipe_id}_step{step_id}.aig')
                    if os.path.exists(aig_file):
                        target_file = os.path.join(target_path, f'{design}_syn{recipe_id}_step{step_id}.aig')
                        shutil.copy2(aig_file, target_file)
                        processed_count += 1
        
        if processed_count % 1000 == 0:
            print(f"  Processed {processed_count} files so far...")
    
    # Copy synthesis scripts if available
    synscripts_paths = [
        os.path.join(openabc_dataset_path, 'synScripts'),
        os.path.join(openabc_dataset_path, 'OPENABC_DATASET', 'synScripts')
    ]
    
    for synscripts_src in synscripts_paths:
        if os.path.exists(synscripts_src):
            # Check if there's a single zip file or design-specific directories
            synscripts_zip = os.path.join(synscripts_src, 'synScripts.zip')
            if os.path.exists(synscripts_zip):
                # Copy the main synScripts.zip
                target_zip = os.path.join(full_dataset_path, 'synScripts', 'synScripts.zip')
                shutil.copy2(synscripts_zip, target_zip)
                print("  Copied main synScripts.zip")
            else:
                # Create per-design zip files
                for design in OPENABC_DESIGNS:
                    design_scripts_src = os.path.join(synscripts_src, design)
                    if os.path.exists(design_scripts_src):
                        design_zip = os.path.join(full_dataset_path, 'synScripts', f'{design}.zip')
                        with zipfile.ZipFile(design_zip, 'w') as zf:
                            for script_file in glob.glob(os.path.join(design_scripts_src, '*.script')):
                                zf.write(script_file, os.path.basename(script_file))
                        print(f"  Created synthesis scripts archive for {design}")
            break
    
    print(f"OpenABC-D dataset processing complete. Processed {processed_count} files.")
    return True

def copy_library_files(source_paths, full_dataset_path):
    """
    Copy library files to the FULL_DATASET structure.
    """
    print("Copying library files...")
    
    lib_target_path = os.path.join(full_dataset_path, 'metadata', 'library')
    
    for source_path in source_paths:
        if os.path.exists(source_path):
            if os.path.isdir(source_path):
                # Copy all library files from directory
                for lib_file in glob.glob(os.path.join(source_path, '*.lib')):
                    shutil.copy2(lib_file, lib_target_path)
                    print(f"  Copied {os.path.basename(lib_file)}")
                
                # Create zip archive if multiple files
                lib_files = glob.glob(os.path.join(lib_target_path, '*.lib'))
                if len(lib_files) > 0:
                    zip_path = os.path.join(lib_target_path, 'nangate45.lib.zip')
                    with zipfile.ZipFile(zip_path, 'w') as zf:
                        for lib_file in lib_files:
                            zf.write(lib_file, os.path.basename(lib_file))
                    print(f"  Created library archive: nangate45.lib.zip")
            else:
                # Single library file
                shutil.copy2(source_path, lib_target_path)
                print(f"  Copied {os.path.basename(source_path)}")

def create_dataset_manifest(full_dataset_path):
    """
    Create a manifest file describing the dataset structure.
    """
    manifest = {
        'dataset_name': 'Full AIG Dataset',
        'version': '1.0',
        'designs': {
            'random': RANDOM_DESIGNS,
            'openabc': OPENABC_DESIGNS
        },
        'structure': {
            'base_aigs': 'Original and synthesized AIG files',
            'synScripts': 'ABC synthesis scripts per design',
            'optimized_aigs': 'Algorithm outputs (tiered)',
            'metadata': 'Statistics and library files'
        },
        'statistics': {
            'total_designs': len(ALL_DESIGNS),
            'random_designs': len(RANDOM_DESIGNS),
            'openabc_designs': len(OPENABC_DESIGNS),
            'recipes_per_design': 1500,
            'steps_per_recipe': 21,
            'expected_base_aigs': len(ALL_DESIGNS) * (1 + 1500 * 21)
        }
    }
    
    manifest_path = os.path.join(full_dataset_path, 'dataset_manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"Dataset manifest created: {manifest_path}")

def main():
    parser = argparse.ArgumentParser(description='Reorganize datasets into FULL_DATASET structure')
    parser.add_argument('--random-dataset', required=True, 
                       help='Path to Random AIG dataset (containing OPENABC_DATASET/)')
    parser.add_argument('--openabc-dataset', 
                       help='Path to OpenABC-D dataset (if separate from random)')
    parser.add_argument('--output', '-o', required=True, 
                       help='Output directory for FULL_DATASET')
    parser.add_argument('--lib-paths', nargs='*', 
                       help='Paths to library files or directories')
    parser.add_argument('--dry-run', action='store_true', 
                       help='Show what would be done without actually doing it')
    
    args = parser.parse_args()
    
    if args.dry_run:
        print("DRY RUN MODE - No files will be moved/copied")
    
    # Validate input paths
    if not os.path.exists(args.random_dataset):
        print(f"Random dataset path does not exist: {args.random_dataset}")
        sys.exit(1)
    
    # Create output directory
    if not args.dry_run:
        os.makedirs(args.output, exist_ok=True)
        create_full_dataset_structure(args.output)
    
    # Process Random dataset
    if not args.dry_run:
        if not process_random_dataset(args.random_dataset, args.output):
            print("Failed to process Random dataset")
            sys.exit(1)
    
    # Process OpenABC dataset (if provided separately)
    if args.openabc_dataset:
        if not os.path.exists(args.openabc_dataset):
            print(f"OpenABC dataset path does not exist: {args.openabc_dataset}")
            sys.exit(1)
        
        if not args.dry_run:
            if not process_openabc_dataset(args.openabc_dataset, args.output):
                print("Failed to process OpenABC dataset")
                sys.exit(1)
    else:
        print("Note: No separate OpenABC dataset path provided. Assuming it's included in Random dataset.")
    
    # Copy library files
    if args.lib_paths and not args.dry_run:
        copy_library_files(args.lib_paths, args.output)
    
    # Create manifest
    if not args.dry_run:
        create_dataset_manifest(args.output)
    
    print("\nDataset reorganization complete!")
    print(f"Full dataset available at: {args.output}")
    print("\nNext steps:")
    print("1. Run metadata generation script to create per-design CSV files")
    print("2. Verify all expected files are present")
    print("3. Run algorithm optimization pipelines")

if __name__ == '__main__':
    main()