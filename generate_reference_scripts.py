#!/usr/bin/env python3
"""
Generate 1500 random synthesis recipe scripts for ABC
Each recipe contains L=20 random atomic transformations

Based on the OpenABC-D methodology - uses EXACT same commands as the original paper
"""

import os
import random
import argparse

# ABC atomic synthesis commands from the original OpenABC-D paper
# These are the EXACT transformations used in the original dataset
ABC_COMMANDS = [
    'balance',
    'rewrite',
    'rewrite -z',
    'refactor',
    'refactor -z',
    'resub',
    'resub -z',
    'resub -K 6',
    'resub -K 8',
    'resub -K 10',
    'resub -K 12',
]

def generate_synthesis_recipe(recipe_length=20, seed=None):
    """
    Generate a random synthesis recipe with L transformations

    Uses the exact same ABC commands from the original OpenABC-D paper.

    Args:
        recipe_length: Number of transformation steps (default: 20)
        seed: Random seed for reproducibility

    Returns:
        List of ABC commands
    """
    if seed is not None:
        random.seed(seed)

    recipe = []
    for _ in range(recipe_length):
        cmd = random.choice(ABC_COMMANDS)
        recipe.append(cmd)

    return recipe


def write_reference_script(output_path, recipe_id, recipe, add_io_commands=False):
    """
    Write a single reference script file

    Args:
        output_path: Output file path
        recipe_id: Script ID number
        recipe: List of transformation commands
        add_io_commands: Whether to add read/write commands (False for reference scripts)
    """
    with open(output_path, 'w') as f:
        # Reference scripts don't include read/write commands
        # Those are added by automate_synthesisScriptGen.py for each design

        if add_io_commands:
            # These are placeholder lines that will be replaced by automate_synthesisScriptGen.py
            f.write("# Placeholder for library read command\n")
            f.write("# Placeholder for design read command\n")

        # Write the transformation sequence
        for cmd in recipe:
            f.write(cmd + "\n")

        if add_io_commands:
            # Placeholder for final commands
            f.write("# Placeholder for map command\n")
            f.write("# Placeholder for topo command\n")
            f.write("# Placeholder for stime command\n")
            f.write("# Placeholder for write command\n")
            f.write("# Additional placeholder lines\n")
            f.write("# Additional placeholder lines\n")
            f.write("# Additional placeholder lines\n")
            f.write("# Additional placeholder lines\n")


def generate_reference_scripts(output_dir, num_scripts=1500, recipe_length=20,
                               base_seed=42):
    """
    Generate all reference scripts using the original OpenABC-D ABC commands

    Args:
        output_dir: Directory to save scripts
        num_scripts: Number of scripts to generate (default: 1500)
        recipe_length: Length of each recipe (default: 20)
        base_seed: Base random seed
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"Generating {num_scripts} reference scripts...")
    print(f"Recipe length: {recipe_length} transformations")
    print(f"Using ORIGINAL OpenABC-D ABC command set")
    print(f"Output directory: {output_dir}")
    print()

    for i in range(num_scripts):
        # Use different seed for each script to ensure variety
        seed = base_seed + i if base_seed is not None else None

        # Generate the recipe
        recipe = generate_synthesis_recipe(
            recipe_length=recipe_length,
            seed=seed
        )

        # Write the script file
        script_path = os.path.join(output_dir, f'abc{i}.script')
        write_reference_script(script_path, i, recipe, add_io_commands=True)

        # Progress update
        if (i + 1) % 100 == 0:
            print(f"Generated {i + 1}/{num_scripts} scripts...")

    print(f"\n✓ Successfully generated {num_scripts} reference scripts!")
    print(f"  Location: {output_dir}")
    print(f"\nNext step:")
    print(f"  Run automate_synthesisScriptGen.py with --script {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate random synthesis recipe reference scripts for OpenABC-D using ORIGINAL ABC commands',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Generate 1500 scripts with default settings (uses original OpenABC-D commands)
  python generate_reference_scripts.py --output ./referenceScripts
  
  # Generate with custom settings
  python generate_reference_scripts.py --output ./referenceScripts --num 1500 --length 20 --seed 42

Commands used (EXACT match to original OpenABC-D):
  balance, rewrite, rewrite -z, refactor, refactor -z,
  resub, resub -z, resub -K 6, resub -K 8, resub -K 10, resub -K 12
        """
    )

    parser.add_argument('--output', '-o', required=True,
                       help='Output directory for reference scripts')
    parser.add_argument('--num', '-n', type=int, default=1500,
                       help='Number of scripts to generate (default: 1500)')
    parser.add_argument('--length', '-l', type=int, default=20,
                       help='Recipe length (number of transformations, default: 20)')
    parser.add_argument('--seed', '-s', type=int, default=42,
                       help='Base random seed for reproducibility (default: 42)')
    parser.add_argument('--version', action='version', version='1.0.0')

    args = parser.parse_args()

    generate_reference_scripts(
        output_dir=args.output,
        num_scripts=args.num,
        recipe_length=args.length,
        base_seed=args.seed
    )


if __name__ == '__main__':
    main()
