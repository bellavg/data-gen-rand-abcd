import argparse
import os

# Configuration
designs = [
    "128",
    "256",
    "512",
    "1024",
    "2048",
    "4096",
    "8192",
    "16384",
    "ac97_ctrl",
    "aes",
    "aes_secworks",
    "aes_xcrypt",
    "apex1",
    "bc0",
    "bp_be",
    "c1355",
    "c5315",
    "c6288",
    "c7552",
    "dalu",
    "des3_area",
    "dft",
    "div",
    "dynamic_node",
    "ethernet",
    "fir",
    "fpu",
    "hyp",
    "i10",
    "i2c",
    "idft",
    "iir",
    "jpeg",
    "k2",
    "log2",
    "mainpla",
    "max",
    "mem_ctrl",
    "multiplier",
    "pci",
    "picosoc",
    "sasc",
    "sha256",
    "simple_spi",
    "sin",
    "spi",
    "sqrt",
    "square",
    "ss_pcm",
    "tinyRocket",
    "tv80",
    "usb_phy",
    "vga_lcd",
    "wb_conmax",
    "wb_dma",
]
numRecipes = 200


def generate_all_scripts(home_dir):
    # Setup Paths based on your provided data/ structure
    designs_dir = os.path.join(home_dir, "data", "designs")
    ref_scripts_dir = os.path.join(home_dir, "data", "abc_scripts", "reference_scripts")
    syn_scripts_dir = os.path.join(home_dir, "data", "abc_scripts", "synthesis_scripts")

    for des in designs:
        print(f"Generating scripts for design: {des}")

        # 1. Prepare Folders
        des_syn_dir = os.path.join(syn_scripts_dir, des)
        tier0_dir = os.path.join(designs_dir, des, "tier0")
        log_dir = os.path.join(
            designs_dir, des, "design_metadata", "raw_logs", "synthesis_logs"
        )

        os.makedirs(des_syn_dir, exist_ok=True)
        os.makedirs(tier0_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        # 2. Create a "Runner" shell script for this specific design
        # This is the .sh that will eventually call ABC for all 200 recipes
        master_sh_path = os.path.join(des_syn_dir, f"run_synthesis_{des}.sh")

        with open(master_sh_path, "w") as master_sh:
            master_sh.write("#!/bin/bash\n\n")

            for i in range(numRecipes):
                ref_path = os.path.join(ref_scripts_dir, f"abc{i}.script")
                out_script_path = os.path.join(des_syn_dir, f"abc{i}.script")

                if not os.path.exists(ref_path):
                    continue

                # --- Step A: Read and clean the reference ABC commands ---
                with open(ref_path, "r") as f:
                    # Strip existing IO commands so we can insert our own pathing logic
                    cmds = [
                        line.strip()
                        for line in f
                        if line.strip()
                        and not line.startswith(("read", "write", "print", "strash"))
                    ]

                # --- Step B: Write the design-specific ABC script ---
                # Use __SCRATCH__ placeholder for the design dir and __SCRIPTS__ for
                # the scripts dir. These are replaced with real scratch paths at runtime
                # by 3_run_synthesis.sh before execution.
                with open(out_script_path, "w") as f:
                    f.write(f"read_aiger __SCRATCH__/tier0/{des}_synX_step0.aig\n")
                    f.write("strash\n")

                    for idx, cmd in enumerate(cmds):
                        step_num = idx + 1
                        f.write(f"{cmd}\n")
                        f.write(f"write_aiger __SCRATCH__/tier0/{des}_syn{i}_step{step_num}.aig\n")

                # --- Step C: Add command to the design's master .sh file ---
                scratch_log = f"__SCRATCH__/design_metadata/raw_logs/synthesis_logs/log_{des}_syn{i}.log"
                scratch_script = f"__SCRIPTS__/abc{i}.script"
                master_sh.write(f"abc -f {scratch_script} > {scratch_log} 2>&1\n")

        # Make the .sh executable
        os.chmod(master_sh_path, 0o755)


def main():
    parser = argparse.ArgumentParser(
        description="Generate ABC synthesis recipes for Tier-0"
    )
    parser.add_argument("--home", required=True, help="Root path of the project")
    args = parser.parse_args()

    home_path = os.path.abspath(args.home)
    generate_all_scripts(home_path)


if __name__ == "__main__":
    main()
