# Note: before running this script, make sure to do `cargo build --release` first!

import os
import subprocess
import glob

if __name__ == '__main__':

    bril_files = glob.glob("test/*.bril")

    # Create a directory for benchmark results if it doesn't exist
    os.makedirs("benchmark_results", exist_ok=True)

    for bril_file in bril_files:
        # Extract just the filename without path and extension
        filename = os.path.basename(bril_file)
        base_name = os.path.splitext(filename)[0]
        
        print(f"Benchmarking {filename}...")
        
        # Create the hyperfine command with the specific file
        hyperfine_cmd = [
            "hyperfine",
            f"--export-csv=benchmark_results/{base_name}.csv",
            "-w1",
            "-n", "flat-bril", f"turnt -e interp {bril_file}",
            "-n", "brili-ts", f"turnt -e brili {bril_file}",
            "-n", "brili-rs", f"turnt -e brilirs {bril_file}"
        ]
        
        # Run the hyperfine command
        try:
            subprocess.run(hyperfine_cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error benchmarking {filename}: {e}")

    print("All benchmarks completed!")
