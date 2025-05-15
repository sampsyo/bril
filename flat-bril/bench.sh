#!/bin/bash

# Create a hyperfine command with all benchmarks
CMD="hyperfine -w3 --shell=none --show-output --export-markdown json_roundtrip_bench.md"

benchmarks=("bitshift" "call" "catalan" "euclid" "main-args" "montgomery" "nop" "perfect" "reverse" "rot13")

for file in test/*.json; do
  # Extract just the benchmark name (without path or extension)
  name=$(basename "$file" .json)

  for benchmark in "${benchmarks[@]}"; do
    if [ "$name" = "$benchmark" ]; then
    # Add to hyperfine command
    CMD+=" --command-name \"$name\" \"./target/release/flat-bril --json --filename $file\""
    break 
    fi 
  done
done

# Execute the command
eval "$CMD"
