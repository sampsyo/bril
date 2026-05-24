#!/usr/bin/env bash
# tools/run_perf.sh — parallel-vs-sequential performance comparison
set -euo pipefail

# List of .bril benchmarks to run
benchmarks=(
  benchmarks/concurrency/perf/perf_par_sum_100k.bril
  benchmarks/concurrency/perf/perf_matmul_split.bril
  benchmarks/concurrency/perf/perf_big_par_sum_10M.bril
)

for b in "${benchmarks[@]}"; do
  base=$(basename "$b" .bril)
  json="${base}.json"
  perf_json="perf_${base}.json"

  echo "=== Benchmark: $base ==="

  # 1) Emit Bril JSON
  bril2json < "$b" > "$json"

  # 2) Run hyperfine: sequential (--no-workers) vs concurrent
  hyperfine \
    --warmup 3 \
    --export-json "$perf_json" \
    --show-output \
    "deno run -A brili.ts --no-workers < $json > /dev/null" \
    "deno run -A brili.ts               < $json > /dev/null"

  # 3) Clean up the intermediate JSON file
  rm -f "$json"
done

echo "Perf JSON results written to: perf_*.json"