# Creates the scatter plot in `bench_results.png`

import matplotlib.pyplot as plt
import glob
import os


if __name__ == "__main__":
    data = []

    csv_files = glob.glob("benchmark_results/*.csv")

    for csv_file in csv_files:
        with open(csv_file, "r") as file:
            lines = file.readlines()[1:]
            for line in lines:
                basename = os.path.basename(csv_file)
                benchmark = os.path.splitext(basename)[0]
                run, result = line.strip().split(",")[:2]
                if result != "timeout":
                    data.append((benchmark, run, round(float(result), 3)))

    # Sort the benchmarks by the name of the benchmark file
    benchmarks = list(sorted({benchmark for (benchmark, _, _) in data}))
    flat_results = []
    ts_results = []
    rs_results = []

    for benchmark in benchmarks:
        flat_result = [
            result
            for (bench_name, run, result) in data
            if bench_name == benchmark and run == "flat-bril"
        ]
        ts_result = [
            result
            for (bench_name, run, result) in data
            if bench_name == benchmark and run == "brili-ts"
        ]
        rs_result = [
            result
            for (bench_name, run, result) in data
            if bench_name == benchmark and run == "brili-rs"
        ]

        flat_results.append(flat_result)
        ts_results.append(ts_result)
        rs_results.append(rs_result)

    x = range(len(benchmarks))
    plt.figure(figsize=(16, 6))

    plt.scatter(x, flat_results, color="green", label="Flat-Bril", zorder=2)
    plt.scatter(x, ts_results, color="blue", label="Brili (TypeScript)", zorder=2)
    plt.scatter(x, rs_results, color="red", label="Brili (Rust)", zorder=2) 

    plt.title(
        "Mean execution time of interpreter on benchmarks (lower is better)"
    )
    plt.xticks(x, benchmarks, rotation=45, ha="right")

    plt.xlabel("Benchmarks")
    plt.ylabel("Mean execution time over 10 runs (s)")
    plt.legend()
    plt.grid(zorder=1, linestyle="--", alpha=0.6)

    plot_filename = "bench_results.png"

    plt.tight_layout()
    plt.savefig(plot_filename)
    print(f'Plot saved in {plot_filename}')
    plt.close()
    