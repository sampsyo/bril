#!/usr/bin/env python3
import json
import sys
import os
import csv
import statistics
import re
from collections import defaultdict

MODES = {
    'brili': r'\bbrili\b',
    'brilirs': r'\bbrilirs\b',
    'brilift-jit': r'\bbrilift -j',
    'brilift-aot': r'^\./[^/]+ '
}
BASELINE = 'brili'


def get_results(bench_files):
    for fn in bench_files:
        with open(fn) as f:
            bench_data = json.load(f)

        bench, _ = os.path.basename(fn).split('.', 1)
        for res in bench_data["results"]:
            for mode, pat in MODES.items():
                if re.search(pat, res['command']):
                    break
            else:
                assert False, "unknown benchmark command"

            yield bench, mode, res


def summarize(bench_files):
    means = defaultdict(dict)
    results = list(get_results(bench_files))
    for bench, mode, res in results:
        means[bench][mode] = res['mean']

    writer = csv.DictWriter(
        sys.stdout,
        ['bench', 'mode', 'mean', 'stddev', 'speedup'],
    )
    writer.writeheader()
    speedups = {k: [] for k in MODES}
    for bench, mode, res in results:
        speedup = means[bench][BASELINE] / res['mean']
        print('{} {} {:.2f}x'.format(bench, mode, speedup), file=sys.stderr)
        speedups[mode].append(speedup)

        writer.writerow({
            'bench': bench,
            'mode': mode,
            'mean': res['mean'],
            'stddev': res['stddev'],
            'speedup': speedup,
        })

    # Filter out modes which don't have data, and thus no speedups.
    speedups = {k: v for k, v in speedups.items() if v}

    for mode, speedup_list in speedups.items():
        print('{}: {:.2f}x'.format(
            mode,
            statistics.harmonic_mean(speedup_list)
        ), file=sys.stderr)


if __name__ == '__main__':
    summarize(sys.argv[1:])
