#!/usr/bin/env python3
import json
import sys
import os
import csv
import statistics
from collections import defaultdict

MODES = {
    'brili': 'brili ',
    'brilift-jit': 'brilift -j'
}
BASELINE = 'brili'


def get_results(bench_files):
    for fn in bench_files:
        with open(fn) as f:
            bench_data = json.load(f)

        bench, _ = os.path.basename(fn).split('.', 1)
        for res in bench_data["results"]:
            for mode, pat in MODES.items():
                if pat in res['command']:
                    break
            else:
                assert False, "unknown benchmark command"

            yield bench, mode, res


def summarize(bench_files):
    means = defaultdict(dict)

    writer = csv.DictWriter(sys.stdout, ['bench', 'mode', 'mean', 'stddev'])
    writer.writeheader()
    for bench, mode, res in get_results(bench_files):
        means[bench][mode] = res['mean']
        writer.writerow({
            'bench': bench,
            'mode': mode,
            'mean': res['mean'],
            'stddev': res['stddev']
        })

    speedups = {k: [] for k in MODES}
    for bench, modes in means.items():
        baseline = modes[BASELINE]
        print(bench, file=sys.stderr, end=': ')
        for mode in MODES:
            speedup = baseline / modes[mode]
            speedups[mode].append(speedup)
            print('{} {:.2f}x'.format(mode, speedup), file=sys.stderr, end=' ')
        print(file=sys.stderr)
    for mode, speedup_list in speedups.items():
        print('{}: {:.2f}x'.format(
            mode,
            statistics.harmonic_mean(speedup_list)
        ), file=sys.stderr)


if __name__ == '__main__':
    summarize(sys.argv[1:])
