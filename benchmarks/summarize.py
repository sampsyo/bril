#!/usr/bin/env python3
import json
import sys
import os
import csv

MODES = ['brili', 'brilift']


def get_results(bench_files):
    for fn in bench_files:
        with open(fn) as f:
            bench_data = json.load(f)

        bench, _ = os.path.basename(fn).split('.', 1)
        for res in bench_data["results"]:
            for mode in MODES:
                if mode in res['command']:
                    break
            else:
                assert False, "unknown benchmark command"

            yield bench, mode, res


def summarize(bench_files):
    writer = csv.DictWriter(sys.stdout, ['bench', 'mode', 'mean', 'stddev'])
    writer.writeheader()
    for bench, mode, res in get_results(bench_files):
        writer.writerow({
            'bench': bench,
            'mode': mode,
            'mean': res['mean'],
            'stddev': res['stddev']
        })


if __name__ == '__main__':
    summarize(sys.argv[1:])
