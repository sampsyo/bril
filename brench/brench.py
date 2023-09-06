"""Simple comparative benchmark runner.
"""

import click
import tomlkit
import subprocess
import re
import csv
import sys
import os
from concurrent import futures
import glob

__version__ = '1.0.0'

ARGS_RE = r'ARGS: (.*)'


def run_pipe(cmds, input, timeout):
    """Execute a pipeline of shell commands.

    Send the given input (text) string into the first command, then pipe
    the output of each command into the next command in the sequence.
    Collect and return the stdout and stderr from the final command.
    """
    procs = []
    for cmd in cmds:
        last = len(procs) == len(cmds) - 1
        proc = subprocess.Popen(
            cmd,
            shell=True,
            text=True,
            stdin=procs[-1].stdout if procs else subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE if last else subprocess.DEVNULL,
        )
        procs.append(proc)

    try:
        # Send stdin and collect stdout.
        procs[0].stdin.write(input)
        procs[0].stdin.close()
        return procs[-1].communicate(timeout=timeout)
    finally:
        for proc in procs:
            proc.kill()


def run_bench(pipeline, fn, timeout):
    """Run a single benchmark pipeline.
    """
    # Load the benchmark.
    with open(fn) as f:
        in_data = f.read()

    # Extract arguments.
    match = re.search(ARGS_RE, in_data)
    args = match.group(1) if match else ''

    # Run pipeline.
    cmds = [
        c.format(args=args)
        for c in pipeline
    ]
    return run_pipe(cmds, in_data, timeout)


def get_result(strings, extract_re):
    """Extract a group from a regular expression in any of the strings.
    """
    for s in strings:
        match = re.search(extract_re, s)
        if match:
            return match.group(1)
    return None


@click.command()
@click.option('-j', '--jobs', default=None, type=int,
              help='parallel threads to use (default: suitable for machine)')
@click.argument('config_path', metavar='CONFIG', type=click.Path(exists=True))
@click.argument('files', nargs=-1, type=click.Path(exists=True))
def brench(config_path, files, jobs):
    """Run a batch of benchmarks and emit a CSV of results.
    """
    with open(config_path) as f:
        config = tomlkit.loads(f.read())

    # Use configured file list, if none is specified via the CLI.
    if not files and 'benchmarks' in config:
        files = glob.glob(config['benchmarks'], recursive=True)

    timeout = config.get('timeout', 5)

    with futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        # Submit jobs.
        futs = {}
        for fn in files:
            for name, run in config['runs'].items():
                futs[(fn, name)] = pool.submit(run_bench, run['pipeline'], fn,
                                               timeout)

        # Collect results and print CSV.
        writer = csv.writer(sys.stdout)
        writer.writerow(['benchmark', 'run', 'result'])
        for fn in files:
            first_out = None
            for name in config['runs']:
                try:
                    stdout, stderr = futs[(fn, name)].result()
                except subprocess.TimeoutExpired:
                    stdout, stderr = '', ''
                    status = 'timeout'
                else:
                    status = None

                # Check correctness.
                if first_out is None:
                    first_out = stdout
                elif stdout != first_out and not status:
                    status = 'incorrect'

                # Extract the figure of merit.
                result = get_result([stdout, stderr], config['extract'])
                if not result and not status:
                    status = 'missing'

                # Report the result.
                bench, _ = os.path.splitext(os.path.basename(fn))
                writer.writerow([
                    bench,
                    name,
                    status if status else result,
                ])


if __name__ == '__main__':
    brench()
