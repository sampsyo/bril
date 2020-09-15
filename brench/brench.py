import click
import tomlkit
import subprocess
import re
import csv
import sys
import os
from concurrent import futures

ARGS_RE = r'ARGS: (.*)'
TIMEOUT = 5


def run_pipe(cmds, input):
    last_proc = None
    for i, cmd in enumerate(cmds):
        last = i == len(cmds) - 1
        proc = subprocess.Popen(
            cmd,
            shell=True,
            text=True,
            stdin=last_proc.stdout if last_proc else subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE if last else subprocess.DEVNULL,
        )
        if not last_proc:
            first_proc = proc
        last_proc = proc

    # Send stdin and collect stdout.
    first_proc.stdin.write(input)
    first_proc.stdin.close()
    return last_proc.communicate(timeout=TIMEOUT)


def run_bench(pipeline, fn):
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
    return run_pipe(cmds, in_data)


def get_result(strings, extract_re):
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
    with open(config_path) as f:
        config = tomlkit.loads(f.read())

    with futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        # Submit jobs.
        futs = {}
        for fn in files:
            for name, run in config['runs'].items():
                futs[(fn, name)] = pool.submit(run_bench, run['pipeline'], fn)

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
                if not result:
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
