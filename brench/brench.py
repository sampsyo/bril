import click
import tomlkit
import subprocess
import re
import csv
import sys
import os

ARGS_RE = r'ARGS: (.*)'
TIMEOUT = 5


def run_pipe(cmds, input):
    last_proc = None
    for i, cmd in enumerate(cmds):
        proc = subprocess.Popen(
            cmd,
            shell=True,
            text=True,
            stdin=last_proc.stdout if last_proc else subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
                   if i == len(cmds) - 1 else subprocess.DEVNULL,
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
@click.option('-c', '--config', 'config_path',
              nargs=1, type=click.Path(exists=True))
@click.argument('file', nargs=-1, type=click.Path(exists=True))
def brench(config_path, file):
    with open(config_path) as f:
        config = tomlkit.loads(f.read())

    # CSV for collected outputs.
    writer = csv.writer(sys.stdout)
    writer.writerow(['benchmark', 'run', 'result'])

    for fn in file:
        first_out = None
        for name, run in config['runs'].items():
            # Actually run the benchmark.
            try:
                stdout, stderr = run_bench(run['pipeline'], fn)
            except subprocess.TimeoutExpired:
                stdout, stderr = '', ''
                timeout = True
            else:
                timeout = False

            # Check correctness.
            if first_out is None:
                first_out = stdout
                correct = True
            else:
                correct = stdout == first_out

            # Extract the figure of merit.
            result = get_result([stdout, stderr], config['extract'])

            # Report the result.
            bench, _ = os.path.splitext(os.path.basename(fn))
            writer.writerow([
                bench,
                name,
                'timeout' if timeout else
                (result or '') if correct else 'incorrect',
            ])


if __name__ == '__main__':
    brench()
