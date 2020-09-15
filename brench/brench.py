import click
import tomlkit
import subprocess
import re
import csv
import sys
import os

ARGS_RE = r'ARGS: (.*)'


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
    return last_proc.communicate()


def run_bench(pipeline, fn, extract_re):
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
    stdout, stderr = run_pipe(cmds, in_data)

    # Look for results.
    for out in stdout, stderr:
        match = re.search(extract_re, out)
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
        for name, run in config['runs'].items():
            result = run_bench(run['pipeline'], fn, config['extract'])
            bench, _ = os.path.splitext(os.path.basename(fn))
            writer.writerow([bench, name, result or ''])


if __name__ == '__main__':
    brench()
