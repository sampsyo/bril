import click
import tomlkit
import subprocess


def run_pipe(cmds, input):
    last_proc = None
    for cmd in cmds:
        proc = subprocess.Popen(
            cmd,
            shell=True,
            text=True,
            stdin=last_proc.stdout if last_proc else subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
        if not last_proc:
            first_proc = proc
        last_proc = proc

    first_proc.stdin.write(input)
    stdout, _ = last_proc.communicate()
    return stdout


@click.command()
@click.option('-c', '--config', 'config_path',
              nargs=1, type=click.Path(exists=True))
@click.argument('file', nargs=-1, type=click.Path(exists=True))
def brench(config_path, file):
    with open(config_path) as f:
        config = tomlkit.loads(f.read())

    for name, run in config['runs'].items():
        cmds = [
            c.format(args='5')
            for c in run['pipeline']
        ]
        print(cmds)
        for fn in file:
            with open(fn) as f:
                in_data = f.read()
            out_data = run_pipe(cmds, in_data)
            print(out_data)


if __name__ == '__main__':
    brench()
