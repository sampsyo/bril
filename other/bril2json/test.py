import sys, subprocess, json
import multiprocessing
import difflib


def init_worker(shared_failure_event, shared_oracle):
    global failure_event
    global oracle
    failure_event = shared_failure_event
    oracle = shared_oracle


def check_file(file):
    oracle_output = json.loads(subprocess.getoutput(f"{oracle} <{file}"))
    my_output = json.loads(subprocess.getoutput(f"cargo run --quiet -- {file}"))
    oracle_pretty = subprocess.run(
        "bril2txt",
        input=json.dumps(oracle_output).encode("utf-8"),
        stdout=subprocess.PIPE,
    ).stdout.decode("utf-8")
    my_pretty = subprocess.run(
        "bril2txt",
        input=json.dumps(my_output).encode("utf-8"),
        stdout=subprocess.PIPE,
    ).stdout.decode("utf-8")
    if oracle_pretty == my_pretty:
        print(f"\x1b[32m{file} OK\x1b[m")
    else:
        print(f"\x1b[31m{file} ERROR\x1b[m")
        failure_event.set()

        red = lambda text: f"\033[38;2;255;0;0m{text}\033[m"
        green = lambda text: f"\033[38;2;0;255;0m{text}\033[m"
        blue = lambda text: f"\033[38;2;0;0;255m{text}\033[m"
        white = lambda text: f"\033[38;2;255;255;255m{text}\033[m"
        gray = lambda text: f"\033[38;2;128;128;128m{text}\033[m"

        diff = difflib.ndiff(oracle_pretty.splitlines(), my_pretty.splitlines())
        print("--- DIFF ---")
        for line in diff:
            if line.startswith("+"):
                print(green(line))
            elif line.startswith("-"):
                print(red(line))
            elif line.startswith("^"):
                print(blue(line))
            elif line.startswith("?"):
                print(gray(line))
            else:
                print(white(line))


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: python3 test.py <oracle> <file>...")
        sys.exit(1)
    oracle = sys.argv[1]
    files = sys.argv[2:]

    with multiprocessing.Manager() as manager:
        failure_event = manager.Event()

        with multiprocessing.Pool(
            multiprocessing.cpu_count(),
            initializer=init_worker,
            initargs=(failure_event, oracle),
        ) as pool:
            pool.imap_unordered(check_file, files)
            pool.close()
            pool.join()
            if failure_event.is_set():
                print("Exiting due to errors")
                pool.terminate()
                sys.exit(1)
