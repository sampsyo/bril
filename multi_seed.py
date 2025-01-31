import subprocess
import sys

LIMIT = 100_000


def run(prog, n, seed):
    output = subprocess.run([f"bril2json < {BRIL_PROG} | brili {n} {seed}"], 
                            shell=True, 
                            encoding='utf-8', 
                            stdout=subprocess.PIPE)
    return [line for line in output.stdout.split('\n') if len(line) > 0]


if __name__ == '__main__':
    if len(sys.argv) != 5:
        print(f"Usage: ./{sys.argv[0]} <bril_prog> <dimension> <min_seed> <max_seed>")
        exit(1)

    BRIL_PROG = sys.argv[1]
    DIM       = sys.argv[2]
    MIN_SEED  = int(sys.argv[3])
    MAX_SEED  = int(sys.argv[4])

    long_walk = 0
    long_seed = None
    returned  = 0
    for seed in range(MIN_SEED, MAX_SEED):
        print(f"seed = {seed} -> walk_len = ", end='')
        walk     = run(BRIL_PROG, DIM, seed)
        walk_len = sum(1 if x == '-' else 0 for x in walk)
        if walk_len != LIMIT:
            returned += 1
        if long_walk < walk_len and walk_len != LIMIT:
            long_walk = walk_len
            long_seed = seed
        print(walk_len)
   

    print(f"longest walk length (that returned to start) = {long_walk}")
    print(f"associated seed                              = {long_seed}")
    print(f"{returned} out of {MAX_SEED - MIN_SEED} returned to (0, ..., 0)")
