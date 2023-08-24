# Brillvm

## Runtime

You must have a linkable runtime library available in the LLVM bc format. You can get this by calling `make rt`

## Usage

Example: `bril2json < ../../benchmarks/mem/sieve.bril | cargo run -- -i 100`

A couple of notes about flags:

- `-i` enables the `lli` interpreter to interpret the llvm code. Leave this off if you just want the resulting `.ll` file.
- `-f <file>` can be used to provide the Bril JSON file if not being passed via stdin.
- `-r <file>` can be used to provide a path to the runtime library `rt.bc` if it is not contained in the same directory.
- `<args>` All other arguments should be passable as normal if in `-i` mode.

Valid Bril programs are assumed as input with no attempt at error handling. Each compiler `.ll` file is verified before being emitted. If the line `llvm_prog.verify().unwrap();` raises an error then open an issue with your Bril program!

## TroubleShooting

### Floating Point values

Be careful which execution engine you are using which may enable some kind of fast math mode behind your back. This can cause some floating point values to be off by a small epsilon.

### Missing LLVM

```shell
error: No suitable version of LLVM was found system-wide or pointed
              to by LLVM_SYS_150_PREFIX.
```

This tool relies on the `llvm-sys` crate to find the correct version of LLVM to use. Often, this will fail in which case you will need to provide the appropriate path as an environment variable. For example:

```shell
LLVM_SYS_150_PREFIX="/opt/homebrew/Cellar/llvm/15.0.7_1/"
```
