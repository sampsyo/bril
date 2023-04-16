# Brillvm

## Runtime

You must have a linkable runtime library available in the llvm bc format. You can get this by calling `make rt`

## TroubleShooting

### Missing LLVM

```shell
error: No suitable version of LLVM was found system-wide or pointed
              to by LLVM_SYS_150_PREFIX.
```

This tool relies on the `llvm-sys` crate finding the correct version of LLVM to use. Often, this will fail in which case you will need to provide the appropriate path as an environment variable. For example:

```shell
LLVM_SYS_150_PREFIX="/opt/homebrew/Cellar/llvm/15.0.7_1/"
```