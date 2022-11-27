# Brillvm

## TroubleShooting

```shell
error: No suitable version of LLVM was found system-wide or pointed
              to by LLVM_SYS_140_PREFIX.
```

This tool relies on the `llvm-sys` crate finding the correct version of LLVM to use. Often, this will fail in which case you will need to provide the appropriate path as an environment variable. For example:

```shell
LLVM_SYS_140_PREFIX=/opt/homebrew/Cellar/llvm\@14/14.0.6/ cargo check
```