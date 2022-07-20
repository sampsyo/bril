# Brilirs

`brilirs` is a fast interpreter written in Rust.

It is available as both a command-line interface and a rust library.

## Command-line interface

The main use case of `brilirs` is to be a faster `brili`. Using `cargo`; run `cargo install --path .` and make sure `$HOME/.cargo/bin` is on your path. Run `brilirs --help` for all of the supported flags.

## Rust interface

`brilirs` can also be used in your rust code which may be advantageous. Add `brilirs` to your `Cargo.toml` with:

```toml
[dependencies.brilirs]
version      = "0.1.0"
path         = "../brilirs"
```

Check out `cargo doc --open` for exposed functions. One possible workflow is that you have a `bril_rs::Program` called `program` and a list of `args` that you want to run through the interpreter.

```rust
let bbprog = BBProgram::new(program)?;
check::type_check(&bbprog)?;
interp::execute_main(&bbprog, std::io::stdout(), &args, false, std::io::stderr())?;
```

You can also use a `bril_rs::AbstractProgram` called `abstract_program` by converting it into a `bril_rs::Program` using `abstract_program.try_into()?`.

## PGO

You can get a modest performance benefit(~5-7%) by using LLVM's profile guided optimization. See `pgo.sh` and `make pgo`/`make pgo-install` for more details.

## Contributing

Issues and PRs are welcome. For pull requests, make sure to run the test harness with `make test` and `make benchmark`. There is also `.github/workflows/rust.yaml` which will format your code and check that it is conforming with clippy.
