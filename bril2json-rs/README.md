# `bril2json`

This is a simple shell utility to convert programs in Bril's textual
representation into Bril's canonical JSON representation.

## Install

```
cd other/bril2json
cargo install --path .
```

## Usage

```
Usage: bril2json [<input_path>]

converts Bril's textual representation to its canonical JSON form

Positional Arguments:
  input_path        input Bril file: omit for stdin

Options:
  --help, help      display usage information
```
