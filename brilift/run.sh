#!/bin/sh
set -e

TARGET=x86_64-unknown-darwin-macho

bril2json < $1 | RUST_BACKTRACE=1 cargo run -- -t $TARGET -o tmp_run.o
cc -target $TARGET -o tmp_run tmp_run.o rt.o
./tmp_run
rm tmp_run.o tmp_run
