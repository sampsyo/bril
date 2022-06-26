#!/bin/sh
set -e

# Set this if you need to use a target other than the native platform---for
# example, to run under Rosetta on macOS with Apple Silicon.
# TARGET=x86_64-unknown-darwin-macho

HERE=`dirname $0`

if [ -n "$TARGET" ]; then
    CFLAGS="-target $TARGET"
    BFLAGS="-t $TARGET"
fi

RUST_BACKTRACE=1 cargo run --manifest-path $HERE/Cargo.toml --quiet -- $BFLAGS -o tmp_run.o
cc $CFLAGS -o tmp_run tmp_run.o $HERE/rt.o
./tmp_run $@
rm tmp_run.o tmp_run
