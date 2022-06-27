#!/bin/sh
set -e

# This script is a one-stop shop for building and executing programs with
# Brilift's AOT compiler (making it a drop-in replacement for the JIT compiler
# or an interpreter). Set $TARGET if you need to use a target other than the
# native platform. For example, to run under Rosetta on macOS with Apple
# Silicon, use:
#
#     export TARGET=x86_64-unknown-darwin-macho

HERE=`dirname $0`

if [ -n "$TARGET" ]; then
    CFLAGS="-target $TARGET"
    BFLAGS="-t $TARGET"
fi

tmpdir=`mktemp -d`

RUST_BACKTRACE=1 cargo run --manifest-path $HERE/Cargo.toml --quiet -- $BFLAGS -o $tmpdir/bril.o
cc $CFLAGS -o $tmpdir/bril $tmpdir/bril.o $HERE/rt.o
$tmpdir/bril $@

rm -r $tmpdir
