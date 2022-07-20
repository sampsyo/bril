#!/bin/bash

# https://doc.rust-lang.org/rustc/profile-guided-optimization.html

set -e

# You need the path to where ever cargo has the llvm-profdata command
# rustup component add llvm-tools-preview
export PATH="$PATH:~/.rustup/toolchains/stable-aarch64-apple-darwin/lib/rustlib/aarch64-apple-darwin/bin/"
# export PATH="$PATH:~/.rustup/toolchains/stable-x86_64-apple-darwin/lib/rustlib/x86_64-apple-darwin/bin"

# STEP 0: Make sure there is no left-over profiling data from previous runs
rm -rf /tmp/pgo-data || true
rm merged.profdata || true

# STEP 1: Build the instrumented binaries
RUSTFLAGS="-Cprofile-generate=/tmp/pgo-data" \
  cargo build --release

# STEP 2: Run the instrumented binaries with some typical data
./long_benchmark.sh

# STEP 3: Merge the `.profraw` files into a `.profdata` file
touch merged.profdata
llvm-profdata merge -o merged.profdata /tmp/pgo-data

# STEP 4: Use the `.profdata` file for guiding optimizations
make pgo