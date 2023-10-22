TESTS := ../test/check/*.bril \
../test/interp*/core*/*.bril \
../test/interp*/float/*.bril \
../test/interp*/mem*/*.bril \
../test/interp*/char*/*.bril \
../test/interp*/mixed/*.bril \
../test/interp*/ssa*/*.bril \

#../test/spec*/*.bril \

BENCHMARKS := ../benchmarks/core/*.bril \
../benchmarks/float/*.bril \
../benchmarks/mem/*.bril \
../benchmarks/mixed/*.bril \

LONGBENCHMARKS := ../benchmarks/long/*.bril \
../benchmarks/core/ackermann.bril \
../benchmarks/mem/eight-queens.bril \
../benchmarks/mem/mat-mul.bril \

.PHONY: install
install:
	RUSTFLAGS="-C target-cpu=native" cargo install --path .

.PHONY: test
test:
	turnt -e brilirs $(TESTS)

.PHONY: benchmark
benchmark:
	turnt -e brilirs $(BENCHMARKS)

.PHONY: release
release:
	RUSTFLAGS="-C target-cpu=native" cargo build --release

#https://doc.rust-lang.org/rustc/profile-guided-optimization.html
.PHONY: pgo
pgo:
	RUSTFLAGS="-C target-cpu=native -Cprofile-use=$(shell pwd)/merged.profdata" cargo build --release

.PHONY: pgo-install
pgo-install:
	RUSTFLAGS="-Cprofile-use=$(shell pwd)/merged.profdata" cargo install --path .

# `make bench` or `make longbench` to get performance metrics
# `make bench_rust.csv` to get the harmonic mean of each benchmark
# `make plot` to get a plot of your evaluation

.PHONY: bench
bench:
	turnt -c ../brilirs/turnt.toml -e bench --save $(BENCHMARKS)

.PHONY: longbench
longbench:
	turnt -c ../brilirs/turnt.toml -e bench --save $(LONGBENCHMARKS)

bench_rust.csv: $(wildcard ../benchmarks/**/*.bench_rust.json)
	python3 ../benchmarks/summarize.py $^ > $@

plot: plot.svg
%.svg: %.vl.json bench_rust.csv
	npx -p vega -p vega-lite vl2svg $*.vl.json > $@

# This is primarily used for running examples and debuging a bril program
.PHONY: example
example:
	bril2json < ../benchmarks/float/sqrt.bril | cargo run
