TESTS := ../test/interp*/core*/*.bril \
../test/interp*/float/*.bril \
../test/interp*/mem*/*.bril \
../test/interp*/mixed/*.bril \
../test/interp*/ssa*/*.bril \

#../test/spec*/*.bril \

CHECK := ../test/check/*.bril

BENCHMARKS := ../benchmarks/*.bril

.PHONY: install
install:
	cargo install --path .

.PHONY: test
test:
	turnt -c ../turnt_brilirs.toml $(TESTS)
	turnt -c turnt_brilirs.toml $(CHECK)

.PHONY: benchmark
benchmark:
	turnt -c turnt_brilirs.toml $(BENCHMARKS)

.PHONY: release
release:
	RUSTFLAGS="-C target-cpu=native" cargo build --release

.PHONY: compare
compare: release
	./benchmark.sh
	#hyperfine --export-markdown results.md --warmup 5 \
	-L interp brili,./target/release/brilirs \
	"bril2json < ../benchmarks/check-primes.bril | {interp} -p 50"

# This is primarily used for running examples and debuging a bril program
.PHONY: example
example:
	bril2json < ../benchmarks/sqrt.bril | cargo run