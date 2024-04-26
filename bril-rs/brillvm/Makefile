TESTS := ../../test/interp/core/*.bril \
         ../../test/interp/float/*.bril \
		 ../../test/interp/ssa/*.bril \
		 ../../test/interp/mem/*.bril \
		 ../../test/interp/mixed/*[^r].bril # A hack to exclude store-char.bril by excluding any file ending in r.bril

BENCHMARKS := ../../benchmarks/core/*.bril \
			  ../../benchmarks/float/*.bril \
			  ../../benchmarks/mem/*.bril \
			  ../../benchmarks/mixed/*.bril

build:
	cargo build

clean:
	cargo clean
	rm rt.bc
	rm tmp.ll

.PHONY: rt
rt:
	cargo rustc --bin rt --release -- --emit=llvm-bc && mv target/release/deps/rt-*.bc rt.bc

example:
# bril2json < ../../benchmarks/mem/sieve.bril | cargo run -- -i 100
	bril2json < ../../benchmarks/mixed/cholesky.bril | cargo run -- -i

.PHONY: test
test: build rt
	turnt -e brillvm $(TESTS)

.PHONY: benchmark
benchmark: build rt
	turnt -e brillvm $(BENCHMARKS)