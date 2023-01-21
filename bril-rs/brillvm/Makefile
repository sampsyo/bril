TESTS := ../../test/interp/core*/*.bril \
         ../../test/interp/float/*.bril \
# ../../test/interp/mem*/*.bril \
# ../../test/interp/mixed/*.bril \
# ../../test/interp/ssa*/*.bril \

BENCHMARKS := ../../benchmarks/core/*.bril \
			  ../../benchmarks/float/*.bril \
# ../benchmarks/mem/*.bril \
# ../benchmarks/mixed/*.bril \

clean:
	cargo clean
	rm rt.bc
	rm tmp.ll

.PHONY: rt
rt:
	cargo rustc --bin rt --release -- --emit=llvm-bc && mv target/release/deps/rt-*.bc rt.bc

example:
	bril2json < ../../test/interp/core/main-args.bril | cargo run | lli - 1 2

.PHONY: test
test:
	turnt -e brillvm $(TESTS)

.PHONY: benchmark
benchmark:
	turnt -e brillvm $(BENCHMARKS)