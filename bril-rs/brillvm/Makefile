TESTS := ../../test/interp/core*/*.bril \
# ../../test/interp/float/*.bril \
# ../../test/interp/mem*/*.bril \
# ../../test/interp/mixed/*.bril \
# ../../test/interp/ssa*/*.bril \

rt:
#	rustc +1.64 bin/rt.rs --emit=llvm-bc --crate-type lib -L target/release/deps
#	cargo +1.64 rustc --bin rt --release -- --emit=llvm-ir
#   clang -emit-llvm -o rc.bc -c rt.c

example:
	bril2json < ../../test/interp/core/main-bool-args.bril | cargo run

.PHONY: test
test:
	turnt -v -e brillvm $(TESTS)