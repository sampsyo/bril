TESTS :=  ../test/interp/*.bril \
	../test/mem/*.bril \
	../test/fail/*.bril

BENCHMARKS := ../benchmarks/*.bril

.PHONY: test
test:
	turnt -c turnt_brilirs.toml $(TESTS)

.PHONY: benchmark
benchmark:
	turnt -v -c turnt_brilirs.toml $(BENCHMARKS)

# This is primarily used for running examples and debuging a bril program
.PHONY: example
example:
	bril2json < ../benchmarks/mat-mul.bril | cargo run -- -p 50 109658