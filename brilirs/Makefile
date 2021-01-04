TESTS :=  ../test/interp/*.bril \
	../test/mem/*.bril \
	../test/fail/*.bril \
	../benchmarks/*.bril

.PHONY: test
test:
	turnt -c turnt_brilirs.toml $(TESTS)

.PHONY: example
example:
	bril2json < ../test/interp/call-with-args.bril | cargo run --