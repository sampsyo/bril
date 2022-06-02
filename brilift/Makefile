TARGET := x86_64-unknown-darwin-macho
RUST_TARGET := x86_64-apple-darwin

TESTS :=  ../test/interp/core/*.bril
BENCHMARKS := ../benchmarks/*.bril

.PHONY: build
build:
	cargo build

.PHONY: install
install:
	cargo install --path .

.PHONY: test
test: rt.o
	turnt -c turnt_brilift_aot.toml $(TESTS)
	turnt -c turnt_brilift_jit.toml $(TESTS)

.PHONY: benchmark
benchmark: rt.o
	turnt -c turnt_brilift_aot.toml $(BENCHMARKS)
	turnt -c turnt_brilift_jit.toml $(BENCHMARKS)

rt.o: rt.c
	cc -target $(TARGET) -c -o $@ $^

%.o: %.bril
	bril2json < $^ | ./target/debug/brilift -t $(TARGET) -o $@

%: %.o rt.o
	cc -target $(TARGET) -o $@ $^
