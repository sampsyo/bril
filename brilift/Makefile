TARGET := x86_64-unknown-darwin-macho
RUST_TARGET := x86_64-apple-darwin

# Brilift only supports core Bril for now, so we select those tests &
# benchmarks.
TESTS :=  ../test/interp/core/*.bril
BENCHMARKS := $(shell grep -L 'ptr\|float' ../benchmarks/*.bril)

foo:
	echo $(BENCHMARKS)

.PHONY: build
build:
	cargo build

.PHONY: install
install:
	cargo install --path .

.PHONY: test
test: rt.o
	turnt -e brilift-aot -e brilift-jit $(TESTS)

.PHONY: benchmark
benchmark: rt.o
	turnt -e brilift-aot -e brilift-jit $(BENCHMARKS)

rt.o: rt.c
	cc -target $(TARGET) -c -o $@ $^

%.o: %.bril
	bril2json < $^ | ./target/debug/brilift -t $(TARGET) -o $@

%: %.o rt.o
	cc -target $(TARGET) -o $@ $^
