# Brilift only supports core Bril for now, so we select those tests &
# benchmarks.
TESTS := ../test/interp/core/*.bril ../test/interp/float/*.bril ../test/interp/mem/*.bril ../test/interp/mixed/*[^r].bril # A hack to exclude store-char.bril by excluding any file ending in r.bril
BENCHMARKS := ../benchmarks/core/*.bril ../benchmarks/float/*.bril ../benchmarks/mem/*.bril ../benchmarks/mixed/*.bril

CFLAGS := $(if $(TARGET),-target $(TARGET))
BRILFLAGS := $(if $(TARGET),-t $(TARGET))

.PHONY: build
build:
	cargo build

.PHONY: install
install:
	cargo install --path .

release:
	cargo build --release

.PHONY: test
test: rt.o release
	turnt -e brilift-aot -e brilift-jit $(TURNTARGS) $(TESTS)

.PHONY: benchmark
benchmark: rt.o release
	turnt -e brilift-aot -e brilift-jit $(TURNTARGS) $(BENCHMARKS)

rt.o: rt.c
	cc $(CFLAGS) -c -o $@ $^

%.o: %.bril
	bril2json < $^ | cargo run -- $(BRILFLAGS) -o $@

%: %.o rt.o
	cc $(CFLAGS) -o $@ $^
