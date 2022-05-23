TARGET := x86_64-unknown-darwin-macho
RUST_TARGET := x86_64-apple-darwin

.PHONY: build
build:
	cargo build

rt.o: rt.rs
	rustc -C panic=abort --crate-type=staticlib --emit=obj \
		--edition=2021 --target=$(RUST_TARGET) $^

%.o: %.bril
	bril2json < $^ | ./target/debug/brilift -t $(TARGET) -o $@

%: %.o rt.o
	cc -target $(TARGET) -o $@ $^
