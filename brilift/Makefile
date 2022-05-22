RUST_TARGET := x86_64-apple-darwin
LONG_TARGET := x86_64-unknown-darwin-macho

.PHONY: build
build:
	cargo build
	cd ../bril-crt ; cargo build --target $(RUST_TARGET)

%.o: %.bril
	bril2json < $^ | ./target/debug/brilift -t $(LONG_TARGET)
	mv bril.o $@

%: %.o
	cc -target $(LONG_TARGET) -o $@ \
		$^ ../bril-crt/target/$(RUST_TARGET)/debug/librt.a
