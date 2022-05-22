TARGET := x86_64-unknown-darwin-macho

.PHONY: build
build:
	cargo build

rt.o: rt.c
	cc -target $(TARGET) -c -o $@ $^

%.o: %.bril
	bril2json < $^ | ./target/debug/brilift -t $(TARGET)
	mv bril.o $@

%: %.o rt.o
	cc -target $(TARGET) -o $@ $^
