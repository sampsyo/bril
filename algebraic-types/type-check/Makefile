default: install

build:
	dune build

install: build
	dune install

clean:
	dune clean
	ocamlfind remove briltc

fresh: clean default

.PHONY: default build clean fresh
