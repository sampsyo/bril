TESTS := test/parse/*.bril \
	test/print/*.json \
	test/ts*/*.ts \
	test/check/*.bril \
	test/interp*/core*/*.bril \
	test/interp*/char*/*.bril \
	test/interp*/float/*.bril \
	test/interp*/mem*/*.bril \
	test/interp*/mixed/*.bril \
	test/interp*/spec*/*.bril \
	test/interp*/ssa*/*.bril \
	examples/test/*/*.bril \
	benchmarks/core/*.bril \
	benchmarks/float/*.bril \
	benchmarks/mem/*.bril \
	benchmarks/mixed/*.bril \

CHECKS := test/parse/*.bril \
	test/interp/core/*.bril \
	test/interp/char/*.bril \
	test/interp/float/*.bril \
	test/interp/mixed/*.bril \
	test/interp/spec/*.bril \
	test/interp/ssa/*.bril \
	test/interp/mem/*.bril \
	examples/test/*/*.bril \
	benchmarks/core/*.bril \
	benchmarks/float/*.bril \
	benchmarks/mem/*.bril \
	benchmarks/mixed/*.bril \

# https://stackoverflow.com/a/25668869
EXECUTABLES = bril2json bril2txt ts2bril brili brilck

.PHONY: test
test:
	$(foreach exec,$(EXECUTABLES), $(if $(shell which $(exec)),,$(error "No $(exec) in PATH: Either refer to the documentation for their installation instructions or run a subset of the tests manually with `turnt test/interp*/**/*.bril`")))
	turnt $(TURNTARGS) $(TESTS)

.PHONY: check
check:
	for fn in $(CHECKS) ; do \
		bril2json -p < $$fn | brilck $$fn || failed=1 ; \
	done ; \
	exit $$failed

.PHONY: book
book:
	rm -rf book
	mdbook build

.PHONY: ts
ts:
	cd bril-ts ; \
	yarn ; \
	yarn build

.PHONY: deploy
RSYNCARGS := --compress --recursive --checksum --itemize-changes \
	--delete -e ssh --perms --chmod=Du=rwx,Dgo=rx,Fu=rw,Fog=r \
	--exclude=.DS_Store
DEST := courses:coursewww/capra.cs.cornell.edu/htdocs/bril
deploy: book
	rsync $(RSYNCARGS) ./book/ $(DEST)
