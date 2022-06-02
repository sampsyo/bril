TESTS := test/parse/*.bril \
	test/print/*.json \
	test/ts*/*.ts \
	test/check/*.bril \
	test/interp*/core*/*.bril \
	test/interp*/float/*.bril \
	test/interp*/mem*/*.bril \
	test/interp*/mixed/*.bril \
	test/interp*/spec*/*.bril \
	test/interp*/ssa*/*.bril \
	examples/test/*/*.bril \
	benchmarks/*.bril

CHECKS := test/parse/*.bril \
	test/interp/core/*.bril \
	test/interp/float/*.bril \
	test/interp/mixed/*.bril \
	test/interp/spec/*.bril \
	test/interp/ssa/*.bril \
	test/interp/mem/*.bril \
	examples/test/*/*.bril \
	benchmarks/*.bril

.PHONY: test
test:
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
