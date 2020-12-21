TESTS := test/parse/*.bril \
	test/print/*.json \
	test/interp*/*.bril \
	test/ts*/*.ts \
	test/mem/*.bril \
	test/fail/*.bril \
	examples/test/*/*.bril \
	benchmarks/*.bril

.PHONY: test
test:
	turnt $(TURNTARGS) $(TESTS)

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
