TESTS := test/parse/*.bril \
	test/print/*.json \
	test/interp/*.bril \
	test/ts/*.ts

EXAMPLE_TESTS :=  examples/*_test/*.bril 

.PHONY: test
test:
	turnt $(TESTS)

.PHONY: test_examples
test_examples:
	turnt $(EXAMPLE_TESTS)

.PHONY: test_examples_save
test_examples_save:
	turnt --save $(EXAMPLE_TESTS)

.PHONY: save
save:
	turnt --save $(TESTS)

.PHONY: book
book:
	rm -rf book
	mdbook build

.PHONY: ts
ts:
	cd bril-ts ; \
	yarn ; \
	yarn build ; \

.PHONY: deploy
RSYNCARGS := --compress --recursive --checksum --itemize-changes \
	--delete -e ssh --perms --chmod=Du=rwx,Dgo=rx,Fu=rw,Fog=r
DEST := courses:coursewww/capra.cs.cornell.edu/htdocs/bril
deploy: book
	rsync $(RSYNCARGS) ./book/ $(DEST)
