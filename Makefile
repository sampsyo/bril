TESTS := \
	test/ts/*.ts \
	test/cli/*.bril \
	test/func/*.bril \
    test/parse/*.bril \
	test/interp/*.bril \
	test/print/*.json 

.PHONY: test
test:
	turnt $(TESTS)

.PHONY: save
save:
	turnt --save $(TESTS)

.PHONY: book
book:
	rm -rf book
	mdbook build

.PHONY: deploy
RSYNCARGS := --compress --recursive --checksum --itemize-changes \
	--delete -e ssh --perms --chmod=Du=rwx,Dgo=rx,Fu=rw,Fog=r
DEST := courses:coursewww/capra.cs.cornell.edu/htdocs/bril
deploy: book
	rsync $(RSYNCARGS) ./book/ $(DEST)
