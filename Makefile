ifeq ($(VTEST),1)
TESTS := vtest/parse/*.bril \
	vtest/print/*.json \
	vtest/interp/*.bril
else
TESTS := test/parse/*.bril \
	test/print/*.json \
	test/interp/*.bril \
	test/ts/*.ts 
endif

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
