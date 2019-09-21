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
