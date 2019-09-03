TESTS := test/parse/*.bril \
	test/print/*.json \
	test/interp/*.bril \
	test/ts/*.ts

.PHONY: test
test:
	turnt $(TESTS)

.PHONY: save
save:
	turnt --save $(TESTS)
