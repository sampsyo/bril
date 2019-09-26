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
