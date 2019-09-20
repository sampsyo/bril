ifeq ($(OPTEST),1)
TESTS := examples/tdce_test/*.bril \
	examples/lvn_test/*.bril \
#	examples/df_test/*.bril
else ifeq ($(VTEST),1)
TESTS := vtest/parse/*.bril \
	vtest/print/*.json
#	vtest/interp/*.brili
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
