
ifeq ($(VOPTEST),1)
TESTS := vtest/cfg_test/*.bril
#	vtest/tdce_test/*.bril
#	vtest/lvn_test/*.bril
else ifeq ($(OPTEST),1)
TESTS := test/cfg_test/*.bril
#	test/tdce_test/*.bril
#	test/lvn_test/*.bril
#	test/df_test/*.bril
else ifeq ($(VTEST),1)
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
