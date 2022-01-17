TESTS :=  ../test/print/*.json \
		../test/parse/*.bril

.PHONY: test
test:
	turnt --diff -c turnt_bril_rs.toml $(TESTS)