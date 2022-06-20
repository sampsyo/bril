CORE_BENCH := $(shell grep -L 'ptr' *.bril)

.PHONY: bench clean plot
bench:
	turnt -e bench --save $(CORE_BENCH)
clean:
	rm -f *.bench.json plot.svg bench.csv
plot: plot.svg

bench.csv: $(wildcard *.bench.json)
	python3 summarize.py $^ > $@

%.svg: %.vl.json bench.csv
	npx -p vega -p vega-lite vl2svg $*.vl.json > $@
