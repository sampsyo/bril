BENCHMARKS := $(wildcard *.bril)

.PHONY: bench clean plot
bench:
	turnt -e bench --save $(BENCHMARKS)
clean:
	rm -f *.bench.json plot.svg bench.csv
plot: plot.svg

bench.csv: $(wildcard *.bench.json)
	python3 summarize.py $^ > $@

%.svg: %.vl.json bench.csv
	npx -p vega -p vega-lite vl2svg $*.vl.json > $@
