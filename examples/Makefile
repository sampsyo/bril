.PHONY: all
all: ssa_plot.svg

ssa_results.csv: ssa_brench.toml
	brench $^ > $@

%.norm.csv: %.csv
	python normalize.py < $^ > $@

ssa_plot.svg: ssa_plot.json ssa_results.norm.csv
	npx vl2svg $< > $@
