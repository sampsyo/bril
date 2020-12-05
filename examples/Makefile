.PHONY: all
all: ssa_plot.png

ssa_results.csv: ssa_brench.toml
	brench $^ > $@

%.norm.csv: %.csv
	python normalize.py < $^ > $@

ssa_plot.png: ssa_plot.json ssa_results.norm.csv
	vl2png $< > $@
