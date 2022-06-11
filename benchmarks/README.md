Bril Benchmarks
===============

This directory contains a suite of benchmarks in Bril contributed by the community. You can read [more details in the documentation][bench-docs].

There is also some basic infrastructure here for benchmarking the wall-clock execution time of Bril implementations. Run the benchmarks comparing [the reference interpreter][brili], [brilirs][], and [brilift][] by installing [Hyperfine][] and then typing:

    make bench

Then you can aggregate the statistics with:

    make bench.csv

That shows you the [harmonic mean][hm] speedups over the reference interpreter as a baseline.
You can also generate a bar chart using [Vega-Lite][]:

    make plot

[vega-lite]: https://vega.github.io/vega-lite/
[bench-docs]: https://capra.cs.cornell.edu/bril/tools/bench.html
[brili]: https://capra.cs.cornell.edu/bril/tools/interp.html
[brilirs]: https://capra.cs.cornell.edu/bril/tools/brilirs.html
[brilift]: https://capra.cs.cornell.edu/bril/tools/brilift.html
[hm]: https://en.wikipedia.org/wiki/Harmonic_mean
[hyperfine]: https://github.com/sharkdp/hyperfine
