Brench
======

Brench is a simple benchmark runner to help you measure the impact of optimizations.
It can run the same set of benchmarks under multiple treatments, check that they still produce the correct answer, and report their performance under every condition.

Set Up
------

Brench is a Python tool.
There is a `brench/` subdirectory in the Bril repository.
Get [Flit][] and then type:

    $ flit install --symlink --user

[flit]: https://flit.readthedocs.io/

Configure
---------

Write a configuration file using [TOML][].
Start with something like this:

    extract = 'total_dyn_inst: (\d+)'
    benchmarks = '../benchmarks/*.bril'

    [runs.baseline]
    pipeline = [
        "bril2json",
        "brili -p {args}",
    ]

    [runs.myopt]
    pipeline = [
        "bril2json",
        "myopt",
        "brili -p {args}",
    ]

The global options are:

* `extract`:
  A regular expression to extract the figure of merit from a given run of a given benchmark.
  The example above gets the simple profiling output from [the Bril interpreter][interp] in `-p` mode.
* `benchmarks` (optional):
  A shell glob matching the benchmark files to run.
  You can also specify the files on the command line (see below).
* `timeout` (optional):
  The timeout of each benchmark run in seconds. Default of 5 seconds.

Then, define an map of *runs*, which are the different treatments you want to give to each benchmark.
Each one needs a `pipeline`, which is a list of shell commands to run in a pipelined fashion on the benchmark file, which Brench will send to the first command's standard input.
The first run constitutes the "golden" output; subsequent runs will need to match this output.

[toml]: https://toml.io/
[interp]: interp.md

Run
---

Just give Brench your config file and it will give you results as a CSV:

    $ brench example.toml > results.csv

You can also specify a list of files after the configuration file to run a specified list of benchmarks, ignoring the pre-configured glob in the configuration file.

The command has only one command-line option:

* `--jobs` or `-j`:
  The number of parallel jobs to run. Set to 1 to run everything sequentially.
  By default, Brench tries to guess an adequate number of threads to fill up your machine.

The output CSV has three columns: `benchmark`, `run`, and `result`.
The latter is the value extracted from the run's standard output and standard error using the `extract` regular expression or one of these three status indicators:

* `incorrect`: The output did not match the "golden" output (from the first run).
* `timeout`: Execution took too long.
* `missing`: The `extract` regex did not match in the final pipeline stage's standard output or standard error.

To check that a run's output is "correct," Brench compares its standard output
to that of the first run (`baseline` in the above example, but it's whichever run
configuration comes first). The comparison is an exact string match.
