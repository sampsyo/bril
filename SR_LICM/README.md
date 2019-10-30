### Description
`funcs.py` has utility functions that's useful for Loop Invariant Code Motion
and Strength Reduction. This file includes the `examples` folder using a
relative path.

`loop_opt` is doing LICM and implementing Strength
Reduction using `funcs.py`.

### Usage
`python3 loop_opt.py` takes a JSON format and outputs an "optimized" JSON
format.
We have benchmarking tests in the `test/loop` directory. To run these files use:
```
ts2bril <path to filename.ts> | python3 loop_opt.py
```


