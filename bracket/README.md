# bracket

A compiler connecting two educational intermediate languages: [exprs-lang-v7](https://www.students.cs.ubc.ca/~cs-411/2022w2/v7_Languages.html#%28def._%28%28lib._cpsc411%2Flangs%2Fv7..rkt%29._exprs-lang-v7%29%29), Racket's subset from UBC's [CPSC411](https://www.students.cs.ubc.ca/~cs-411/2022w2/index.html) (Compiler Construction) course, and [Bril](https://capra.cs.cornell.edu/bril/), the Big Red Intermediate Language.

## Overview

`bracket` compiles programs from exprs-lang-v7 to Bril JSON. This enables Racket's expressive subset to serve as a rich frontend for generating Bril programs.

## Installation

### Prerequisites

1. **Install Racket**
   ```bash
   # Download from https://racket-lang.org/
   ```

2. **Install CPSC411 compiler library (2022w2 version)**
   ```bash
   raco pkg install https://github.com/cpsc411/cpsc411-pub.git?path=cpsc411-lib#2022w2
   ```

3. **Install Bril tools** (for interpreting/viewing output)
   ```bash
   # Follow instructions at https://github.com/sampsyo/bril
   ```
### Building bracket

```bash
mkdir -p bin
raco exe -o bin/bracket src/bracket.rkt
```

## Usage

### Basic Compilation

Compile an exprs-lang-v7 program to Bril JSON:
```bash
./bin/bracket tests/fib_recursive.rkt
```

### View as Bril Text

Convert the JSON output to human-readable Bril text:
```bash
./bin/bracket tests/fib_recursive.rkt | bril2txt
```

### Interpret with brili

Execute the compiled program:
```bash
./bin/bracket tests/fib_recursive.rkt | brili
```

**Expected output:** `55` (the 10th Fibonacci number as computed in `tests/fib_recursive.rkt`)

## Example

**Input** (`tests/fib_recursive.rkt`):
```racket
(module
    (define fib
      (lambda (n)
        (if (call <= n 1)
          1
          (call + (call fib (call - n 1)) 
                  (call fib (call - n 2))))))
    (let ([arg.x 10])
      (call fib arg.x)))
```

**Output:** Bril JSON program that computes the 10th Fibonacci number

## Benchmarking

Run the benchmark suite with:

```bash
brench turnt.toml
```

This executes all programs in `tests/` using two pipelines:

* **exprs**: direct interpretation of exprs-lang-v7 with `interp-exprs-lang-v7`
* **bril**: compilation to Bril and interpretation with `brili`

The output reports each benchmark’s result; matching values indicate correctness across pipelines.

## References

- [More about bracket](https://www.cs.cornell.edu/courses/cs6120/2025fa/blog/bracket/)
- [exprs-lang-v7 Grammar](https://www.students.cs.ubc.ca/~cs-411/2022w2/v7_Languages.html)
- [CPSC411 Course Materials](https://www.students.cs.ubc.ca/~cs-411/2022w2/index.html)
- [Bril Documentation](https://capra.cs.cornell.edu/bril/)
