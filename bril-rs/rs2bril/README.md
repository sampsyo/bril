# rs2bril

This project is a Rust version of the `ts2bril` tool. `rs2bril` compiles a subset of Rust to Bril. See the test cases or `example.rs` for the subset of supported syntax. The goal is to support core bril operations like integer arithmetic/comparisons, function calls, control flow like if/while, and printing. It will additionally support floating point operations like those in Bril and memory operations via Rust arrays/slices.

View the interface with `cargo doc --open` or install with `make install` using the Makefile in `bril/bril_rs`. Then use `rs2bril --help` to get the help page for `rs2bril` with all of the supported flags.

## Limitations

- Currently, types are not inferred for variable declarations so all let bindings must be explicitly annotated: `let x:i64 = 5;`
- The `println!` macro is special cased to be converted into a `print` call in Bril. It takes a valid `println!` call, ignores the first argument which it assumes to be a format string, and assumes all of the following comma-separated arguments are variables: `println!("this is ignored{}", these, must, be, variables);`
- There currently isn't a way to pass arguments to the main function that is also valid Rust. You can either declare arguments in the main function such that it is no longer valid Rust but will generate the correct Bril code, or declared them as const values in the first line and edit the Bril IR later: `fn main(a:i64) {}` or `fn main() {let a:i64 = 0;}`
- Automatic static memory management <https://www.cs.cornell.edu/courses/cs6120/2020fa/blog/asmm/> has not been implemented so arrays must be explicitly dropped where `drop` is specialized to translate to a call to free: `drop([0]);`
- For loops, `continue`, `break`, and ranges have not been implemented(but could be).
- Memory is implemented using Rust arrays. These are statically sized values unlike how calls to Bril `alloc` can be dynamically sized. One solution is to just allocate a large enough array and then treat the dynamic size like the length. (A subset of vectors could also be specialized in the future).
- In normal Rust, `if` can also be used as an expression that evaluates a value to be put in a variable. This is not implemented and it is assumed that there will only be if statements.
- The Bril code that it produces is super inefficient and it is left to other tools to optimize it. Array initialization is unrolled is not an optimal solution.
- `!=` and automatic promotions of integer literals to floats are not implemented.
- to support indexing into arrays, you can cast to usize in the Rust code. This will be ignored when generating Bril. `arr[i as usize];`
- The parts of Rust which make it valid like lifetimes, references, mutability, and function visibility are ignored and compiled away in Bril.
