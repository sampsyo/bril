Benchmarks
==========

The `bench` directory in the Bril repository contains a fledgling suite of microbenchmarks that you can use to measure the impact of your optimizations.
(Benchmarks are different from tests because they are meant to actually calculate something instead of just exercising a language feature.)

The current benchmarks are:

* `ackermann`: Print the value of `Ack(m, n)`, the [two-argument Ackermann–Péter function][ackermann].
* `adj2csr`: Convert a graph in [adjacency matrix][adj] format (dense representation) to [Compressed Sparse Row (CSR)][csr] format (sparse representation). The random graph is generated using the same [linear congruential generator][rng].
* `adler32`: Computes the [Adler-32 Checksum][adler32] of an integer array.
* `armstrong`: Determines if the input is an [Armstrong number][armstrong], a number that is the sum of its own digits each raised to the power of the number of digits.
* `binary-fmt`: Print the binary format for the given positive integer.
* `binary-search`: Search a target integer within an integer array, outputs the index of target.
* `birthday`: Simulation of the [birthday][birthday] paradox with an input of `n` people in a given room.
* `bitwise-ops`: Computes the OR, AND, or XOR between two 64-bit integers. (Three modes: 0 = AND, 1 = OR, 2 = XOR)
* `bitshift`: Computes the LEFTSHIFT and RIGHTSHIFT for any integer, also implements an efficient pow function for integers
* `bubblesort`: Sorting algorithm that works by repeatedly swapping the adjacent elements if they are in wrong order.
* `catalan`: Print the *n*th term in the [Catalan][catalan] sequence, compute using recursive function calls.
* `check-primes`: Check the first *n* natural numbers for primality, printing out a 1 if the number is prime and a 0 if it's not.
* `cholesky`: Perform Cholesky decomposition of a Hermitian and positive definite matrix. The result is validated by comparing with Python's `scipy.linalg.cholesky`.
* `collatz`: Print the [Collatz][collatz] sequence starting at *n*. Note: it is not known whether this will terminate for all *n*.
* `conjugate-gradient`: Uses conjugate gradients to solve `Ax=b` for any arbitrary positive semidefinite `A`.
* `cordic`: Print an approximation of sine(radians) using 8 iterations of the [CORDIC algorithm](https://en.wikipedia.org/wiki/CORDIC).
* `csrmv`: Multiply a sparse matrix in the [Compressed Sparse Row (CSR)][csr] format with a dense vector. The matrix and input vector are generated using a [Linear Feedback Shift Register](https://en.wikipedia.org/wiki/Linear-feedback_shift_register) random number generator.
* `digial-root`: Computes the digital root of the input number.
* `dot-product`: Computes the dot product of two vectors.
* `eight-queens`: Counts the number of solutions for *n* queens problem, a generalization of [Eight queens puzzle][eight_queens].
* `euclid`: Calculates the greatest common divisor between two large numbers using the [Euclidean Algorithm][euclid] with a helper function for the modulo operator.
* `euler`: Approximates [Euler's number][euler] using the Taylor series.
* `fact`: Prints the factorial of *n*, computing it recursively.
* `factors`: Print the factors of the *n* using the [trial division][trialdivision] method.
* `fib`: Calculate the *n*th Fibonacci number by allocating and filling an [array](../lang/memory.md) of numbers up to that point.
* `fizz-buzz`: The infamous [programming test][fizzbuzz].
* `function_call`: For benchmarking the overhead of simple function calls.
* `gcd`: Calculate Greatest Common Divisor (GCD) of two input positive integer using [Euclidean algorithm][euclidean_into].
* `hanoi`: Print the solution to the *n*-disk [Tower of Hanoi][hanoi] puzzle.
* `is-decreasing`: Print if a number contains strictly decreasing digits.
* `lcm`: Compute LCM for two numbers using a very inefficient loop.
* `loopfact`: Compute *n!* imperatively using a loop.
* `major-elm`: Find the majority element in an array using [a linear time voting algorithm](https://www.cs.utexas.edu/~moore/best-ideas/mjrty/).
* `mandelbrot`: Generates a really low resolution, ascii, [mandelbrot set][mandelbrot].
* `mat-inv` : Calculates the inverse of a 3x3 matrix and prints it out.
* `mat-mul`: Multiplies two `nxn` matrices using the [naive][matmul] matrix multiplication algorithm. The matrices are randomly generated using a [linear congruential generator][rng].
* `max-subarray`: solution to the classic Maximum Subarray problem.
* `mod_inv`: Calculates the [modular inverse][modinv] of `n` under to a prime modulus p.
* `newton`: Calculate the square root of 99,999 using the [newton method][newton]
* `norm`: Calculate the [euclidean norm][euclidean] of a vector 
* `n_root`: Calculate nth root of a float using newton's method.
* `orders`: Compute the order ord(u) for each u in a cyclic group [<Zn,+>][cgroup] of integers modulo *n* under the group operation + (modulo *n*). Set the second argument *is_lcm* to true if you would like to compute the orders using the lowest common multiple and otherwise the program will use the greatest common divisor.
* `pascals-row`: Computes a row in Pascal's Triangle.
* `palindrome`: Outputs a 0-1 value indicating whether the input is a [palindrome][palindrome] number.
* `perfect`: Check if input argument is a perfect number.  Returns output as Unix style return code.
* `pow`: Computes the n^<sup>th</sup> power of a given (float) number.
* `primes-between`: Print the primes in the interval `[a, b]`.
* `primitive-root`: Computes a [primitive root][primitive_root] modulo a prime number input.
* `pythagorean_triple`: Prints all Pythagorean triples with the given c, if such triples exist. An intentionally very naive implementation.
* `quadratic`: The [quadratic formula][qf], including a hand-rolled implementation of square root.
* `quickselect`: Find the kth smallest element in an array using the quickselect algorithm.
* `quicksort`: [Quicksort using the Lomuto partition scheme][qsort]. 
* `quicksort-hoare`: Quicksort using [Hoare partioning][qsort-hoare] and median of three pivot selection.
* `recfact`: Compute *n!* using recursive function calls.
* `rectangles-area-difference`: Output the difference between the areas of rectangles (as a positive value) given their respective side lengths.
* `fitsinside`: Output whether or not a rectangle fits inside of another rectangle given the width and height lengths.
* `relative-primes`: Print all numbers relatively prime to *n* using [Euclidean algorithm][euclidean_into].
* `riemann`: Prints the left, midpoint, and right [Riemann][riemann] Sums for a specified function, which is the square function in this benchmark.
* `sieve`: Print all prime numbers up to *n* using the [Sieve of Eratosthenes][sievee].
* `sqrt`: Implements the [Newton–Raphson Method][newton] of approximating the square root of a number to arbitrary precision
* `sum-bit`: Print the number of 1-bits in the binary representation of the input integer.
* `sum-check`: Compute the sum of [1, n] by both loop and formula, and check if the result is the same.
* `sum-divisors`: Prints the positive integer divisors of the input integer, followed by the sum of the divisors.
* `sum-sq-diff`: Output the difference between the sum of the squares of the first *n* natural numbers and the square of their sum.
* `totient`: Computes [Euler's totient function][totient] on an input integer *n*.
* `two-sum`: Print the indices of two distinct elements in the list [2, 7, 11, 13] whose sum equals the input.
* `up-arrow`: Computes [Knuth's up arrow][uparrow] notation, with the first argument being the number, the second argument being the number of Knuth's up arrows, and the third argument being the number of repeats.
* `vsmul`: Multiplies a constant scalar to each element of a large array. Tests the performance of vectorization optimizations.
* `reverse`: Compute number with reversed digits (e.g. 123 -> 321).

Credit for several of these benchmarks goes to Alexa VanHattum and Gregory Yauney, who implemented them for their [global value numbering project][gvnblog].

[birthday]: https://en.wikipedia.org/wiki/Birthday_problem
[cgroup]: https://en.wikipedia.org/wiki/Cyclic_group#Cyclically_ordered_groups
[fizzbuzz]: https://wiki.c2.com/?FizzBuzzTest
[qf]: https://en.wikipedia.org/wiki/Quadratic_formula
[gvnblog]: https://www.cs.cornell.edu/courses/cs6120/2019fa/blog/global-value-numbering/
[sievee]: https://en.wikipedia.org/wiki/Sieve_of_Eratosthenes
[collatz]: https://en.wikipedia.org/wiki/Collatz_conjecture
[catalan]: https://en.wikipedia.org/wiki/Catalan_number
[ackermann]: https://en.wikipedia.org/wiki/Ackermann_function
[newton]: https://en.wikipedia.org/wiki/Newton%27s_method
[matmul]: https://en.wikipedia.org/wiki/Matrix_multiplication_algorithm#Iterative_algorithm
[rng]: https://en.wikipedia.org/wiki/Linear_congruential_generator
[euclidean_into]: https://en.wikipedia.org/wiki/Euclidean_algorithm
[euclid]: https://en.wikipedia.org/wiki/Euclidean_algorithm#Euclidean_division
[eight_queens]: https://en.wikipedia.org/wiki/Eight_queens_puzzle
[newton]: https://en.wikipedia.org/wiki/Newton%27s_method
[trialdivision]: https://en.wikipedia.org/wiki/Trial_division
[adj]: https://en.wikipedia.org/wiki/Adjacency_matrix
[csr]: https://en.wikipedia.org/wiki/Sparse_matrix
[armstrong]: https://en.wikipedia.org/wiki/Narcissistic_number
[adler32]: https://en.wikipedia.org/wiki/Adler-32
[uparrow]: https://en.wikipedia.org/wiki/Knuth%27s_up-arrow_notation
[riemann]: https://en.wikipedia.org/wiki/Riemann_sum
[primitive_root]: https://en.wikipedia.org/wiki/Primitive_root_modulo_n
[mandelbrot]: https://en.wikipedia.org/wiki/Mandelbrot_set
[palindrome]: https://en.wikipedia.org/wiki/Palindrome
[hanoi]: https://en.wikipedia.org/wiki/Tower_of_Hanoi
[euler]: https://en.wikipedia.org/wiki/E_(mathematical_constant)
[euclidean]: https://en.wikipedia.org/wiki/Norm_(mathematics)
[qsort]: https://en.wikipedia.org/wiki/Quicksort#Lomuto_partition_scheme
[qsort-hoare]: https://en.wikipedia.org/wiki/Quicksort#Hoare_partition_scheme
[modinv]: https://en.wikipedia.org/wiki/Modular_multiplicative_inverse
[totient]: https://en.wikipedia.org/wiki/Euler's_totient_function
