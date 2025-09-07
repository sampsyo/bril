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
* `binpow`: Recursive binary exponentiation. Implemented to be as close to tail-recursive as possible.
* `birthday`: Simulation of the [birthday][birthday] paradox with an input of `n` people in a given room.
* `bitwise-ops`: Computes the OR, AND, or XOR between two 64-bit integers. (Three modes: 0 = AND, 1 = OR, 2 = XOR)
* `bitshift`: Computes the LEFTSHIFT and RIGHTSHIFT for any integer, also implements an efficient pow function for integers
* `bbs`: Prints the least significant bit of each element in a given Blum Blum Blum PRNG sequence.
* `bubblesort`: Sorting algorithm that works by repeatedly swapping the adjacent elements if they are in wrong order.
* `catalan`: Print the *n*th term in the [Catalan][catalan] sequence, compute using recursive function calls.
* `char-poly`: Calculates the characteristic polynomial of a given 3x3 matrix, returning its coefficients as an array.
* `check-primes`: Check the first *n* natural numbers for primality, printing out a 1 if the number is prime and a 0 if it's not.
* `cholesky`: Perform Cholesky decomposition of a Hermitian and positive definite matrix. The result is validated by comparing with Python's `scipy.linalg.cholesky`.
* `collatz`: Print the [Collatz][collatz] sequence starting at *n*. Note: it is not known whether this will terminate for all *n*.
* `conjugate-gradient`: Uses conjugate gradients to solve `Ax=b` for any arbitrary positive semidefinite `A`.
* `connected-components`: Compute and print each [connected component][component] in the [adjacency matrix][adj] of an undirected graph.
* `1dconv`: Creates a kernel and array, performs a one-dimensional convolution operation, and prints out the values in the resulting array.
* `2dconvol`: Creates a 2d image and kernel and performs a 2d convolution on the image, and prints out the values of the image, kernel, and resulting output. 
* `cordic`: Print an approximation of sine(radians) using 8 iterations of the [CORDIC algorithm](https://en.wikipedia.org/wiki/CORDIC).
* `csrmv`: Multiply a sparse matrix in the [Compressed Sparse Row (CSR)][csr] format with a dense vector. The matrix and input vector are generated using a [Linear Feedback Shift Register](https://en.wikipedia.org/wiki/Linear-feedback_shift_register) random number generator.
* `digial-root`: Computes the digital root of the input number.
* `dead-branch`: Repeatedly call a br instruction whose condition always evaluates to false. The dead branch should be pruned by a smart compiler.
* `delannoy`: Recusively computes the number of paths on square board with size `n` from southwest corner (0, 0) to northeast corner (n, n), using only single steps north, northeast or east. This number is known as the [Delannoy number](https://en.wikipedia.org/wiki/Delannoy_number).
* `dot-product`: Computes the dot product of two vectors.
* `eight-queens`: Counts the number of solutions for *n* queens problem, a generalization of [Eight queens puzzle][eight_queens].
* `euclid`: Calculates the greatest common divisor between two large numbers using the [Euclidean Algorithm][euclid] with a helper function for the modulo operator.
* `euler`: Approximates [Euler's number][euler] using the Taylor series.
* `exponentiation-by-squaring`: Fast iterative computation of large integer powers using [exponentiation by squaring][exp_by_squaring].
* `fact`: Prints the factorial of *n*, computing it recursively.
* `factors`: Print the factors of the *n* using the [trial division][trialdivision] method.
* `fib`: Calculate the *n*th Fibonacci number by allocating and filling an [array](../lang/memory.md) of numbers up to that point.
* `fizz-buzz`: The infamous [programming test][fizzbuzz].
* `fnv1-hash`: Compute the [Fowler-Noll-Vo hash function](https://en.wikipedia.org/wiki/Fowler%E2%80%93Noll%E2%80%93Vo_hash_function) of an integer array.
* `function_call`: For benchmarking the overhead of simple function calls.
* `gebmm`: Perform binary matrix multiplication of two matrices represented by packed integer arguments.
* `gcd`: Calculate Greatest Common Divisor (GCD) of two input positive integer using [Euclidean algorithm][euclidean_into].
* `geometric-sum`: Calculate [Geometric Sum](https://en.wikipedia.org/wiki/Geometric_series) given first term, common ratio and number of terms.
* `gol`: Print the next iteration for a matrix in [Conway's Game of Life](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life).
* `hanoi`: Print the solution to the *n*-disk [Tower of Hanoi][hanoi] puzzle.
* `hamming`: Computes the hamming distance between two integers.
* `insertion-sort`: Insertion sort algorithm in ascending order. 
* `is-decreasing`: Print if a number contains strictly decreasing digits.
* `karatsuba`: Computes the product of 2 integers using [Karatsuba's Algorithm](https://en.wikipedia.org/wiki/Karatsuba_algorithm).
* `lcm`: Compute LCM for two numbers using a very inefficient loop.
* `leibniz`: Approximates Pi using [Leibniz formula](https://en.wikipedia.org/wiki/Leibniz_formula_for_%CF%80).
* `lis`: Calculates the longest increasing subsequence of an 8-element array using dynamic programming.
* `logistic`: Compute the [logistic map](https://en.wikipedia.org/wiki/Logistic_map).
* `loopfact`: Compute *n!* imperatively using a loop.
* `major-elm`: Find the majority element in an array using [a linear time voting algorithm](https://www.cs.utexas.edu/~moore/best-ideas/mjrty/).
* `mandelbrot`: Generates a really low resolution, ascii, [mandelbrot set][mandelbrot].
* `mat-inv` : Calculates the inverse of a 3x3 matrix and prints it out.
* `mat-mul`: Multiplies two `nxn` matrices using the [naive][matmul] matrix multiplication algorithm. The matrices are randomly generated using a [linear congruential generator][rng].
* `max-subarray`: solution to the classic Maximum Subarray problem.
* `mccarthy91`: Run the [McCarthy 91 function][mccarthy91], a nested-recursive function which always returns 91.
* `mod_inv`: Calculates the [modular inverse][modinv] of `n` under to a prime modulus p.
* `montecarlo`: Calculates the value of pi using a [xorshift](https://en.wikipedia.org/wiki/Xorshift) as the rng.
* `montgomery`: Computes modular multiplication using the [Montgomery reduction][montgomery] algorithm.
* `mod_pow`: Performs [modular exponentiation][modpow] using the right-to-left binary method.
* `newton`: Calculate the square root of 99,999 using the [newton method][newton]
* `norm`: Calculate the [euclidean norm][euclidean] of a vector
* `n_root`: Calculate nth root of a float using newton's method.
* `orders`: Compute the order ord(u) for each u in a cyclic group [<Zn,+>][cgroup] of integers modulo *n* under the group operation + (modulo *n*). Set the second argument *is_lcm* to true if you would like to compute the orders using the lowest common multiple and otherwise the program will use the greatest common divisor.
* `pascals-row`: Computes a row in Pascal's Triangle.
* `palindrome`: Outputs a 0-1 value indicating whether the input is a [palindrome][palindrome] number.
* `perfect`: Check if input argument is a perfect number.  Returns output as Unix style return code.
* `permutation`: Calculates the number of possible permutations of k objects from a set of n.
* `pow`: Computes the n^<sup>th</sup> power of a given (float) number.
* `primes-between`: Print the primes in the interval `[a, b]`.
* `primitive-root`: Computes a [primitive root][primitive_root] modulo a prime number input.
* `pythagorean_triple`: Prints all Pythagorean triples with the given c, if such triples exist. An intentionally very naive implementation.
* `quadratic`: The [quadratic formula][qf], including a hand-rolled implementation of square root.
* `quickselect`: Find the kth smallest element in an array using the quickselect algorithm.
* `quicksort`: [Quicksort using the Lomuto partition scheme][qsort].
* `quicksort-hoare`: Quicksort using [Hoare partioning][qsort-hoare] and median of three pivot selection.
* `random-walk`: Perform a simple random walk on the integer lattice [`Z^d`][int_lattice]: that is, starting from the origin, repeatedly make a random move of length one in one of the lattice directions. The walk terminates after 100000 steps or upon returning to the start.
* `ray-bbox-intersection`: Finds whether a ray intersects an axis-aligned bounding box.
* `recfact`: Compute *n!* using recursive function calls.
* `rectangles-area-difference`: Output the difference between the areas of rectangles (as a positive value) given their respective side lengths.
* `fitsinside`: Output whether or not a rectangle fits inside of another rectangle given the width and height lengths.
* `relative-primes`: Print all numbers relatively prime to *n* using [Euclidean algorithm][euclidean_into].
* `riemann`: Prints the left, midpoint, and right [Riemann][riemann] Sums for a specified function, which is the square function in this benchmark.
* `rot13`: Prints the [rot13][rot13] substitution of a character (represented as an integer in the range 0 to 25).
* `shufflesort`: Sorts a list by shuffling it until it is sorted.
* `sieve`: Print all prime numbers up to *n* using the [Sieve of Eratosthenes][sievee].
* `sorting-network-five`: An optimal sorting network for 5 integer inputs.
* `sqrt`: Implements the [Newton–Raphson Method][newton] of approximating the square root of a number to arbitrary precision
* `sqrt_bin_search`: Uses a binary search to find the floor of the square root of an integer
* `sum-bit`: Print the number of 1-bits in the binary representation of the input integer.
* `sum-check`: Compute the sum of [1, n] by both loop and formula, and check if the result is the same.
* `sum-digits`: Compute the sum of the (base-10) digits of the input integer.
* `sum-divisors`: Prints the positive integer divisors of the input integer, followed by the sum of the divisors.
* `sum-sq-diff`: Output the difference between the sum of the squares of the first *n* natural numbers and the square of their sum.
* `sum-of-cubes`: Computes the sum of the first n cubes using the closed-cube formula and prints the result.
* `totient`: Computes [Euler's totient function][totient] on an input integer *n*.
* `two-sum`: Print the indices of two distinct elements in the list [2, 7, 11, 13] whose sum equals the input.
* `up-arrow`: Computes [Knuth's up arrow][uparrow] notation, with the first argument being the number, the second argument being the number of Knuth's up arrows, and the third argument being the number of repeats.
* `vsmul`: Multiplies a constant scalar to each element of a large array. Tests the performance of vectorization optimizations.
* `reverse`: Compute number with reversed digits (e.g. 123 -> 321).
* `combination`: Compute binomial combination, ie. n choose k, for positive integers.
* `fib_recursive`: Computes the *n*th Fibonacci number using recursion, where `fib(n) = fib(n-1) + fib(n-2)`, with base cases `fib(0) = 0` and `fib(1) = 1`. Demonstrates recursive function calls and branching.
* `cordic`: Compute the sine of a number using the [CORDIC][cordic] algorithm

Credit for several of these benchmarks goes to Alexa VanHattum and Gregory Yauney, who implemented them for their [global value numbering project][gvnblog].

[birthday]: https://en.wikipedia.org/wiki/Birthday_problem
[cgroup]: https://en.wikipedia.org/wiki/Cyclic_group#Cyclically_ordered_groups
[fizzbuzz]: https://wiki.c2.com/?FizzBuzzTest
[qf]: https://en.wikipedia.org/wiki/Quadratic_formula
[gvnblog]: https://www.cs.cornell.edu/courses/cs6120/2019fa/blog/global-value-numbering/
[sievee]: https://en.wikipedia.org/wiki/Sieve_of_Eratosthenes
[collatz]: https://en.wikipedia.org/wiki/Collatz_conjecture
[catalan]: https://en.wikipedia.org/wiki/Catalan_number
[component] : https://en.wikipedia.org/wiki/Component_(graph_theory)
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
[rot13]: https://en.wikipedia.org/wiki/ROT13
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
[mccarthy91]: https://en.wikipedia.org/wiki/McCarthy_91_function
[exp_by_squaring]: https://en.wikipedia.org/wiki/Exponentiation_by_squaring
[montgomery]: https://en.wikipedia.org/wiki/Montgomery_modular_multiplication#The_REDC_algorithm
[modpow]: https://en.wikipedia.org/wiki/Modular_exponentiation
[cordic]: https://en.wikipedia.org/wiki/CORDIC
[int_lattice]: https://en.wikipedia.org/wiki/Integer_lattice

