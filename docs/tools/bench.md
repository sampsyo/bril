Benchmarks
==========

The `bench` directory in the Bril repository contains a fledgling suite of microbenchmarks that you can use to measure the impact of your optimizations.
(Benchmarks are different from tests because they are meant to actually calculate something instead of just exercising a language feature.)

The current benchmarks are:

* `check-primes`: Check the first *n* natural numbers for primality, printing out a 1 if the number is prime and a 0 if it's not.
* `fib`: Calculate the *n*th Fibonacci number by allocating and filling an [array](../lang/memory.md) of numbers up to that point.
* `fizz-buzz`: The infamous [programming test][fizzbuzz].
* `loopfact`: Compute *n!* imperatively using a loop.
* `quadratic`: The [quadratic formula][qf], including a hand-rolled implementation of square root.
* `recfact`: Compute *n!* using recursive function calls.

Credit for several of these benchmarks goes to Alexa VanHattum and Gregory Yauney, who implemented them for their [global value numbering project][gvnblog].

[fizzbuzz]: https://wiki.c2.com/?FizzBuzzTest
[qf]: https://en.wikipedia.org/wiki/Quadratic_formula
[gvnblog]: https://www.cs.cornell.edu/courses/cs6120/2019fa/blog/global-value-numbering/
