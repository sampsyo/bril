#!/bin/bash

files=( "../benchmarks/ackermann.bril" "../benchmarks/binary-fmt.bril" "../benchmarks/check-primes.bril" \
        "../benchmarks/collatz.bril" "../benchmarks/digital-root.bril" "../benchmarks/eight-queens.bril" \
        "../benchmarks/euclid.bril" "../benchmarks/fib.bril" "../benchmarks/fizz-buzz.bril" \
        "../benchmarks/gcd.bril" "../benchmarks/loopfact.bril" "../benchmarks/mat-mul.bril" \
        "../benchmarks/orders.bril" "../benchmarks/perfect.bril" "../benchmarks/pythagorean_triple.bril" \
        "../benchmarks/quadratic.bril" "../benchmarks/ray-sphere-intersection.bril" \
        "../benchmarks/recfact.bril" "../benchmarks/sieve.bril" "../benchmarks/sqrt.bril" \
        "../benchmarks/sum-bits.bril" "../benchmarks/sum-sq-diff.bril" "../benchmarks/function_call.bril"
        )
jsons=( "../benchmarks/ackermann.json" "../benchmarks/binary-fmt.json" "../benchmarks/check-primes.json" \
        "../benchmarks/collatz.json" "../benchmarks/digital-root.json" "../benchmarks/eight-queens.json" \
        "../benchmarks/euclid.json" "../benchmarks/fib.json" "../benchmarks/fizz-buzz.json" \
        "../benchmarks/gcd.json" "../benchmarks/loopfact.json" "../benchmarks/mat-mul.json" \
        "../benchmarks/orders.json" "../benchmarks/perfect.json" "../benchmarks/pythagorean_triple.json" \
        "../benchmarks/quadratic.json" "../benchmarks/ray-sphere-intersection.json" \
        "../benchmarks/recfact.json" "../benchmarks/sieve.json" "../benchmarks/sqrt.json" \
        "../benchmarks/sum-bits.json" "../benchmarks/sum-sq-diff.json" "../benchmarks/function_call.json"
        )
args=( "3 6" "128" "50" "7" "645634654" "8" "" "10" "101" "4 20" "8" "50 109658" "96 false" "496" "125" \
        "-5 8 21" "" "8" "100" "" "42" "100" "25")

for i in "${!files[@]}"; do
    bril2json < ${files[i]} > ${jsons[i]}

    export json=${jsons[i]}
    export arg=${args[i]}
    echo "file is ${files[i]}"
    echo "arg is $arg"
    hyperfine --warmup 5 -L interp brili,./target/release/brilirs '{interp} -p $arg < $json'

    rm ${jsons[i]}
done