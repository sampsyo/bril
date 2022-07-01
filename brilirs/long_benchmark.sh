#!/bin/bash

# Some ideas for faster code
# Feature gate type checking?

files=( "../benchmarks/core/ackermann.bril" "../benchmarks/mem/eight-queens.bril" \
        "../benchmarks/mem/mat-mul.bril" "../benchmarks/long/function_call.bril"

        )
jsons=( "../benchmarks/core/ackermann.json" "../benchmarks/mem/eight-queens.json" \
        "../benchmarks/mem/mat-mul.json" "../benchmarks/long/function_call.json"

        )
args=( "3 6" "8" "50 109658" "25")

for i in "${!files[@]}"; do
    bril2json < ${files[i]} > ${jsons[i]}

    export json=${jsons[i]}
    export arg=${args[i]}
    echo "file is ${files[i]}"
    echo "arg is $arg"
    hyperfine --warmup 5 -L interp brili,./target/release/brilirs '{interp} -p $arg < $json'

    rm ${jsons[i]}
done