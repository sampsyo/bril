#!/bin/bash

# Some ideas for faster code
# Feature gate type checking?

files=( "../benchmarks/ackermann.bril" "../benchmarks/eight-queens.bril" \
        "../benchmarks/mat-mul.bril" "../benchmarks/function_call.bril"

        )
jsons=( "../benchmarks/ackermann.json" "../benchmarks/eight-queens.json" \
        "../benchmarks/mat-mul.json" "../benchmarks/function_call.json"

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