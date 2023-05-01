import * as mem from "../../bril-ts/mem";

function allocateMatrix(size: bigint, startValue: bigint): mem.Pointer<mem.Pointer<bigint>> {
    let value = startValue;
    let matrix = mem.alloc<mem.Pointer<bigint>>(size);
    for (let i = 0n; i < size; i = i + 1n) {
        let row = mem.alloc<bigint>(size);
        for (let j = 0n; j < size; j = j + 1n) {
            mem.store(mem.ptradd(row, j), value); // row[j] = value
            value = value + 1n;
        }
        mem.store(mem.ptradd(matrix, i), row); // matrix[i] = row
    }
    return matrix;
}

function freeMatrix(matrix: mem.Pointer<mem.Pointer<bigint>>, size: bigint) {
    for (let i = 0n; i < size; i = i + 1n) {
        let row = mem.load(mem.ptradd(matrix, i)); // row = matrix[i]
        mem.free(row)
    }
    mem.free(matrix);
}

function printMatrix(matrix: mem.Pointer<mem.Pointer<bigint>>, size: bigint) {
    for (let i = 0n; i < size; i = i + 1n) {
        let row = mem.load(mem.ptradd(matrix, i)); // row = matrix[i]
        for (let j = 0n; j < size; j = j + 1n) {
            let value = mem.load(mem.ptradd(row, j)); // value = row[j]
            console.log(value);
        }
        let separator = true;
        console.log(separator)
    }
}

let size = 3n;
let m = allocateMatrix(size, 1n);
printMatrix(m, size)
freeMatrix(m, size);