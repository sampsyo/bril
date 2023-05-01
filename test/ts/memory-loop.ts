import * as mem from "../../bril-ts/mem";

let size = 100n;
let arr = mem.alloc<bigint>(size);

for (let i = 0n; i < size; i = i + 1n) {
    mem.store(mem.ptradd(arr, i), i); // arr[i] = i
}

let sum = 0n;
for (let i = 0n; i < size; i = i + 1n) {
    sum = sum + mem.load(mem.ptradd(arr, i)); // sum += arr[i]
}
console.log(sum);

mem.free(arr);