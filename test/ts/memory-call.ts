import * as mem from "../../bril-ts/mem";

function update(p: mem.Pointer<bigint>): mem.Pointer<bigint> {
    mem.store(p, 42n);
    return p;
}

let p1 = mem.alloc<bigint>(1n);
let p2 = update(p1);
console.log(mem.load(p2));

mem.free(p1);