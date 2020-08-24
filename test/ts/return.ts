var x = 1n;
var y = 2n;
var v = add2(x, y);
console.log(v);

function add2(x: bigint, y: bigint): bigint {
    return x + y;
}
