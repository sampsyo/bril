var x: bigint = 5n;
var f: bigint = fac(x);
console.log(f);

function fac(x: bigint): bigint {
    if (x <= 1n) {
        return 1n;
    }
    var result = x * fac(x - 1n);
    return result;
}
