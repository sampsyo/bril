var x : number = 5;
var f : number = fac(x);
console.log(f);

function fac(x : number) : number {
    if (x <= 1) {
        return 1;
    }
    var result = x * fac(x - 1);
    return result; 
}