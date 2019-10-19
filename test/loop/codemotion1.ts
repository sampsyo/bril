let a = 8;
let x = 0;
let y = 8;
let z = 1;
let n = 10;
for (let i = n; i > 0; i = i - 1) {
    x = y + z;
    a = a + x * x;
}
console.log(a);
