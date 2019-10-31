let a = 8;
let n = 1000;
for (let i = n; i > 0; i = i - 1) {
    for (let j = n; j > i; j = j - 1){
       a = a + 6 * i;
    }
}
console.log(a);
