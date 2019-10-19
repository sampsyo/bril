let value = 8;
let x = 1;
let y = 10;
let z = 5;
let result = 1;
for (let i = value; i > 0; i = i - 1) {
  x = y + z;
  result = result + 6 * i + x * x;
}
console.log(result);
