let value = 8n;
let result = 1n;
for (let i = value; i > 0n; i = i - 1n) {
  result = result * i;
}
console.log(result);
