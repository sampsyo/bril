let x = 0;
for (let i = 0; i < 10; i = i + 1) {
  if (i < 5) {
    x = x + 1;
  }
  x = x * 2;
}
console.log(x);
