quadratic(0-5, 8, 21);

function sqrt(x : number) : number {
  for (let i : number = 1; i < x - 1; i = i + 1) {
      if (i * i >= x) {
          return i;
      }
  }
  return 0;
}

function quadratic(a : number, b : number, c : number) : void {
    let s : number = b * b - 4 * a * c;
    let d : number = 2 * a;
    let r1 : number = 0-b + sqrt(s);
    let r2 : number = 0-b - sqrt(s);
    console.log(r1 / d);
    console.log(r2 / d);
}