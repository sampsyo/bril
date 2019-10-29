var n : number = 50;

for (let i : number = 1; i < n; i = i + 1) { 
  let isPrime : boolean = checkPrime(i);
  if (isPrime) {
    console.log(1);
  } else {
    console.log(0);
  }
}

function checkPrime(x : number) : boolean {
  if (x <= 1) {
     return false; 
  }

  // Check from 2 to x - 1 
  for (let i : number = 2; i < x; i = i + 1) {
    let div : number = x / i;
    let isDivisible : boolean = div * i === x; 
    if (isDivisible) {
      return false;
    } 
  }
  return true;
}