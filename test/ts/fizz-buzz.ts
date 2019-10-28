for (let index : number = 1; index < 101; index = index + 1) {
  let div3 : number = index / 3;
  let isFizz : boolean = div3 * 3 === index; 
  let div5 : number = index / 5;
  let isBuzz : boolean = div5 * 5 === index; 
  if (isFizz) {
    if (isBuzz) {
      console.log(0-1);
    } else {
      console.log(0-2);
    }
  } else if (isBuzz) {
    console.log(0-3);
  } else {
    console.log(index);    
  }
}