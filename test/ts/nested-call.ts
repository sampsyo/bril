var y = 1n + getTwo();
console.log(y);

function getTwo(): bigint {
    return 2n;
}
