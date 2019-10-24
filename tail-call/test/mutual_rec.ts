function isOdd(n: number): number {
    if (n == 0) {
        return 0;
    } else if (n == 1) {
        return 1;
    } else {
        return isEven(n-1);
    }
}

function isEven(n: number): number {
    if (n == 0) {
        return 1;
    } else if (n == 1) {
        return 0;
    } else {
        return isOdd(n-1);
    }
}

function main(): void {
    let n1 = 100000;
    let n1_is_even = isEven(n1);
    console.log(n1_is_even);
}
