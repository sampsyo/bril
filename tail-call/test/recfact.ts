function fact(n: number, acc: number): number {
    if (n == 0) {
        return acc;
    } else {
        return fact(n-1, acc * n);
    }
}

function main(): void {
    let n = 8;
    let result = fact(n, 1);
    console.log(result);
}
