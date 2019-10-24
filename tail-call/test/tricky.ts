function foo(n: number): number {
    if (n <= 0) {
        return 0;
    }

    let result = 0;
    if (n/2 * 2 == n) {
        result = foo(n-1);
    } else {
        result = foo(n-3);
    }
    return result;
}

function main(): void {
    let result = foo(100);
    console.log(result);
}
