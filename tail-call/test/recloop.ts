function loop(n: number) : number {
    if (n == 0) {
        return 0;
    } else {
        return loop(n - 1);
    }
}

function main() : void {
    let result = loop(100000);
    console.log(result);
}
