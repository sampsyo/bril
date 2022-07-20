fn main() {
    let x:i64 = 5;
    let f:i64 = fac(x);
    println!("{:?}", f);
}

fn fac(x:i64) -> i64 {
    if x <= 1 {
        return 1;
    }
    let result: i64 = x * fac(x -1);
    return result;
}
