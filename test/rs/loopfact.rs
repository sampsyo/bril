fn main() {
    let value:i64 = 8;
    let mut result:i64 = 1;
    let mut i:i64 = value;
    while i > 0 {
        result *= i;
        i-= 1;
    }
    println!("{:?}", result);
}
