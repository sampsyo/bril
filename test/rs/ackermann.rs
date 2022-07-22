//# Compute the Ackermann function recursively.
//# WARNING: Will quickly exceed stack size
//# ARGS: 3 6
fn main(m:i64, n:i64) {
    let tmp : i64 = ack(m, n);
    println!("{:?}", tmp);
}

fn ack(m:i64, n:i64) -> i64 {
    if m == 0 {
        return n + 1;
    } else if n == 0 {
        return ack(m -1, 1);
    } else {
        let t1 : i64 = ack(m, n -1);
        return ack(m -1, t1);
    }
}
