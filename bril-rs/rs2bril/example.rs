fn test(x: i64, f: f64, b: bool, ptr: [i64; 2], ptr2: &[[i64; 2]]) -> i64 {
    let x: i64 = -1;
    let y: bool = !(!true);
    let z: f64 = -0.0;
    z = -1.0;
    f = 5.0;
    f /= 2.0;
    ptr[0] = 1;
    ptr[1] = ptr[2 + x];

    let foo : i64 = test2(1, 2);
    test3();

    let test: [i64;2] = [0, 1];
    let test2: [[i64; 2]; 1] = [test];
    let test3: [f64;10] = [z; 10];

    if true {
        let cond: i64 = 0;
        while cond < 2 {
            cond += 1;
        }
    } else {
        if true {
            let cond: bool = false;
            while cond {}
        }
    }

    println!("{}", x);

    return x;
}


fn test2(x:i64, y:i64) -> i64 {
    return x;
}

fn test3() {
    return;
}