fn main() {
    let x: i32 = 4;
    let y: i32 = x + x;
    println!(y, y);
    let z: i32 = foobar();
    println!(z);
    let w: i32 = 3 + sum(x, z);
    println!(w);
    foobar();
    procedure();

    if !(w > x) {
        println!(w);
    } else {
        println!(x);
    }

    let mut x: i32 = 0;
    while x < 10 {
        println!(x);
        x: i32 = x + 1;
    }
}

fn foobar() -> i32 {
    return 42;
}

fn sum(x: i32, y: i32) -> i32 {
    return x + y;
}

fn procedure() {}
