fn main(x: i32, y: i32) {
    let res: Res = safe_divide(x, y);
    match res {
        Res::Res(res) => println!(res),
        Res::DivideByZero => {
            let pr: i32 = -1;
            println!(pr)
        }
    }
}

enum Res {
    Res(i32),
    DivideByZero,
}

fn safe_divide(x: i32, y: i32) -> Res {
    if y == 0 {
        return Res::DivideByZero;
    } else {
        return Res::Res(x / y);
    }
}
