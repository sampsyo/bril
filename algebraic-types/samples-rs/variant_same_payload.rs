enum T {
    X,
    Y,
}

fn main() {
    let t: T = T::Y;
    match t {
        T::X => {
            let pr: i32 = 1;
            println!(pr);
        }
        T::Y => {
            let pr: i32 = 0;
            println!(pr);
        }
    }
}
