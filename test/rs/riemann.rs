//# riemann sums from wikipedia article on riemann sums

fn main() {
    let a: f64 = 2.0;
    let b: f64 = 10.0;
    let n: f64 = 8.0;
    let left: f64 = left_riemann(a, b, n);
    println!("{:.17}", left);
    let midpoint: f64 = midpoint_riemann(a, b, n);
    println!("{:.17}", midpoint);
    let right: f64 = right_riemann(a, b, n);
    println!("{:.17}", right);
}

fn square_function(x: f64) -> f64 {
    return x * x;
}

fn left_riemann(a: f64, b: f64, n: f64) -> f64 {
    let diff: f64 = b - a;
    let delta: f64 = diff / n;
    let mut i: f64 = n - 1.0;
    let mut sum: f64 = 0.0;
    while !(i == -1.0) {
        sum += square_function(a + (delta * i));
        i -= 1.0;
    }
    return sum * delta;
}

fn right_riemann(a: f64, b: f64, n: f64) -> f64 {
    let diff: f64 = b - a;
    let delta: f64 = diff / n;
    let mut i: f64 = n;
    let mut sum: f64 = 0.0;
    while !(i == 0.0) {
        sum += square_function(a + (delta * i));
        i -= 1.0;
    }
    return sum * delta;
}

fn midpoint_riemann(a: f64, b: f64, n: f64) -> f64 {
    let diff: f64 = b - a;
    let delta: f64 = diff / n;
    let mut i: f64 = n-1.0;
    let mut sum: f64 = 0.0;
    while !(i == -1.0) {
        sum += square_function(a + ((delta * i) + delta /2.0));
        i -= 1.0;
    }
    return sum * delta;
}
