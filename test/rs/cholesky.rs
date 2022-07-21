fn fillarray() -> [f64; 16] {
    return [
        34.0, 28.0, 38.0, 29.0, 26.0, 78.0, 91.0, 83.0, 67.0, 92.0, 56.0, 92.0, 67.0, 826.0, 38.0,
        43.0,
    ];
}

fn print_array(size: i64, arr: &[f64]) {
    let mut i: i64 = 0;
    while i < size {
        let val: f64 = arr[i as usize];
        println!("{:.17}", val);
        i += 1
    }
}

fn matmul(size: i64, arr1: &[f64], arr2: &[f64], dest: &mut [f64]) {
    let mut row: i64 = 0;
    while row < size {
        let mut col: i64 = 0;
        while col < size {
            let mut sum: f64 = 0.0;
            let mut i: i64 = 0;
            while i < size {
                sum += arr1[((row * size) + i) as usize] * arr2[((i * size) + col) as usize];
                i += 1;
            }
            dest[((row * size) + col) as usize] = sum;
            col += 1;
        }
        row += 1;
    }
}

fn transpose(size: i64, input: &[f64], output: &mut [f64]) {
    let mut row: i64 = 0;
    while row < size {
        let mut col: i64 = 0;
        while col < size {
            output[((col * size) + row) as usize] = input[((row * size) + col) as usize];
            col += 1;
        }
        row += 1;
    }
}

fn sqrt(input: f64) -> f64 {
    let n: f64 = input;
    let precision: f64 = 0.00001;
    let mut x: f64 = input;
    let mut notdone: bool = true;
    while notdone {
        let root: f64 = 0.5 * (x + (n / x));
        let mut diff: f64 = root - x;
        if diff < 0.0 {
            diff = -diff;
        }

        if (diff < precision) {
            notdone = false;
        }

        x = root;
    }
    return x;
}

fn cholesky(size: i64, arr1: &mut [f64], arr2: &mut [f64]) {
    let mut i: i64 = 0;
    while (i < size) {
        let mut j: i64 = 0;
        while (j <= i) {
            let mut k: i64 = 0;
            while (k < j) {
                // prepare indices
                let ik_index: i64 = (i * size) + k;

                let jk_index: i64 = (j * size) + k;

                let ij_index: i64 = (i * size) + j;

                // load values
                let b_ik: f64 = arr2[(ik_index) as usize];

                let b_jk: f64 = arr2[(jk_index) as usize];

                let a_ij: f64 = arr1[(ij_index) as usize];

                let value: f64 = a_ij - (b_ik * b_jk);
                arr1[(ij_index) as usize] = value;

                k += 1;
            }

            let ij_index: i64 = (i * size) + j;
            let jj_index: i64 = (j * size) + j;

            arr2[(ij_index) as usize] = (arr1[(ij_index) as usize] / arr2[(jj_index) as usize]);

            j += 1;
        }
        let index: i64 = (i * size) + i;
        arr2[index as usize] = sqrt(arr1[index as usize]);

        i += 1;
    }
    return;
}

fn main() {
    let size: i64 = 4;
    let arr1: [f64; 16] = fillarray();
    let mut arr1_transposed: [f64; 16] = fillarray();
    let mut hermitian: [f64; 16] = fillarray();
    let mut res: [f64; 16] = [0.0; 16];
    transpose(size, &arr1, &mut arr1_transposed);
    matmul(size, &arr1, &arr1_transposed, &mut hermitian);
    cholesky(size, &mut hermitian, &mut res);
    print_array(16, &res);
    drop(arr1);
    drop(arr1_transposed);
    drop(hermitian);
    drop(res);
    return;
}
