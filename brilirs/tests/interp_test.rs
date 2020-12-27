extern crate brilirs;

use std::process::Command;

fn run_both_interpreters<T: ToString>(input_file: T) {
  let input_file = input_file.to_string();
  let path = std::path::PathBuf::from(&input_file);
  let filename = path.file_name().unwrap();
  let input: Box<dyn std::io::Read> = Box::new(std::fs::File::open(&input_file).unwrap());
  let mut buf = Vec::new();
  brilirs::run_input(input, &mut buf);
  let brilirs_output = String::from_utf8(buf).unwrap();
  let brilirs_output = brilirs_output.trim();

  let brili_output_obj = Command::new("brili")
    .stdin(std::fs::File::open(input_file).unwrap())
    .output()
    .unwrap();
  let brili_output = String::from_utf8(brili_output_obj.stdout).unwrap();
  let brili_output = brili_output.trim();

  assert!(
    brilirs_output == brili_output,
    "{:?}: brilirs >{}<, brili >>{}<<",
    filename,
    brilirs_output,
    brili_output
  );
}

macro_rules! interp_tests {
    ($($name:ident: $path:expr,)*) => {
        $(
            #[test]
            fn $name() {
                run_both_interpreters($path);
            }
        )*
    }
}

interp_tests! {
    div: "./testdata/div.json",
    float: "./testdata/float.json",
    jmp: "./testdata/jmp.json",
    add: "./testdata/add.json",
    mul: "./testdata/mul.json",
    sub: "./testdata/sub.json",
    eq: "./testdata/eq.json",
    lt: "./testdata/lt.json",
    gt: "./testdata/gt.json",
    le: "./testdata/le.json",
    ge: "./testdata/ge.json",
    not: "./testdata/not.json",
    and: "./testdata/and.json",
    or: "./testdata/or.json",
    id: "./testdata/id.json",
    br: "./testdata/br.json",
    call: "./testdata/call.json",
    call_with_args: "./testdata/call-with-args.json",
}
