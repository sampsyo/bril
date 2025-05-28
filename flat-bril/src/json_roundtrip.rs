use std::fs::File;
use std::io::{self, BufReader, Read};

use crate::flatten;
use crate::unflatten;

/// Does a round trip from JSON -> flattened representation -> back to JSON
/// - `input_json` is the filename of the input JSON (if supplied)
/// - if `verbose = true`, the resultant JSON from the round-trip is
///   printed to `stdout`
pub fn json_roundtrip(input_json: Option<String>, verbose: bool) {
    // Parse the JSON into serde_json's `Value` datatype
    let json: serde_json::Value = if let Some(path) = input_json {
        let file = File::open(path).expect("unable to open JSON file");
        let reader = BufReader::new(file);
        serde_json::from_reader(reader).expect("unable to read from JSON file")
    } else {
        // Read in the JSON representation of a Bril file from stdin
        let mut buffer = String::new();
        io::stdin()
            .read_to_string(&mut buffer)
            .expect("Unable to read from stdin");
        serde_json::from_str(&buffer).expect("Unable to parse malformed JSON")
    };

    let functions = json["functions"]
        .as_array()
        .expect("Expected `functions` to be a JSON array");
    let mut func_json_vec = vec![];
    for func in functions {
        let instr_store = flatten::flatten_instrs(func);
        let func_json = unflatten::unflatten_instrs(&instr_store);
        func_json_vec.push(func_json);
    }
    let prog_json = serde_json::json!({
        "functions": func_json_vec
    });

    if verbose {
        println!("{:#}", prog_json);
    }
}
