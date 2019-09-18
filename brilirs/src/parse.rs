use std::error::Error;

use crate::ir_types::Program;

pub fn load(bril_file_stream: Box<dyn std::io::Read>) -> Result<Program, Box<dyn Error>> {
    let bril_prog = serde_json::from_reader(bril_file_stream)?;
    Ok(bril_prog)
}
