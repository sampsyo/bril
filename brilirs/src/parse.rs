use std::error::Error;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use crate::ir_types::Program;

pub fn load(bril_file_path: &Path) -> Result<Program, Box<dyn Error>> {
  let bril_file = File::open(bril_file_path)?;
  let bril_data = BufReader::new(bril_file);
  let bril_prog = serde_json::from_reader(bril_data)?;
  Ok(bril_prog)
}
