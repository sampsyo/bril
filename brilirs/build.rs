// https://kbknapp.dev/shell-completions/

use clap::IntoApp;
use clap_complete::{
  generate_to,
  shells::{Bash, Elvish, Fish, PowerShell, Zsh},
};
use std::io::Error;
use std::{env, path::PathBuf};

include!("src/cli.rs");

fn main() -> Result<(), Error> {
  let out_dir = match env::var_os("CARGO_MANIFEST_DIR") {
    None => return Ok(()),
    Some(out_dir) => out_dir,
  };

  let mut app = Cli::into_app();
  let bin_name = app.get_name().to_string();

  // This is an attempt at being smart. Instead, one could just generate completion scripts for all of the shells in a completions/ directory and have the user choose the appropriate one.
  let path = match env!("SHELL") {
    s if s.contains("bash") => generate_to(Bash, &mut app, bin_name, out_dir)?,
    s if s.contains("fish") => generate_to(Fish, &mut app, bin_name, out_dir)?,
    s if s.contains("zsh") => generate_to(Zsh, &mut app, bin_name, out_dir)?,
    s if s.contains("elvish") => generate_to(Elvish, &mut app, bin_name, out_dir)?,
    s if s.contains("powershell") => generate_to(PowerShell, &mut app, bin_name, out_dir)?,
    _ => {
      let mut x = PathBuf::new();
      x.push("Your shell could not be detected from the $SHELL environment variable so no shell completions were generated. Check the build.rs file if you want to see how this was generated.");
      x
    }
  };

  println!(
    "cargo:warning={} completion file is generated: {:?}",
    app.get_name(),
    path
  );
  println!("cargo:warning=enable this by running `source {:?}`", path);
  Ok(())
}
