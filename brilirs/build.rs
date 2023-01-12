// https://kbknapp.dev/shell-completions/

use std::io::Error;

#[cfg(feature = "completions")]
include!("src/cli.rs");

fn main() -> Result<(), Error> {
  #[cfg(feature = "completions")]
  create_cli_completions()?;

  Ok(())
}

#[cfg(feature = "completions")]
fn create_cli_completions() -> Result<(), Error> {
  use clap::CommandFactory;
  use clap_complete::{
    shells::{Bash, Elvish, Fish, PowerShell, Zsh},
    Generator,
  };
  use std::{env, path::PathBuf};
  // Waiting on https://github.com/rust-lang/cargo/issues/5457 / https://github.com/rust-lang/cargo/issues/6790 to clean this up
  let out_dir = match env::var_os("OUT_DIR") {
    None => {
      println!("cargo:warning=Did not find out dir",);
      return Ok(());
    }
    Some(out_dir) => out_dir,
  };

  let mut app = Cli::command();
  app.set_bin_name("brilirs");

  let bin_name = app.get_name().to_string();

  let shell: Box<dyn Generator> = match env::var("SHELL") {
    Ok(s) if s.contains("bash") => Box::new(Bash),
    Ok(s) if s.contains("fish") => Box::new(Fish),
    Ok(s) if s.contains("zsh") => Box::new(Zsh),
    Ok(s) if s.contains("elvish") => Box::new(Elvish),
    Ok(s) if s.contains("powershell") => Box::new(PowerShell),
    Ok(_) | Err(_) => {
      println!(
        "cargo:warning=Your shell could not be detected from the $SHELL environment variable so no shell completions were generated. Check the build.rs file if you want to see how this was generated.",
      );
      println!("cargo:warning=Raise an issue if this doesn't work for you",);
      return Ok(());
    }
  };

  let mut path = PathBuf::from(out_dir);
  path.set_file_name(shell.file_name(&bin_name));
  // Check if tab completions file already exists and return if so
  if path.is_file() {
    return Ok(());
  }

  // This is an attempt at being smart. Instead, one could just generate completion scripts for all of the shells in a completions/ directory and have the user choose the appropriate one.
  shell.generate(&app, &mut std::fs::File::create(path.clone())?);

  println!(
    "cargo:warning={} completion file is generated: {path:?}",
    app.get_name()
  );
  println!("cargo:warning=enable this by running `source {path:?}`");
  Ok(())
}
