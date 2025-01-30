use thiserror::Error;

#[expect(
    clippy::module_name_repetitions,
    reason = "I allow for the Error suffix on enums"
)]
#[derive(Error, Debug)]
pub enum BrildError {
    #[error("Could not find a complete path for `{0}` from the list of provided libraries")]
    NoPathExists(std::path::PathBuf),
    #[error("Imported file is missing or has an unknown a file extension: `{0}`")]
    MissingOrUnknownFileExtension(std::path::PathBuf),
    #[error("Function `{0}` declared more than once")]
    DuplicateFunction(String),
    #[error(transparent)]
    IoError(#[from] std::io::Error),
}
