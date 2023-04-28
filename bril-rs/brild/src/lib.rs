#![warn(clippy::all, clippy::pedantic, clippy::nursery, clippy::cargo)]
#![warn(missing_docs)]
#![doc = include_str!("../README.md")]
#![allow(clippy::module_name_repetitions)]

#[doc(hidden)]
pub mod cli;

#[doc(hidden)]
pub mod error;

use std::collections::HashMap;
use std::fs::{canonicalize, File};
use std::hash::BuildHasher;
use std::path::{Path, PathBuf};

use bril2json::parse_abstract_program_from_read;
use bril_rs::{
    load_abstract_program_from_read, AbstractCode, AbstractFunction, AbstractInstruction,
    AbstractProgram, ImportedFunction,
};

use crate::error::BrildError;

fn mangle_name(path: &Path, func_name: &String) -> String {
    let mut parts = path.components();
    parts.next();

    let mut p: Vec<_> = parts
        .map(|c| {
            c.as_os_str().to_str().expect(
                "Panics if the path does not contain valid unicode which I'm not worried about",
            )
        })
        .collect();
    p.push(func_name);

    p.join("___")
}

fn mangle_instr(code: AbstractCode, name_resolution_map: &HashMap<String, String>) -> AbstractCode {
    match code {
        AbstractCode::Instruction(AbstractInstruction::Value {
            op,
            funcs,
            args,
            dest,
            labels,
            pos,
            op_type,
        }) => AbstractCode::Instruction(AbstractInstruction::Value {
            op,
            funcs: funcs
                .into_iter()
                .map(|f| {
                    name_resolution_map
                        .get(&f)
                        .expect("Could not find name for {f}")
                        .clone()
                })
                .collect(),
            args,
            dest,
            labels,
            pos,
            op_type,
        }),
        AbstractCode::Instruction(AbstractInstruction::Effect {
            op,
            funcs,
            args,
            labels,
            pos,
        }) => AbstractCode::Instruction(AbstractInstruction::Effect {
            op,
            funcs: funcs
                .into_iter()
                .map(|f| {
                    name_resolution_map
                        .get(&f)
                        .expect("Could not find name for {f}")
                        .clone()
                })
                .collect(),
            args,
            labels,
            pos,
        }),
        _ => code,
    }
}

fn mangle_function(
    AbstractFunction {
        name,
        args,
        instrs,
        pos,
        return_type,
    }: AbstractFunction,
    name_resolution_map: &HashMap<String, String>,
    is_toplevel: bool,
) -> AbstractFunction {
    AbstractFunction {
        name: if is_toplevel && name == "main" {
            name
        } else {
            name_resolution_map.get(&name).unwrap().to_string()
        },
        args,
        instrs: instrs
            .into_iter()
            .map(|i| mangle_instr(i, name_resolution_map))
            .collect(),
        pos,
        return_type,
    }
}

#[doc(hidden)]
pub fn handle_program<S: BuildHasher>(
    path_map: &mut HashMap<PathBuf, Option<AbstractProgram>, S>,
    program: AbstractProgram,
    canonical_path: &Path,
    libs: &[PathBuf],
    is_toplevel: bool,
) -> Result<(), BrildError> {
    let mut name_resolution_map = HashMap::new();

    // Get the mangled names of functions declared in the current file
    program
        .functions
        .iter()
        .try_for_each(|AbstractFunction { name, .. }| {
            if name_resolution_map
                .insert(name.clone(), mangle_name(canonical_path, name))
                .is_some()
            {
                // Error if the same function is declared twice
                Err(BrildError::DuplicateFunction(name.clone()))
            } else {
                Ok(())
            }
        })?;

    // Locate any imports in the current program
    program
        .imports
        .iter()
        .try_for_each::<_, Result<(), BrildError>>(|i| {
            let next_path = locate_import(path_map, &i.path, libs, false)?;

            i.functions
                .iter()
                .try_for_each(|ImportedFunction { name, alias }| {
                    if name_resolution_map
                        .insert(
                            alias.as_ref().unwrap_or(name).clone(),
                            mangle_name(&next_path, name),
                        )
                        .is_some()
                    {
                        Err(BrildError::DuplicateFunction(name.clone()))
                    } else {
                        Ok(())
                    }
                })?;
            Ok(())
        })?;

    // Do mangling
    let mangled_program = AbstractProgram {
        imports: Vec::new(),
        functions: program
            .functions
            .into_iter()
            .map(|f| mangle_function(f, &name_resolution_map, is_toplevel))
            .collect(),
    };

    // Add the mangled program back to the map
    path_map.insert(canonical_path.to_path_buf(), Some(mangled_program));

    Ok(())
}

// Adds the mangled, imported `canonical_path` to the path_map
// Path is assumed to exist(caller checked)
// `libs` is the path to check for locating Bril programs
// If it's the toplevel bril file, the `main` function will not be mangled
#[doc(hidden)]
pub fn do_import<S: BuildHasher>(
    path_map: &mut HashMap<PathBuf, Option<AbstractProgram>, S>,
    canonical_path: &PathBuf,
    libs: &[PathBuf],
    is_toplevel: bool,
) -> Result<(), BrildError> {
    // Check whether we've seen this path
    if path_map.contains_key(canonical_path) {
        return Ok(());
    }

    path_map.insert(canonical_path.clone(), None);

    // Find the correct parser for this path based on the extension
    let f: Box<dyn Fn(_) -> AbstractProgram> =
        match canonical_path.extension().and_then(std::ffi::OsStr::to_str) {
            Some("bril") => Box::new(|s| {
                parse_abstract_program_from_read(
                    s,
                    true,
                    true,
                    Some(canonical_path.display().to_string()),
                )
            }),
            Some("json") => Box::new(load_abstract_program_from_read),
            Some(_) | None => {
                return Err(BrildError::MissingOrUnknownFileExtension(
                    canonical_path.clone(),
                ))
            }
        };

    // Get the AbstractProgram representation of the file
    let program = f(File::open(canonical_path)?);

    handle_program(path_map, program, canonical_path, libs, is_toplevel)?;
    Ok(())
}

// Finds the correct full path for `path` by adding it to each of the lib paths till it gets a hit
#[doc(hidden)]
fn locate_import<S: BuildHasher>(
    path_map: &mut HashMap<PathBuf, Option<AbstractProgram>, S>,
    path: &PathBuf,
    libs: &[PathBuf],
    is_toplevel: bool,
) -> Result<PathBuf, BrildError> {
    let located_libs: Vec<_> = libs.iter().filter(|lib| lib.join(path).exists()).collect();

    if located_libs.is_empty() {
        return Err(BrildError::NoPathExists(path.clone()));
    }

    if located_libs.len() > 1 {
        eprintln!("Warning, more than one valid path for {path:?} was found, using the first one.");
    }

    let next_path = canonicalize(located_libs.get(0).unwrap().join(path))?;

    do_import(path_map, &next_path, libs, is_toplevel)?;

    Ok(next_path)
}
