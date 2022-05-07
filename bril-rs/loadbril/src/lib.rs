#![warn(clippy::all, clippy::pedantic, clippy::nursery, clippy::cargo)]
#![warn(missing_docs)]
#![doc = include_str!("../README.md")]

#[doc(hidden)]
pub mod cli;

use std::collections::HashMap;
use std::ffi::OsStr;
use std::fs::{canonicalize, File};
use std::path::{Path, PathBuf};

use bril2json::parse_abstract_program_from_read;
use bril_rs::{
    load_abstract_program_from_read, AbstractCode, AbstractFunction, AbstractInstruction,
    AbstractProgram, ImportedName,
};

fn mangle_name(path: &Path, func_name: &String) -> String {
    let mut parts = path.components();
    parts.next();

    let mut p: Vec<_> = parts
        .into_iter()
        .map(|c| c.as_os_str().to_str().unwrap())
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
                .map(|f| name_resolution_map.get(&f).unwrap().clone())
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
                .map(|f| name_resolution_map.get(&f).unwrap().clone())
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

pub fn locate_imports(
    path_map: &mut HashMap<PathBuf, Option<AbstractProgram>>,
    canonical_path: &PathBuf,
    is_toplevel: bool,
) -> std::io::Result<()> {
    // Check whether we've seen this path
    if path_map.contains_key(canonical_path) {
        return Ok(());
    } else {
        path_map.insert(canonical_path.clone(), None);
    }

    // Find the correct parser for this path based on the extension
    let f = match canonical_path.extension().and_then(|s: &OsStr| s.to_str()) {
        Some("bril") => |s| parse_abstract_program_from_read(s, true),
        Some("json") => load_abstract_program_from_read,
        Some(_) | None => {
            panic!("Don't know how to handle a file without an extension like *.bril/*.json");
        }
    };

    // Get the AbstractProgram representation of the file
    let program = f(File::open(&canonical_path)?);

    let mut name_resolution_map = HashMap::new();

    // Get the mangled names of functions declared in the current file
    program
        .functions
        .iter()
        .for_each(|AbstractFunction { name, .. }| {
            if name_resolution_map
                .insert(name.clone(), mangle_name(&canonical_path, name))
                .is_some()
            {
                panic!()
            }
        });

    // Locate any imports in the current program
    let parent = canonical_path.parent().unwrap();
    program
        .imports
        .iter()
        .try_for_each::<_, std::io::Result<()>>(|i| {
            let mut next_path = i.path.clone();
            if next_path.is_relative() {
                next_path = parent.join(next_path)
            }
            next_path = canonicalize(next_path)?;
            locate_imports(path_map, &next_path, false)?;

            i.names.iter().for_each(|ImportedName { name, alias }| {
                if name_resolution_map
                    .insert(
                        alias.as_ref().unwrap_or(name).clone(),
                        mangle_name(&next_path, name),
                    )
                    .is_some()
                {
                    panic!()
                }
            });
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
    path_map
        .insert(canonical_path.clone(), Some(mangled_program))
        .unwrap();
    Ok(())
}
