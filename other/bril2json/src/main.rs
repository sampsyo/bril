// Copyright (C) 2024 Ethan Uppal.
//
// SPDX-License-Identifier: MIT

use std::{
    fs,
    io::{self},
    path::PathBuf,
};

use annotate_snippets::{Level, Renderer, Snippet};
use argh::FromArgs;
use bril2json::program_to_json;
use bril_frontend::{lexer::Token, loc::Loc, logos::Logos, parser::Parser};
use snafu::{whatever, ResultExt, Whatever};

/// converts Bril's textual representation to its canonical JSON form
#[derive(FromArgs)]
struct Opts {
    /// input Bril file: omit for stdin
    #[argh(positional)]
    input_path: Option<PathBuf>,
}

#[snafu::report]
fn main() -> Result<(), Whatever> {
    let opts = argh::from_env::<Opts>();

    let (input_path_string, mut reader): (String, Box<dyn io::Read>) =
        if let Some(input_path) = opts.input_path {
            let input_path_string = input_path.to_string_lossy().to_string();
            (
                input_path_string.clone(),
                Box::new(
                    fs::File::open(&input_path)
                        .whatever_context(format!("Failed to open {}", input_path_string))?,
                ),
            )
        } else {
            ("<stdin".to_owned(), Box::new(io::stdin()))
        };
    let mut contents = vec![];
    reader
        .read_to_end(&mut contents)
        .whatever_context(format!("Failed to read contents of {}", input_path_string))?;
    let code = String::from_utf8(contents).whatever_context("Couldn't decode file as UTF-8")?;

    let mut lexer = Token::lexer(&code);
    let mut tokens = vec![];
    while let Some(next) = lexer.next() {
        if let Ok(token) = next {
            tokens.push(Loc::new(token, lexer.span()));
        } else {
            whatever!("Failed to lex. Leftover: {}", lexer.remainder());
        }
    }

    let mut parser = Parser::new(&tokens);

    let Ok(program) = parser.parse_program() else {
        let renderer = Renderer::styled();
        for diagnostic in parser.diagnostics() {
            let mut message = Level::Error.title(&diagnostic.message);
            for (text, span) in &diagnostic.labels {
                message = message.snippet(
                    Snippet::source(&code)
                        .origin(&input_path_string)
                        .fold(true)
                        .annotation(
                            Level::Error
                                .span(span.clone().unwrap_or(diagnostic.span.clone()))
                                .label(text.as_str()),
                        ),
                );
            }
            println!("{}", renderer.render(message));
        }
        whatever!("Exiting due to errors");
    };

    let json = program_to_json(&program);
    println!(
        "{}",
        serde_json::to_string_pretty(&json).whatever_context("Failed to pretty-print JSON")?
    );

    Ok(())
}
