use std::collections::HashMap;
use std::error::Error;
use twox_hash::RandomXxHashBuilder64;

use crate::ir_types::{BrArgs, Function, Identifier, Instruction, Operation, Program, ValueOp};

pub fn load(bril_file_stream: Box<dyn std::io::Read>) -> Result<Program<String>, Box<dyn Error>> {
  let bril_prog = serde_json::from_reader(bril_file_stream)?;
  Ok(bril_prog)
}

pub fn convert_identifiers(prog: Program<String>) -> (Program<usize>, usize) {
  let mut next_id = 0;
  let mut id_map: HashMap<String, usize, RandomXxHashBuilder64> = Default::default();
  let mut get_id_idx = |ident| {
    Identifier::<usize>(if id_map.contains_key(&ident) {
      *id_map.get(&ident).unwrap()
    } else {
      id_map.insert(ident, next_id);
      next_id += 1;
      next_id - 1
    })
  };

  let mut id_replacer = |op: Operation<String>| -> Operation<usize> {
    let mut replace_value_ids = |params: ValueOp<String>| -> ValueOp<usize> {
      ValueOp::<usize> {
        dest: get_id_idx(params.dest.0),
        typ: params.typ,
        args: params
          .args
          .into_iter()
          .map(|arg| get_id_idx(arg.0))
          .collect(),
      }
    };

    use crate::ir_types::Operation::*;
    match op {
      Const { dest, typ, value } => Const {
        dest: get_id_idx(dest.0),
        typ: typ,
        value: value,
      },
      // TODO: Is there a less duplicated way to do these branches for the ValueOps?
      Add { params } => Add {
        params: replace_value_ids(params),
      },
      Sub { params } => Sub {
        params: replace_value_ids(params),
      },
      Mul { params } => Mul {
        params: replace_value_ids(params),
      },
      Div { params } => Div {
        params: replace_value_ids(params),
      },
      Le { params } => Le {
        params: replace_value_ids(params),
      },
      Lt { params } => Lt {
        params: replace_value_ids(params),
      },
      Gt { params } => Gt {
        params: replace_value_ids(params),
      },
      Ge { params } => Ge {
        params: replace_value_ids(params),
      },
      Eq { params } => Eq {
        params: replace_value_ids(params),
      },
      Not { params } => Not {
        params: replace_value_ids(params),
      },
      And { params } => And {
        params: replace_value_ids(params),
      },
      Or { params } => Or {
        params: replace_value_ids(params),
      },
      Br {
        params: BrArgs::StringArgs { test_var, dests },
      } => Br {
        params: BrArgs::IdArgs {
          test_var: get_id_idx(test_var.0),
          dests: dests,
        },
      },
      Id { params } => Id {
        params: replace_value_ids(params),
      },
      Print { args } => Print {
        args: args.into_iter().map(|arg| get_id_idx(arg.0)).collect(),
      },
      // TODO: Similarly, can we reduce duplication with these EffectOp branches?
      Jmp { params } => Jmp { params: params },
      Ret { params } => Ret { params: params },
      Nop => Nop,
      Br {
        params: BrArgs::IdArgs { .. },
      } => unreachable!(),
    }
  };

  let result = Program::<usize> {
    functions: prog
      .functions
      .into_iter()
      .map(|func| Function::<usize> {
        name: func.name,
        instrs: func
          .instrs
          .into_iter()
          .map(|instr| match instr {
            Instruction::Label(l) => Instruction::Label(l),
            Instruction::Operation(op) => Instruction::Operation(id_replacer(op)),
          })
          .collect(),
      })
      .collect(),
  };

  (result, next_id)
}
