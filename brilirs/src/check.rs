use crate::error::{InterpError, PositionalInterpError};
use bril_rs::{Code, ConstOps, EffectOps, Function, Instruction, Program, Type, ValueOps};

use fxhash::FxHashMap;

const fn check_num_args(expected: usize, args: &[String]) -> Result<(), InterpError> {
  if expected == args.len() {
    Ok(())
  } else {
    Err(InterpError::BadNumArgs(expected, args.len()))
  }
}

const fn check_num_funcs(expected: usize, funcs: &[String]) -> Result<(), InterpError> {
  if expected == funcs.len() {
    Ok(())
  } else {
    Err(InterpError::BadNumFuncs(expected, funcs.len()))
  }
}

const fn check_num_labels(expected: usize, labels: &[String]) -> Result<(), InterpError> {
  if expected == labels.len() {
    Ok(())
  } else {
    Err(InterpError::BadNumLabels(expected, labels.len()))
  }
}

fn check_asmt_type(expected: &bril_rs::Type, actual: &bril_rs::Type) -> Result<(), InterpError> {
  if expected == actual || expected == &bril_rs::Type::Any || actual == &bril_rs::Type::Any {
    Ok(())
  } else {
    Err(InterpError::BadAsmtType(expected.clone(), actual.clone()))
  }
}

fn update_env<'a>(
  env: &mut FxHashMap<&'a str, Type>,
  dest: &'a str,
  typ: &Type,
) -> Result<(), InterpError> {
  if typ == &Type::Any {
    Ok(())
  } else {
    let current_typ = env.get(dest).unwrap();
    if matches!(current_typ, Type::Any) {
      env.insert(dest, typ.clone());
      Ok(())
    } else {
      // If the current type is not Any, we check if the types match
      check_asmt_type(current_typ, typ)
    }
  }
}

fn get_type<'a>(
  env: &'a FxHashMap<&'a str, Type>,
  index: usize,
  args: &[String],
) -> Result<&'a Type, InterpError> {
  if index >= args.len() {
    return Err(InterpError::BadNumArgs(index, args.len()));
  }

  env
    .get(&args[index] as &str)
    .ok_or_else(|| InterpError::VarUndefined(args[index].to_string()))
}

fn get_ptr_type(typ: &bril_rs::Type) -> Result<&bril_rs::Type, InterpError> {
  match typ {
    bril_rs::Type::Pointer(ptr_type) => Ok(ptr_type),
    _ => Err(InterpError::ExpectedPointerType(typ.clone())),
  }
}

fn type_check_instruction<'a>(
  instr: &'a Instruction,
  func: &Function,
  prog: &Program,
  env: &mut FxHashMap<&'a str, Type>,
) -> Result<(), InterpError> {
  match instr {
    Instruction::Constant {
      op: ConstOps::Const,
      dest,
      const_type,
      value,
      pos: _,
    } => {
      if const_type == &Type::Float && value.get_type() == Type::Int {
        update_env(env, dest, const_type)
      } else {
        check_asmt_type(const_type, &value.get_type())?;
        update_env(env, dest, &value.get_type())
      }
    }
    Instruction::Value {
      op: ValueOps::Add | ValueOps::Sub | ValueOps::Mul | ValueOps::Div,
      dest,
      op_type,
      args,
      funcs,
      labels,
      pos: _,
    } => {
      check_num_args(2, args)?;
      check_num_funcs(0, funcs)?;
      check_num_labels(0, labels)?;
      check_asmt_type(&Type::Int, get_type(env, 0, args)?)?;
      check_asmt_type(&Type::Int, get_type(env, 1, args)?)?;
      check_asmt_type(&Type::Int, op_type)?;
      update_env(env, dest, op_type)
    }
    Instruction::Value {
      op: ValueOps::Eq | ValueOps::Lt | ValueOps::Gt | ValueOps::Le | ValueOps::Ge,
      dest,
      op_type,
      args,
      funcs,
      labels,
      pos: _,
    } => {
      check_num_args(2, args)?;
      check_num_funcs(0, funcs)?;
      check_num_labels(0, labels)?;
      check_asmt_type(&Type::Int, get_type(env, 0, args)?)?;
      check_asmt_type(&Type::Int, get_type(env, 1, args)?)?;
      check_asmt_type(&Type::Bool, op_type)?;
      update_env(env, dest, op_type)
    }
    Instruction::Value {
      op: ValueOps::Not,
      dest,
      op_type,
      args,
      funcs,
      labels,
      pos: _,
    } => {
      check_num_args(1, args)?;
      check_num_funcs(0, funcs)?;
      check_num_labels(0, labels)?;
      check_asmt_type(&Type::Bool, get_type(env, 0, args)?)?;
      check_asmt_type(&Type::Bool, op_type)?;
      update_env(env, dest, op_type)
    }
    Instruction::Value {
      op: ValueOps::And | ValueOps::Or,
      dest,
      op_type,
      args,
      funcs,
      labels,
      pos: _,
    } => {
      check_num_args(2, args)?;
      check_num_funcs(0, funcs)?;
      check_num_labels(0, labels)?;
      check_asmt_type(&Type::Bool, get_type(env, 0, args)?)?;
      check_asmt_type(&Type::Bool, get_type(env, 1, args)?)?;
      check_asmt_type(&Type::Bool, op_type)?;
      update_env(env, dest, op_type)
    }
    Instruction::Value {
      op: ValueOps::Id,
      dest,
      op_type,
      args,
      funcs,
      labels,
      pos: _,
    } => {
      check_num_args(1, args)?;
      check_num_funcs(0, funcs)?;
      check_num_labels(0, labels)?;
      check_asmt_type(op_type, get_type(env, 0, args)?)?;
      update_env(env, dest, op_type)
    }
    Instruction::Value {
      op: ValueOps::Fadd | ValueOps::Fsub | ValueOps::Fmul | ValueOps::Fdiv,
      dest,
      op_type,
      args,
      funcs,
      labels,
      pos: _,
    } => {
      check_num_args(2, args)?;
      check_num_funcs(0, funcs)?;
      check_num_labels(0, labels)?;
      check_asmt_type(&Type::Float, get_type(env, 0, args)?)?;
      check_asmt_type(&Type::Float, get_type(env, 1, args)?)?;
      check_asmt_type(&Type::Float, op_type)?;
      update_env(env, dest, op_type)
    }
    Instruction::Value {
      op: ValueOps::Feq | ValueOps::Flt | ValueOps::Fgt | ValueOps::Fle | ValueOps::Fge,
      dest,
      op_type,
      args,
      funcs,
      labels,
      pos: _,
    } => {
      check_num_args(2, args)?;
      check_num_funcs(0, funcs)?;
      check_num_labels(0, labels)?;
      check_asmt_type(&Type::Float, get_type(env, 0, args)?)?;
      check_asmt_type(&Type::Float, get_type(env, 1, args)?)?;
      check_asmt_type(&Type::Bool, op_type)?;
      update_env(env, dest, op_type)
    }
    Instruction::Value {
      op: ValueOps::Ceq | ValueOps::Cge | ValueOps::Clt | ValueOps::Cgt | ValueOps::Cle,
      args,
      dest,
      funcs,
      labels,
      pos: _,
      op_type,
    } => {
      check_num_args(2, args)?;
      check_num_funcs(0, funcs)?;
      check_num_labels(0, labels)?;
      check_asmt_type(&Type::Char, get_type(env, 0, args)?)?;
      check_asmt_type(&Type::Char, get_type(env, 1, args)?)?;
      check_asmt_type(&Type::Bool, op_type)?;
      update_env(env, dest, op_type)
    }
    Instruction::Value {
      op: ValueOps::Char2int,
      args,
      dest,
      funcs,
      labels,
      pos: _,
      op_type,
    } => {
      check_num_args(1, args)?;
      check_num_funcs(0, funcs)?;
      check_num_labels(0, labels)?;
      check_asmt_type(&Type::Char, get_type(env, 0, args)?)?;
      check_asmt_type(&Type::Int, op_type)?;
      update_env(env, dest, op_type)
    }
    Instruction::Value {
      op: ValueOps::Int2char,
      args,
      dest,
      funcs,
      labels,
      pos: _,
      op_type,
    } => {
      check_num_args(1, args)?;
      check_num_funcs(0, funcs)?;
      check_num_labels(0, labels)?;
      check_asmt_type(&Type::Int, get_type(env, 0, args)?)?;
      check_asmt_type(&Type::Char, op_type)?;
      update_env(env, dest, op_type)
    }
    Instruction::Value {
      op: ValueOps::Bits2Float,
      args,
      dest,
      funcs,
      labels,
      pos: _,
      op_type,
    } => {
      check_num_args(1, args)?;
      check_num_funcs(0, funcs)?;
      check_num_labels(0, labels)?;
      check_asmt_type(get_type(env, 0, args)?, &Type::Int)?;
      check_asmt_type(op_type, &Type::Float)?;
      update_env(env, dest, op_type)
    }
    Instruction::Value {
      op: ValueOps::Float2Bits,
      args,
      dest,
      funcs,
      labels,
      pos: _,
      op_type,
    } => {
      check_num_args(1, args)?;
      check_num_funcs(0, funcs)?;
      check_num_labels(0, labels)?;
      check_asmt_type(get_type(env, 0, args)?, &Type::Float)?;
      check_asmt_type(op_type, &Type::Int)?;
      update_env(env, dest, op_type)
    }
    Instruction::Value {
      op: ValueOps::Call,
      dest,
      op_type,
      args,
      funcs,
      labels,
      pos: _,
    } => {
      check_num_funcs(1, funcs)?;
      check_num_labels(0, labels)?;
      let callee_func = prog.functions.iter().find(|f| f.name == funcs[0]).unwrap();

      if args.len() != callee_func.args.len() {
        return Err(InterpError::BadNumArgs(callee_func.args.len(), args.len()));
      }
      args
        .iter()
        .zip(callee_func.args.iter())
        .try_for_each(|(arg_name, expected_arg)| {
          let ty = env
            .get(arg_name as &str)
            .ok_or_else(|| InterpError::VarUndefined(arg_name.to_string()))?;

          check_asmt_type(ty, &expected_arg.arg_type)
        })?;

      callee_func.return_type.as_ref().map_or_else(
        || Err(InterpError::NonEmptyRetForFunc(callee_func.name.clone())),
        |t| check_asmt_type(op_type, t),
      )?;

      update_env(env, dest, op_type)
    }
    Instruction::Value {
      op: ValueOps::Get | ValueOps::Undef,
      dest,
      op_type,
      args,
      funcs,
      labels,
      pos: _,
    } => {
      check_num_args(0, args)?;
      check_num_funcs(0, funcs)?;
      check_num_labels(0, labels)?;

      update_env(env, dest, op_type)
    }
    Instruction::Value {
      op: ValueOps::Alloc,
      dest,
      op_type,
      args,
      funcs,
      labels,
      pos: _,
    } => {
      check_num_args(1, args)?;
      check_num_funcs(0, funcs)?;
      check_num_labels(0, labels)?;
      check_asmt_type(&Type::Int, get_type(env, 0, args)?)?;
      get_ptr_type(op_type)?;
      update_env(env, dest, op_type)
    }
    Instruction::Value {
      op: ValueOps::Load,
      dest,
      op_type,
      args,
      funcs,
      labels,
      pos: _,
    } => {
      check_num_args(1, args)?;
      check_num_funcs(0, funcs)?;
      check_num_labels(0, labels)?;
      let ptr_type = get_ptr_type(get_type(env, 0, args)?)?;
      check_asmt_type(ptr_type, op_type)?;
      update_env(env, dest, op_type)
    }
    Instruction::Value {
      op: ValueOps::PtrAdd,
      dest,
      op_type,
      args,
      funcs,
      labels,
      pos: _,
    } => {
      check_num_args(2, args)?;
      check_num_funcs(0, funcs)?;
      check_num_labels(0, labels)?;
      let ty0 = get_type(env, 0, args)?;
      get_ptr_type(ty0)?;
      check_asmt_type(&Type::Int, get_type(env, 1, args)?)?;
      check_asmt_type(ty0, op_type)?;
      update_env(env, dest, op_type)
    }
    Instruction::Effect {
      op: EffectOps::Jump,
      args,
      funcs,
      labels,
      pos: _,
    } => {
      check_num_args(0, args)?;
      check_num_funcs(0, funcs)?;
      check_num_labels(1, labels)?;
      Ok(())
    }
    Instruction::Effect {
      op: EffectOps::Branch,
      args,
      funcs,
      labels,
      pos: _,
    } => {
      check_num_args(1, args)?;
      check_asmt_type(&Type::Bool, get_type(env, 0, args)?)?;
      check_num_funcs(0, funcs)?;
      check_num_labels(2, labels)?;
      Ok(())
    }
    Instruction::Effect {
      op: EffectOps::Return,
      args,
      funcs,
      labels,
      pos: _,
    } => {
      check_num_funcs(0, funcs)?;
      check_num_labels(0, labels)?;
      match &func.return_type {
        Some(t) => {
          check_num_args(1, args)?;
          let ty0 = get_type(env, 0, args)?;
          check_asmt_type(t, ty0)
        }
        None => {
          if args.is_empty() {
            Ok(())
          } else {
            Err(InterpError::NonEmptyRetForFunc(func.name.clone()))
          }
        }
      }
    }
    Instruction::Effect {
      op: EffectOps::Print,
      args,
      funcs,
      labels,
      pos: _,
    } => {
      check_num_funcs(0, funcs)?;
      check_num_labels(0, labels)?;
      args.iter().enumerate().try_for_each(|(i, _)| {
        get_type(env, i, args)?;
        Ok(())
      })
    }
    Instruction::Effect {
      op: EffectOps::Nop,
      args,
      funcs,
      labels,
      pos: _,
    } => {
      check_num_args(0, args)?;
      check_num_funcs(0, funcs)?;
      check_num_labels(0, labels)?;
      Ok(())
    }
    Instruction::Effect {
      op: EffectOps::Call,
      args,
      funcs,
      labels,
      pos: _,
    } => {
      check_num_funcs(1, funcs)?;
      check_num_labels(0, labels)?;
      let callee_func = prog
        .functions
        .iter()
        .find(|f| f.name == funcs[0])
        .ok_or(InterpError::FuncNotFound(funcs[0].clone()))?;

      if args.len() != callee_func.args.len() {
        return Err(InterpError::BadNumArgs(callee_func.args.len(), args.len()));
      }
      args
        .iter()
        .zip(callee_func.args.iter())
        .try_for_each(|(arg_name, expected_arg)| {
          let ty = env
            .get(arg_name as &str)
            .ok_or_else(|| InterpError::VarUndefined(arg_name.to_string()))?;

          check_asmt_type(ty, &expected_arg.arg_type)
        })?;

      if callee_func.return_type.is_some() {
        Err(InterpError::NonEmptyRetForFunc(callee_func.name.clone()))
      } else {
        Ok(())
      }
    }
    Instruction::Effect {
      op: EffectOps::Store,
      args,
      funcs,
      labels,
      pos: _,
    } => {
      check_num_args(2, args)?;
      check_num_funcs(0, funcs)?;
      check_num_labels(0, labels)?;
      let ty0 = get_type(env, 0, args)?;
      let ty1 = get_type(env, 1, args)?;
      check_asmt_type(get_ptr_type(ty0)?, ty1)
    }
    Instruction::Effect {
      op: EffectOps::Free,
      args,
      funcs,
      labels,
      pos: _,
    } => {
      check_num_args(1, args)?;
      check_num_funcs(0, funcs)?;
      check_num_labels(0, labels)?;
      get_ptr_type(get_type(env, 0, args)?)?;
      Ok(())
    }
    Instruction::Effect {
      op: EffectOps::Set,
      args,
      funcs,
      labels,
      pos: _,
    } => {
      check_num_args(2, args)?;
      check_num_funcs(0, funcs)?;
      check_num_labels(0, labels)?;
      let ty1 = get_type(env, 1, args)?.clone();
      update_env(env, &args[0], &ty1)
    }
    Instruction::Effect {
      op: EffectOps::Speculate | EffectOps::Guard | EffectOps::Commit,
      args: _,
      funcs: _,
      labels: _,
      pos: _,
    } => {
      unimplemented!("brilirs does not currently support the speculative execution")
    }
  }
}

fn type_check_func(func: &Function, prog: &Program) -> Result<(), PositionalInterpError> {
  if func.name == "main" && func.return_type.is_some() {
    return Err(InterpError::NonEmptyRetForFunc(func.name.clone()))
      .map_err(|e| e.add_pos(func.pos.clone()));
  }

  let mut env: FxHashMap<&str, Type> =
    FxHashMap::with_capacity_and_hasher(20, fxhash::FxBuildHasher::default());
  func.args.iter().for_each(|a| {
    env.insert(&a.name, a.arg_type.clone());
  });

  func.instrs.iter().try_for_each(|a| match a {
    Code::Label { .. } | Code::Instruction(Instruction::Effect { .. }) => Ok(()),
    Code::Instruction(
      Instruction::Constant {
        dest,
        const_type: op_type,
        pos,
        ..
      }
      | Instruction::Value {
        dest, op_type, pos, ..
      },
    ) => env.insert(dest, op_type.clone()).map_or_else(
      || Ok(()),
      |t| {
        if t == op_type.clone() || !matches!(t, Type::Any) {
          Ok(())
        } else if !matches!(op_type, Type::Any) {
          {
            env.insert(dest, t);
            Ok(())
          }
        } else {
          Err(InterpError::BadAsmtType(t, op_type.clone()).add_pos(pos.clone()))
        }
      },
    ),
  })?;

  func.instrs.iter().try_for_each(|i| match i {
    bril_rs::Code::Label { .. } => Ok(()),
    bril_rs::Code::Instruction(instr) => {
      type_check_instruction(instr, func, prog, &mut env).map_err(|e| e.add_pos(instr.get_pos()))
    }
  })?;

  Ok(())
}

/// Provides validation of Bril programs. This involves
/// statically checking the types and number of arguments to Bril
/// instructions.
/// # Errors
/// Will return an error if typechecking fails or if the input program is not well-formed.
pub fn type_check(bbprog: &Program) -> Result<(), PositionalInterpError> {
  bbprog
    .functions
    .iter()
    .try_for_each(|bbfunc| type_check_func(bbfunc, bbprog))
}
