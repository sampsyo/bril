use std::ops::{Add, Index};

use bril_rs::{ConstOps, EffectOps, Instruction, Literal, ValueOps};
use fxhash::FxHashMap;

use crate::error::InterpError;

/// A type alias for trying different index type sizes
pub type IndexType = u16;

/// A Newtype for function indexing
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct FuncIndex(pub IndexType);

impl FuncIndex {
  /// Creates a new `FuncIndex` from a usize.
  pub fn new(value: usize) -> Self {
    FuncIndex(value as IndexType)
  }
}

impl<T> Index<FuncIndex> for [T] {
  type Output = T;

  fn index(&self, index: FuncIndex) -> &Self::Output {
    &self[index.0 as usize]
  }
}

/// A Newtype for label indexing
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct LabelIndex(pub IndexType);

impl<T> Index<LabelIndex> for [T] {
  type Output = T;

  fn index(&self, index: LabelIndex) -> &Self::Output {
    &self[index.0 as usize]
  }
}

impl LabelIndex {
  /// Creates a new `LabelIndex` from a usize.
  pub fn new(value: usize) -> Self {
    LabelIndex(value as IndexType)
  }
}

/// A Newtype for variable indexing
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct VarIndex(pub IndexType);

impl Add<VarIndex> for usize {
  type Output = usize;

  fn add(self, rhs: VarIndex) -> Self::Output {
    self + (rhs.0 as usize)
  }
}

impl Add<&VarIndex> for usize {
  type Output = usize;

  fn add(self, rhs: &VarIndex) -> Self::Output {
    self + (rhs.0 as usize)
  }
}

// TODO: Maybe swap out the other new functions for this?
impl From<usize> for VarIndex {
  fn from(value: usize) -> Self {
    VarIndex(value as IndexType)
  }
}

#[allow(missing_docs)]
#[derive(Debug)]
pub enum FlatIR {
  Const {
    dest: VarIndex,
    value: Literal,
  },
  ZeroArity {
    op: ValueOps,
    dest: VarIndex,
  },
  UnaryArity {
    op: ValueOps,
    dest: VarIndex,
    arg: VarIndex,
  },
  BinaryArity {
    op: ValueOps,
    dest: VarIndex,
    arg0: VarIndex,
    arg1: VarIndex,
  },
  MultiArityCall {
    func: FuncIndex,
    dest: VarIndex,
    args: Vec<VarIndex>,
  },
  Nop,
  Jump {
    dest: LabelIndex,
  },
  Branch {
    arg: VarIndex,
    true_dest: LabelIndex,
    false_dest: LabelIndex,
  },
  ReturnValue {
    arg: VarIndex,
  },
  ReturnVoid,
  EffectfulCall {
    func: FuncIndex,
    args: Vec<VarIndex>,
  },
  PrintOne {
    arg: VarIndex,
  },
  PrintMultiple {
    args: Vec<VarIndex>,
  },
  Store {
    arg0: VarIndex,
    arg1: VarIndex,
  },
  Set {
    arg0: VarIndex,
    arg1: VarIndex,
  },
  Free {
    arg: VarIndex,
  },
}

const _: () = {
  assert!(32 == std::mem::size_of::<FlatIR>());
};

impl FlatIR {
  /// Converts a bril_rs [Instruction] into a [FlatIR] variant.
  pub fn new(
    i: Instruction,
    func_map: &FxHashMap<String, FuncIndex>,
    num_var_map: &mut FxHashMap<String, VarIndex>,
    num_label_map: &FxHashMap<String, LabelIndex>,
  ) -> Result<Self, InterpError> {
    match i {
      Instruction::Constant {
        dest,
        op: ConstOps::Const,
        pos: _,
        const_type,
        value,
      } => Ok(FlatIR::Const {
        dest: get_num_from_map(dest, num_var_map),
        value: if const_type == bril_rs::Type::Float {
          match value {
            Literal::Int(i) => Literal::Float(i as f64),
            Literal::Float(_) => value,
            _ => unreachable!(),
          }
        } else {
          value
        },
      }),
      Instruction::Value {
        op: op @ (ValueOps::Undef | ValueOps::Get),
        dest,
        args: _,
        funcs: _,
        labels: _,
        pos: _,
        op_type: _,
      } => {
        let dest = get_num_from_map(dest, num_var_map);

        Ok(FlatIR::ZeroArity { op: op, dest })
      }

      Instruction::Value {
        op:
          op @ (ValueOps::Id
          | ValueOps::Not
          | ValueOps::Char2int
          | ValueOps::Int2char
          | ValueOps::Alloc
          | ValueOps::Load
          | ValueOps::Float2Bits
          | ValueOps::Bits2Float),
        args,
        dest,
        funcs: _,
        labels: _,
        pos: _,
        op_type: _,
      } => {
        let dest = get_num_from_map(dest, num_var_map);

        let mut iter = args.into_iter().map(|v| get_num_from_map(v, num_var_map));
        let arg = iter.next().unwrap();

        Ok(FlatIR::UnaryArity { op, dest, arg })
      }
      Instruction::Value {
        op:
          op @ (ValueOps::Add
          | ValueOps::Sub
          | ValueOps::Mul
          | ValueOps::Div
          | ValueOps::And
          | ValueOps::Or
          | ValueOps::Le
          | ValueOps::Eq
          | ValueOps::Gt
          | ValueOps::Ge
          | ValueOps::Lt
          | ValueOps::Fadd
          | ValueOps::Fsub
          | ValueOps::Fmul
          | ValueOps::Fdiv
          | ValueOps::Fle
          | ValueOps::Feq
          | ValueOps::Fgt
          | ValueOps::Fge
          | ValueOps::Flt
          | ValueOps::Ceq
          | ValueOps::Clt
          | ValueOps::Cgt
          | ValueOps::Cge
          | ValueOps::Cle
          | ValueOps::PtrAdd),
        args,
        dest,
        funcs: _,
        labels: _,
        pos: _,
        op_type: _,
      } => {
        let dest = get_num_from_map(dest, num_var_map);

        let mut iter = args.into_iter().map(|v| get_num_from_map(v, num_var_map));
        let arg0 = iter.next().unwrap();
        let arg1 = iter.next().unwrap();

        Ok(FlatIR::BinaryArity {
          op,
          dest,
          arg0,
          arg1,
        })
      }
      Instruction::Value {
        op: ValueOps::Call,
        args,
        dest,
        funcs,
        labels: _,
        pos: _,
        op_type: _,
      } => {
        let dest = get_num_from_map(dest, num_var_map);
        let args = args
          .into_iter()
          .map(|v| get_num_from_map(v, num_var_map))
          .collect();
        let func = func_map.get(&funcs[0]).copied().unwrap();
        Ok(FlatIR::MultiArityCall { func, dest, args })
      }
      Instruction::Effect {
        op: EffectOps::Nop,
        args: _,
        funcs: _,
        labels: _,
        pos: _,
      } => Ok(FlatIR::Nop),
      Instruction::Effect {
        op: EffectOps::Jump,
        args: _,
        funcs: _,
        labels,
        pos: _,
      } => {
        let dest = labels
          .into_iter()
          .map(|v| {
            num_label_map
              .get(&v)
              .copied()
              .ok_or_else(|| InterpError::MissingLabel(v.clone()))
          })
          .next()
          .unwrap()?;
        Ok(FlatIR::Jump { dest })
      }
      Instruction::Effect {
        op: EffectOps::Branch,
        args,
        funcs: _,
        labels,
        pos: _,
      } => {
        let arg = args
          .into_iter()
          .map(|v| get_num_from_map(v, num_var_map))
          .next()
          .unwrap();
        let mut iter = labels.into_iter().map(|v| {
          num_label_map
            .get(&v)
            .copied()
            .ok_or_else(|| InterpError::MissingLabel(v.clone()))
        });
        let true_dest = iter.next().unwrap()?;
        let false_dest = iter.next().unwrap()?;
        Ok(FlatIR::Branch {
          arg,
          true_dest,
          false_dest,
        })
      }
      Instruction::Effect {
        op: EffectOps::Return,
        args,
        funcs: _,
        labels: _,
        pos: _,
      } => {
        if args.is_empty() {
          Ok(FlatIR::ReturnVoid)
        } else {
          let arg = args
            .into_iter()
            .map(|v| get_num_from_map(v, num_var_map))
            .next()
            .unwrap();
          Ok(FlatIR::ReturnValue { arg })
        }
      }
      Instruction::Effect {
        op: EffectOps::Call,
        args,
        funcs,
        labels: _,
        pos: _,
      } => {
        let args = args
          .into_iter()
          .map(|v| get_num_from_map(v, num_var_map))
          .collect();
        let func = func_map.get(&funcs[0]).copied().unwrap();
        Ok(FlatIR::EffectfulCall { func, args })
      }
      Instruction::Effect {
        op: EffectOps::Print,
        args,
        funcs: _,
        labels: _,
        pos: _,
      } => {
        if args.len() == 1 {
          let arg = args
            .into_iter()
            .map(|v| get_num_from_map(v, num_var_map))
            .next()
            .unwrap();
          Ok(FlatIR::PrintOne { arg })
        } else {
          let args = args
            .into_iter()
            .map(|v| get_num_from_map(v, num_var_map))
            .collect();
          Ok(FlatIR::PrintMultiple { args })
        }
      }
      Instruction::Effect {
        op: EffectOps::Store,
        args,
        funcs: _,
        labels: _,
        pos: _,
      } => {
        let mut iter = args.into_iter().map(|v| get_num_from_map(v, num_var_map));
        let arg0 = iter.next().unwrap();
        let arg1 = iter.next().unwrap();
        Ok(FlatIR::Store { arg0, arg1 })
      }
      Instruction::Effect {
        op: EffectOps::Set,
        args,
        funcs: _,
        labels: _,
        pos: _,
      } => {
        let mut iter = args.into_iter().map(|v| get_num_from_map(v, num_var_map));
        let arg0 = iter.next().unwrap();
        let arg1 = iter.next().unwrap();
        Ok(FlatIR::Set { arg0, arg1 })
      }
      Instruction::Effect {
        op: EffectOps::Free,
        args,
        funcs: _,
        labels: _,
        pos: _,
      } => {
        let arg = args
          .into_iter()
          .map(|v| get_num_from_map(v, num_var_map))
          .next()
          .unwrap();
        Ok(FlatIR::Free { arg })
      }
      Instruction::Effect {
        op: EffectOps::Speculate | EffectOps::Guard | EffectOps::Commit,
        ..
      } => unimplemented!(),
    }
  }
}

/// Gets a number from the map, or inserts it if it doesn't exist.
pub fn get_num_from_map<T: Copy + From<usize>>(
  variable_name: String,
  num_var_map: &mut FxHashMap<String, T>,
) -> T {
  num_var_map.get(&variable_name).copied().unwrap_or_else(|| {
    let x = num_var_map.len().into();
    num_var_map.insert(variable_name, x);
    x
  })
}
