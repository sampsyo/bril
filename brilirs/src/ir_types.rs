use std::collections::HashMap;
use std::fmt;

use serde::de::{Deserialize, DeserializeSeed, Deserializer, Error, SeqAccess, Visitor};

// IR JSON Types
macro_rules! define_types {
($($name:ident: $val:ty,)*) => {
  #[derive(Debug, Deserialize, PartialEq, Eq, Clone)]
  #[serde(rename_all = "lowercase")]
  pub enum BrilType {
    $($name,)*
  }

  #[derive(Debug, Deserialize, Clone)]
  pub enum BrilValue {
    $($name($val),)*
    Nil
  }}
}

define_types!(Int: i64, Bool: bool,);

#[derive(Debug)]
pub struct Identifier(pub usize);

#[derive(Debug, Deserialize)]
pub struct Program {
  pub functions: Vec<Function>,
}

#[derive(Debug, Deserialize)]
pub struct Function {
  pub name: String,
  pub instrs: Vec<Instruction>,
}

#[derive(Debug, Deserialize)]
pub struct Label {
  pub label: String,
}

#[derive(Debug, Deserialize)]
pub struct ValueOp {
  pub dest: Identifier,
  #[serde(rename = "type")]
  pub typ: BrilType,
  pub args: Vec<Identifier>,
}

#[derive(Debug, Deserialize)]
pub struct EffectOp {
  pub args: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum Instruction {
  Label(Label),
  Operation(Operation),
}

#[derive(Debug, Deserialize)]
#[serde(tag = "op", rename_all = "lowercase")]
pub enum Operation {
  Const {
    dest: Identifier,
    #[serde(rename = "type")]
    typ: BrilType,
    value: BrilValue,
  },
  Add {
    #[serde(flatten)]
    params: ValueOp,
  },
  Mul {
    #[serde(flatten)]
    params: ValueOp,
  },
  Sub {
    #[serde(flatten)]
    params: ValueOp,
  },
  Div {
    #[serde(flatten)]
    params: ValueOp,
  },
  Eq {
    #[serde(flatten)]
    params: ValueOp,
  },
  Lt {
    #[serde(flatten)]
    params: ValueOp,
  },
  Gt {
    #[serde(flatten)]
    params: ValueOp,
  },
  Le {
    #[serde(flatten)]
    params: ValueOp,
  },
  Ge {
    #[serde(flatten)]
    params: ValueOp,
  },
  Not {
    #[serde(flatten)]
    params: ValueOp,
  },
  And {
    #[serde(flatten)]
    params: ValueOp,
  },
  Or {
    #[serde(flatten)]
    params: ValueOp,
  },
  Jmp {
    #[serde(flatten)]
    params: EffectOp,
  },
  Br {
    #[serde(flatten)]
    params: BrArgs,
  },
  Ret {
    #[serde(flatten)]
    params: EffectOp,
  },
  Id {
    #[serde(flatten)]
    params: ValueOp,
  },
  Print {
    args: Vec<Identifier>,
  },
  Nop,
}

// NOTE: This is a terrible, ugly hack. I'm still trying to figure out how to use
// serde::DeserializeSeed, and will remove it when I get that working. That seems like it'll take
// some more restructuring, though
pub static mut next_id: usize = 0;
pub static mut id_map: HashMap<String, usize> = HashMap::new();

impl<'de> Deserialize<'de> for Identifier {
  fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
  where
    D: Deserializer<'de>,
  {
    deserializer.deserialize_string(IdentifierVisitor)
  }
}

struct IdentifierVisitor;

impl<'de> Visitor<'de> for IdentifierVisitor {
  type Value = Identifier;
  fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
    write!(formatter, "an identifier string")
  }

  fn visit_string<E>(self, v: String) -> Result<Self::Value, E>
  where
    E: Error,
  {
    if id_map.contains_key(&v) {
      Ok(Identifier(*id_map.get(&v).unwrap()))
    } else {
      id_map.insert(v, next_id);
      next_id += 1;
      Ok(Identifier(next_id - 1))
    }
  }
}

#[derive(Debug)]
pub struct BrArgs {
  pub test_var: Identifier,
  pub dests: Vec<String>,
}

impl<'de> Deserialize<'de> for BrArgs {
  fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
  where
    D: Deserializer<'de>,
  {
    struct BrArgsVisitor;
    impl<'de> Visitor<'de> for BrArgsVisitor {
      type Value = BrArgs;
      fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(formatter, "an array of strings")
      }

      fn visit_seq<A>(self, seq: A) -> Result<Self::Value, A::Error>
      where
        A: SeqAccess<'de>,
      {
        let mut ident = None;
        let mut labels = Vec::with_capacity(2);
        // NOTE/TODO: Technically this will allow more than 3 args
        while let Some(elem) = seq.next_element()? {
          if ident.is_none() {
            ident = Some((IdentifierVisitor).visit_string(elem)?);
            continue;
          }

          labels.push(elem);
        }

        Ok(BrArgs {
          test_var: ident.unwrap(),
          dests: labels,
        })
      }
    }

    deserializer.deserialize_seq(BrArgsVisitor)
  }
}
