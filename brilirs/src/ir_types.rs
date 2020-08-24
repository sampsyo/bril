use std::fmt;

use serde::de::{Deserialize, Deserializer, SeqAccess, Visitor};

// IR JSON Types
macro_rules! define_types {
($($name:ident: $val:ty,)*) => {
  #[derive(Debug, Deserialize, PartialEq, Eq, Clone)]
  #[serde(rename_all = "lowercase")]
  pub enum BrilType {
    $($name,)*
    Nil
  }

  #[derive(Debug, Deserialize, Clone)]
  #[serde(untagged)]
  pub enum BrilValue {
    $($name($val),)*
    Nil
  }}
}

define_types!(Int: i64, Bool: bool,);

#[derive(Debug, Deserialize)]
pub struct Identifier<T>(pub T);

#[derive(Debug, Deserialize)]
pub struct Program<T> {
  pub functions: Vec<Function<T>>,
}

#[derive(Debug, Deserialize)]
pub struct Function<T> {
  pub name: String,
  pub instrs: Vec<Instruction<T>>,
}

#[derive(Debug, Deserialize)]
pub struct Label {
  pub label: String,
}

#[derive(Debug, Deserialize)]
pub struct ValueOp<T> {
  pub dest: Identifier<T>,
  #[serde(rename = "type")]
  pub typ: BrilType,
  pub args: Vec<Identifier<T>>,
}

#[derive(Debug, Deserialize)]
pub struct EffectOp {
  pub args: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum Instruction<T> {
  Label(Label),
  Operation(Operation<T>),
}

#[derive(Debug, Deserialize)]
#[serde(tag = "op", rename_all = "lowercase")]
pub enum Operation<T> {
  Const {
    dest: Identifier<T>,
    #[serde(rename = "type")]
    typ: BrilType,
    value: BrilValue,
  },
  Add {
    #[serde(flatten)]
    params: ValueOp<T>,
  },
  Mul {
    #[serde(flatten)]
    params: ValueOp<T>,
  },
  Sub {
    #[serde(flatten)]
    params: ValueOp<T>,
  },
  Div {
    #[serde(flatten)]
    params: ValueOp<T>,
  },
  Eq {
    #[serde(flatten)]
    params: ValueOp<T>,
  },
  Lt {
    #[serde(flatten)]
    params: ValueOp<T>,
  },
  Gt {
    #[serde(flatten)]
    params: ValueOp<T>,
  },
  Le {
    #[serde(flatten)]
    params: ValueOp<T>,
  },
  Ge {
    #[serde(flatten)]
    params: ValueOp<T>,
  },
  Not {
    #[serde(flatten)]
    params: ValueOp<T>,
  },
  And {
    #[serde(flatten)]
    params: ValueOp<T>,
  },
  Or {
    #[serde(flatten)]
    params: ValueOp<T>,
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
    params: ValueOp<T>,
  },
  Print {
    args: Vec<Identifier<T>>,
  },
  Nop,
}

#[derive(Debug)]
pub enum BrArgs {
  StringArgs {
    test_var: Identifier<String>,
    dests: Vec<String>,
  },
  IdArgs {
    test_var: Identifier<usize>,
    dests: Vec<String>,
  },
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

      fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
      where
        A: SeqAccess<'de>,
      {
        let mut ident = None;
        let mut labels = Vec::with_capacity(2);
        // NOTE/TODO: Technically this will allow more than 3 args
        while let Some(elem) = seq.next_element()? {
          if ident.is_none() {
            ident = Some(Identifier::<String>(elem));
            continue;
          }

          labels.push(elem);
        }

        Ok(BrArgs::StringArgs {
          test_var: ident.unwrap(),
          dests: labels,
        })
      }
    }

    deserializer.deserialize_seq(BrArgsVisitor)
  }
}
