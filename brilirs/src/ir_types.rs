use serde_json;
use serde_json::Value;

// IR JSON Types
#[derive(Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Type {
  Int,
  Bool,
}

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
  pub dest: String,
  #[serde(rename = "type")]
  pub typ: Type,
  pub args: Vec<String>,
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
    dest: String,
    #[serde(rename = "type")]
    typ: Type,
    value: Value,
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
    params: EffectOp,
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
    args: Vec<String>,
  },
  Nop,
}
