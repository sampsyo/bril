type U32 = number; // just treat this like a u32 ig i fucking hate typescriptttttt

enum NumType {
  i32 = 0x7F,
  i64 = 0x7E,
  f32 = 0x7D,
  f64 = 0x7C,
}

enum VecType {
  vectype = 0x7B
}

enum RefType {
  funcref = 0x70,
  externref = 0x6F
}
type UnboundedLimit = { min: U32 }
type BoundedLimit = { min: U32, max: U32 }
enum Limit {
  UnboundedLimit = 0x00,
  BoundedLimit = 0x01
}

type ValueType = NumType | VecType | RefType

type ResultType = [ValueType]

type FuncType = { params: [ResultType], results: [ResultType] }


