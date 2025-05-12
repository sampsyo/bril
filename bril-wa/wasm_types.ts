// types that specify binary encoding of WASM types
type U32 = number; // just treat this like a u32 ig i fucking hate typescriptttttt

enum NumTypeEnc {
  i32 = 0x7F,
  i64 = 0x7E,
  f32 = 0x7D,
  f64 = 0x7C,
}

enum VecTypeEnc {
  Vectype = 0x7B
}

enum RefTypeEnc {
  Funcref = 0x70,
  Externref = 0x6F
}
type UnboundedLimit = { min: U32 }
type BoundedLimit = { min: U32, max: U32 }

enum LimitEnc {
  UnboundedLimit = 0x00,
  BoundedLimit = 0x01
}

enum MutEnc {
  Const = 0x00,
  Var = 0x01
}

enum ValueTypeEnc {
  i32 = 0x7F,
  i64 = 0x7E,
  f32 = 0x7D,
  f64 = 0x7C,
  Vectype = 0x7B,
  Funcref = 0x70,
  Externref = 0x6F
}

type ResultTypeEnc = [ValueTypeEnc]

type FuncTypeEnc = { params: [ResultTypeEnc], results: [ResultTypeEnc] }

type MemTypeEnc = { lim: LimitEnc }

type TableTypeEnc = { et: RefTypeEnc, lim: LimitEnc }

type GlobalTypeEnc = {
  mut: MutEnc, t: ValueTypeEnc
}

enum BlockTypeEnc {
  Empty = 0x40,
}
// types that specify encoding for instructions
enum OpCodeEnc {
  Unreachable = 0x00,
  Nop = 0x01,
  Block = 0x02,
  Loop = 0x03,
  If = 0x04,
  Else = 0x05,
  Br = 0x0C,
  BrIf = 0x0D,
  BrTable = 0x0E,
  Ret = 0x0F,
  Call = 0x10,
  CallIndirect = 0x11,
}