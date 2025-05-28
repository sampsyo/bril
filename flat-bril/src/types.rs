#![allow(dead_code, clippy::repr_packed_without_abi, non_camel_case_types)]
use num_derive::FromPrimitive;
use serde::{Deserialize, Serialize};
use std::fmt;
use strum_macros::EnumIter;
use zerocopy::{
    FromBytes, FromZeros, Immutable, IntoBytes, KnownLayout, TryFromBytes,
};

/* -------------------------------------------------------------------------- */
/*                                    Types                                   */
/* -------------------------------------------------------------------------- */

/// Flattened type for Bril instructions.   
/// - The `op` field stores an index `i` into `OPCODE_IDX`, where
///   `OPCODE_IDX[i] = (start, end)`, such that `OPCODE_BUFFER[start..=end]`
///   is the serialized version of the opcode
/// - We can store the actual `type` and `value` inline in the `Instr` struct
///   (since they're either an int or a bool,
///   i.e. they don't need to be heap-allocated)
/// - `dest` stores the start & end indices (inclusive) of the byte representation
///   of the string in the `all_vars` byte vector (see `flatten.rs`)
/// - `args` and `labels` contains the start & end indices (inclusive)
///   in their index vectors (see `all_args_idxes` & `all_labels_idxes` in `flatten.rs`)
/// - For `args` and `labels` we have 2 layers of indirection since
///   an instruction can have multiple args/labels, so
///   `(start, end) = instr.arg ==> all_args_idxes[start..=end] ==> all_vars[...]`
/// - (Well-formedness condition: we must have end_idx >= start_idx always)
#[derive(Debug, PartialEq, Clone)]
pub struct Instr {
    pub op: u32,
    pub label: Option<(u32, u32)>,
    pub dest: Option<(u32, u32)>,
    pub ty: Option<Type>,
    pub value: Option<BrilValue>,
    pub args: Option<(u32, u32)>,
    pub instr_labels: Option<(u32, u32)>,
    pub funcs: Option<(u32, u32)>,
}

/// Struct representation of the pair `(i32, i32)`
/// (we need this b/c `zerocopy` doesn't work for tuples)
#[repr(C)]
#[derive(Debug, PartialEq, Clone, Copy, IntoBytes, Immutable, FromBytes)]
pub struct I32Pair {
    pub first: i32,
    pub second: i32,
}

/// Flattened representation of an instruction, amenable to `zerocopy`
#[derive(Debug, PartialEq, Clone, Copy, IntoBytes, Immutable, TryFromBytes)]
#[repr(packed)]
pub struct FlatInstr {
    pub op: u32,
    pub label: I32Pair,
    pub dest: I32Pair,
    pub args: I32Pair,
    pub instr_labels: I32Pair,
    pub funcs: I32Pair,
    pub ty: FlatType,
    pub value: FlatBrilValue,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub enum InstrKind {
    Label,
    Const,
    ValueOp,
    EffectOp,
    Nop,
}

/// Primitive types in core Bril are either `int` or `bool`
#[repr(C)]
#[derive(Debug, PartialEq, Clone, Copy, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Type {
    Int = 0,
    Bool = 1,
}

#[repr(usize)]
#[derive(
    Debug,
    PartialEq,
    Clone,
    Copy,
    Deserialize,
    IntoBytes,
    FromZeros,
    Immutable,
    KnownLayout,
)]
#[serde(rename_all = "lowercase")]
pub enum FlatType {
    Int = 0,
    Bool = 1,
    Null = 2,
}

/// The type of primitive values in Bril.    
/// - Note: We call this enum `BrilValue` to avoid namespace clashes
///   with `serde_json::Value`
/// - `SurrogateBool` is needed for padding reasons (to make zerocopy happy)
#[derive(Debug, PartialEq, Clone, Copy)]
#[repr(u64)]
pub enum BrilValue {
    IntVal(i64),
    BoolVal(SurrogateBool),
}

impl BrilValue {
    /// Extracts the type from a `BrilValue`
    pub fn get_type(&self) -> Type {
        match self {
            BrilValue::IntVal(_) => Type::Int,
            BrilValue::BoolVal(_) => Type::Bool,
        }
    }
}

#[derive(Debug, PartialEq, Clone, Copy, IntoBytes, Immutable, FromZeros)]
#[repr(u64)]
pub enum FlatBrilValue {
    IntVal(i64),
    BoolVal(SurrogateBool),
    Null(SurrogateNull),
}

/// A null which is represented as a u64 to make zerocopy happy
#[derive(Debug, PartialEq, Clone, Copy, IntoBytes, Immutable, FromBytes)]
pub struct SurrogateNull(u64);

/// A type isomorphic to `bool`, which is represented as a u64
/// (so that it has the same representation as `BrilValue::IntVal`'s)
#[derive(Debug, PartialEq, Clone, Copy, IntoBytes, Immutable, FromBytes)]
pub struct SurrogateBool(u64);

impl Instr {
    /// Represents a label as an `Instr` where
    /// all other fields of the struct are none
    pub fn make_label(label_idxes: (u32, u32)) -> Self {
        Self {
            op: u32::MAX,
            label: Some(label_idxes),
            dest: None,
            ty: None,
            value: None,
            args: None,
            instr_labels: None,
            funcs: None,
        }
    }

    /// Retrieves the kind of an instruction (`Nop, Const, EffectOp, ValueOp`)
    pub fn get_instr_kind(&self) -> InstrKind {
        use Opcode::*;
        let possible_op = Opcode::u32_to_opcode(self.op);
        if let Some(op) = possible_op {
            match op {
                Nop => InstrKind::Nop,
                Const => InstrKind::Const,
                Print | Jmp | Br | Ret => InstrKind::EffectOp,
                Call => {
                    // Function calls can be both value op and effect op
                    // depending on whether the `dest` field of the instr
                    // is present
                    if let Some((_, _)) = self.dest {
                        InstrKind::ValueOp
                    } else {
                        InstrKind::EffectOp
                    }
                }
                _ => InstrKind::ValueOp,
            }
        } else {
            InstrKind::Label
        }
    }
}

impl FlatInstr {
    /// Retrieves the kind of an instruction (`Nop, Const, EffectOp, ValueOp`)
    pub fn get_instr_kind(&self) -> InstrKind {
        use Opcode::*;
        let possible_op = Opcode::u32_to_opcode(self.op);
        if let Some(op) = possible_op {
            match op {
                Nop => InstrKind::Nop,
                Const => InstrKind::Const,
                Print | Jmp | Br | Ret => InstrKind::EffectOp,
                Call => {
                    // Function calls can be both value op and effect op
                    // depending on whether the `dest` field of the instr
                    // is present
                    if self.dest.first == -1 && self.dest.second == -1 {
                        InstrKind::EffectOp
                    } else {
                        InstrKind::ValueOp
                    }
                }
                _ => InstrKind::ValueOp,
            }
        } else {
            InstrKind::Label
        }
    }
}

/// The type of Bril opcodes
#[repr(C)]
#[derive(
    Debug,
    PartialEq,
    Clone,
    Copy,
    Deserialize,
    Serialize,
    EnumIter,
    FromPrimitive,
)]
#[serde(rename_all = "lowercase")]
pub enum Opcode {
    // Arithmetic
    Add = 0,
    Mul = 1,
    Sub = 2,
    Div = 3,

    // Comparison
    Eq = 4,
    Lt = 5,
    Gt = 6,
    Le = 7,
    Ge = 8,

    // Logic operations
    Not = 9,
    And = 10,
    Or = 11,

    // Control flow
    Jmp = 12,
    Br = 13,
    Call = 14,
    Ret = 15,

    // Misc operations
    Id = 16,
    Print = 17,
    Nop = 18,
    Const = 19,
}

impl Opcode {
    /// Determines if an opcode is a binary (value) operation
    pub fn is_binop(self) -> bool {
        !matches!(
            self,
            Opcode::Not
                | Opcode::Jmp
                | Opcode::Br
                | Opcode::Call
                | Opcode::Ret
                | Opcode::Id
                | Opcode::Print
                | Opcode::Nop
                | Opcode::Const
        )
    }

    /// Determines if an opcode is a unary (value) operation (i.e. `not`, `id`)
    pub fn is_unop(self) -> bool {
        matches!(self, Opcode::Not | Opcode::Id)
    }

    /// Converts a `u32` value to the corresponding `Opcode`
    /// - Panics if the `u32` value can't be converted
    pub fn u32_to_opcode(v: u32) -> Option<Self> {
        num_traits::FromPrimitive::from_u32(v)
    }

    /// Returns the `(start idx, end idx)` of the opcode in the `OPCODES` buffer
    pub fn get_buffer_start_end_indexes(&self) -> (usize, usize) {
        let opcode = *self;
        OPCODE_IDX[opcode as usize]
    }

    /// Extracts the index of the opcode's `(start_idx, end_idx)` pair
    /// in `OPCODE_IDX`
    pub fn get_index(&self) -> usize {
        let (start_idx, end_idx) = self.get_buffer_start_end_indexes();
        OPCODE_IDX
            .iter()
            .position(|&x| x.0 == start_idx && x.1 == end_idx)
            .expect("Opcode not present in `OPCODE_IDX`")
    }

    /// Converts an `Opcode` to a `&str` using the `(start_idx, end_idx)`
    /// obtained from `Instr::get_buffer_start_end_indexes`.
    pub fn as_str(&self) -> &str {
        let (start_idx, end_idx) = self.get_buffer_start_end_indexes();
        &OPCODE_BUFFER[start_idx..=end_idx]
    }

    /// Converts an opcode's index in `OPCODE_IDX` to a `String` representation
    /// of an opcode
    pub fn op_idx_to_op_str(op_idx: usize) -> String {
        let (start_idx, end_idx) = OPCODE_IDX[op_idx];
        let op_str = &OPCODE_BUFFER[start_idx..=end_idx];
        op_str.to_string()
    }
}

/// Struct representing the two components of an argument to a Bril function:
/// - The argument name, represented by the start & end indexes in the
///   `var_store` vector of `InstrStore`
/// - The type of the argument
#[derive(Debug, PartialEq, Clone, Copy)]
pub struct FuncArg {
    pub arg_name_idxes: (u32, u32),
    pub arg_type: Type,
}

/// Flat version of a `FuncArg`
#[repr(packed)]
#[derive(Debug, PartialEq, Clone, Copy, IntoBytes, Immutable, TryFromBytes)]
pub struct FlatFuncArg {
    pub arg_name_idxes: I32Pair,
    pub arg_type: FlatType,
}

/// Struct that stores all the instrs and the args/dest/labels/funcs arrays
/// in the same place (note: we create one `InstrStore` per Bril function)
/// - The `func_name` field stores the name of the Bril function
///   corresponding to this `InstrStore`
/// - `func_args` is a list of function parameters
///   (arg type + indexes for the arg name)
/// - `func_ret_ty` is the return type of the function
///   (`None` means the function is void, i.e. has no return type)
/// - args_idxes_stores |-> var_store
/// - labels_idxes_store |-> labels_store
/// - there's only one function so `funcs_store` can just be Vec<u8>
/// - `instrs_and_labels` is a vector containing the instructions/labels in
///   the order they appear in the source Bril file
#[derive(Debug, Clone, PartialEq)]
pub struct InstrStore {
    pub func_name: Vec<u8>,
    pub func_args: Vec<FuncArg>,
    pub func_ret_ty: Option<Type>,
    pub var_store: Vec<u8>,
    pub args_idxes_store: Vec<(u32, u32)>,
    pub labels_idxes_store: Vec<(u32, u32)>,
    pub labels_store: Vec<u8>,
    pub funcs_store: Vec<u8>,
    pub instrs: Vec<Instr>,
}

/// `InstrView` is the same as `InstrStore`:
/// all the slices in `InstrView` are references to the `Vec`s in `InstrStore`
/// All [u8]s are padded to 4 bytes
#[repr(packed)]
#[derive(Debug, PartialEq, Clone, Immutable, IntoBytes)]
pub struct InstrView<'a> {
    pub func_name: &'a [u8],
    pub func_args: &'a [FlatFuncArg],
    pub func_ret_ty: FlatType,
    pub var_store: &'a [u8],
    pub arg_idxes_store: &'a [I32Pair],
    pub labels_idxes_store: &'a [I32Pair],
    pub labels_store: &'a [u8],
    pub funcs_store: &'a [u8],
    pub instrs: &'a [FlatInstr],
}

#[repr(packed)]
#[derive(Debug, PartialEq, Clone, Immutable, IntoBytes)]
pub struct InstrViewFamily<'a> {
    pub instr_views: &'a [InstrView<'a>],
}

/// Top-level metadata in the mmap-ed file, appears before all the `Toc`/`InstrView`s
/// The `sizes` fields contains a list of sizes (no. of bytes) for each
/// of the functions in the Bril program.
#[derive(FromBytes, IntoBytes, Debug, Clone, Copy, Immutable, KnownLayout)]
#[repr(C)]
pub struct Header {
    // TODO: change this in the future? right now we only allow at most 10 functions
    pub sizes: [u64; 10],
}

/// Table of contents for the flat Bril file
/// (each field stores the no. of elements in the corresponding slice
/// in the `InstrView`)
#[derive(FromBytes, IntoBytes, Debug, Clone, Copy, Immutable, KnownLayout)]
#[repr(packed)]
pub struct Toc {
    pub func_name: usize,
    pub func_args: usize,
    pub func_ret_ty: usize,
    pub var_store: usize,
    pub arg_idxes_store: usize,
    pub labels_idxes_store: usize,
    pub labels_store: usize,
    pub funcs_store: usize,
    pub instrs: usize,
}

impl InstrView<'_> {
    /// Returns a `Toc` containing the no. of elements of each field
    /// in the `InstrView` struct
    pub fn get_sizes(&self) -> Toc {
        let func_name = self.func_name.len();
        let func_args = self.func_args.len();
        let func_ret_ty = 1; // Each function only has one return type
        let var_store = self.var_store.len();
        let arg_idxes_store = self.arg_idxes_store.len();
        let labels_idxes_store = self.labels_idxes_store.len();
        let labels_store = self.labels_store.len();
        let funcs_store = self.funcs_store.len();
        let instrs = self.instrs.len();

        Toc {
            func_name,
            func_args,
            func_ret_ty,
            var_store,
            arg_idxes_store,
            labels_idxes_store,
            labels_store,
            funcs_store,
            instrs,
        }
    }

    /// Computes the total no. of bytes occupied by the Toc +
    /// the contents of the `InstrView`
    /// NOTE: This function is broken! don't use it for now
    pub fn total_size_in_bytes(&self) -> u64 {
        let toc_num_bytes = size_of::<Toc>();
        let func_name_num_bytes = std::mem::size_of_val(self.func_name);
        let func_args_num_bytes = std::mem::size_of_val(self.func_args);
        let func_ret_ty_num_bytes = size_of::<FlatType>();
        let var_store_num_bytes = std::mem::size_of_val(self.var_store);
        let arg_idxes_store_num_bytes =
            std::mem::size_of_val(self.arg_idxes_store);
        let labels_idxes_store_num_bytes =
            std::mem::size_of_val(self.labels_idxes_store);
        let labels_store_num_bytes = std::mem::size_of_val(self.labels_store);
        let funcs_store_num_bytes = std::mem::size_of_val(self.funcs_store);
        let instrs_num_bytes = std::mem::size_of_val(self.instrs);

        (toc_num_bytes
            + func_name_num_bytes
            + func_args_num_bytes
            + func_ret_ty_num_bytes
            + var_store_num_bytes
            + arg_idxes_store_num_bytes
            + labels_idxes_store_num_bytes
            + labels_store_num_bytes
            + funcs_store_num_bytes
            + instrs_num_bytes) as u64
    }
}

/* -------------------------------------------------------------------------- */
/*                                  Constants                                 */
/* -------------------------------------------------------------------------- */

/// A string literal storing all distinct opcodes in core Bril
pub const OPCODE_BUFFER: &str =
    "addmulsubdiveqltgtlegenotandorjmpbrcallretidprintnopconst";

/// There are 20 distinct opcodes in core Bril
pub const NUM_OPCODES: usize = 20;

/// Default length of the args array
/// (Rust `Vec`s are initialized with a capacity that is a power of 2,
/// we pick 64 since that seems like a reasonable upper bound for the no. of
/// variables in a Bril function)
pub const NUM_ARGS: usize = 64;

/// Variables are just a way to interpret dests/args, we assume there are 128 of them
pub const NUM_VARS: usize = 128;

/// Similarly, we assume that Bril programs contain at most 128 dests/labels/instrs
pub const NUM_LABELS: usize = 128;
pub const NUM_INSTRS: usize = 128;

/// The only core Bril instruction with a `funcs` field is `call`,
/// whose `funcs` field is just a length-1 list, so we can get away with making
/// `NUM_FUNCS` a really small power of 2, like 8.
pub const NUM_FUNCS: usize = 8;

/// Each pair contains the `(start idx, end idx)` of the opcode in `OPCODES`.     
/// Note that both start and indexes are inclusive.
pub const OPCODE_IDX: [(usize, usize); NUM_OPCODES] = [
    (0, 2),   // Add
    (3, 5),   // Mul
    (6, 8),   // Sub
    (9, 11),  // Div
    (12, 13), // Eq
    (14, 15), // Lt
    (16, 17), // Gt
    (18, 19), // Le
    (20, 21), // Ge
    (22, 24), // Not
    (25, 27), // And
    (28, 29), // Or
    (30, 32), // Jmp
    (33, 34), // Br
    (35, 38), // Call
    (39, 41), // Ret
    (42, 43), // Id
    (44, 48), // Print
    (49, 51), // Nop
    (52, 56), // Const
];

/* -------------------------------------------------------------------------- */
/*                          Converting between types                          */
/* -------------------------------------------------------------------------- */

impl From<Option<Type>> for FlatType {
    fn from(ty_opt: Option<Type>) -> Self {
        match ty_opt {
            Some(Type::Bool) => FlatType::Bool,
            Some(Type::Int) => FlatType::Int,
            None => FlatType::Null,
        }
    }
}

impl From<Type> for FlatType {
    fn from(ty: Type) -> Self {
        match ty {
            Type::Bool => FlatType::Bool,
            Type::Int => FlatType::Int,
        }
    }
}

impl TryFrom<FlatType> for Type {
    type Error = ();

    fn try_from(flat_ty: FlatType) -> Result<Self, Self::Error> {
        match flat_ty {
            FlatType::Bool => Ok(Type::Bool),
            FlatType::Int => Ok(Type::Int),
            FlatType::Null => Err(()),
        }
    }
}

impl From<FlatType> for Option<Type> {
    fn from(flat_type: FlatType) -> Self {
        Result::ok(flat_type.try_into())
    }
}

impl From<Option<BrilValue>> for FlatBrilValue {
    fn from(value_opt: Option<BrilValue>) -> Self {
        match value_opt {
            Some(BrilValue::IntVal(i)) => FlatBrilValue::IntVal(i),
            Some(BrilValue::BoolVal(surrogate_bool)) => {
                FlatBrilValue::BoolVal(surrogate_bool)
            }
            None => FlatBrilValue::Null(SurrogateNull(0)),
        }
    }
}

impl TryFrom<FlatBrilValue> for BrilValue {
    type Error = ();

    fn try_from(flat_val: FlatBrilValue) -> Result<Self, Self::Error> {
        match flat_val {
            FlatBrilValue::BoolVal(surrogate_bool) => {
                Ok(BrilValue::BoolVal(surrogate_bool))
            }
            FlatBrilValue::IntVal(i) => Ok(BrilValue::IntVal(i)),
            FlatBrilValue::Null(_) => Err(()),
        }
    }
}

impl From<FlatBrilValue> for Option<BrilValue> {
    fn from(flat_value: FlatBrilValue) -> Self {
        Result::ok(flat_value.try_into())
    }
}

// `bool::from(surrogate_bool)` is useful
impl From<SurrogateBool> for bool {
    fn from(surrogate_bool: SurrogateBool) -> Self {
        let b = surrogate_bool.0;
        b == 0
    }
}

// For `b:bool`, `b.into()` converts it to a `SurrogateBool`
impl From<bool> for SurrogateBool {
    fn from(b: bool) -> Self {
        if b {
            SurrogateBool(0)
        } else {
            SurrogateBool(1)
        }
    }
}

impl From<(u32, u32)> for I32Pair {
    fn from(pair: (u32, u32)) -> Self {
        Self {
            first: pair.0 as i32,
            second: pair.1 as i32,
        }
    }
}

// Convention: None |-> an `I32Pair` where both fields are -1
impl From<Option<(u32, u32)>> for I32Pair {
    fn from(pair_opt: Option<(u32, u32)>) -> Self {
        match pair_opt {
            None => I32Pair {
                first: -1,
                second: -1,
            },
            Some((i, j)) => I32Pair {
                first: i as i32,
                second: j as i32,
            },
        }
    }
}

// Convention: `I32Pair {first: -1, second: -1} |-> None`
impl From<I32Pair> for Option<(u32, u32)> {
    fn from(i32pair: I32Pair) -> Self {
        let I32Pair { first, second } = i32pair;
        if first == -1 && second == -1 {
            None
        } else {
            let first = first as u32;
            let second = second as u32;
            Some((first, second))
        }
    }
}

impl From<I32Pair> for (u32, u32) {
    fn from(i32pair: I32Pair) -> Self {
        let I32Pair { first, second } = i32pair;
        (first as u32, second as u32)
    }
}

// Converting `Instr` to `FlatInstr`
impl From<Instr> for FlatInstr {
    fn from(instr: Instr) -> Self {
        FlatInstr {
            op: instr.op,
            label: instr.label.into(),
            dest: instr.dest.into(),
            args: instr.args.into(),
            instr_labels: instr.instr_labels.into(),
            funcs: instr.funcs.into(),
            ty: instr.ty.into(),
            value: instr.value.into(),
        }
    }
}

impl From<FlatInstr> for Instr {
    fn from(flat_instr: FlatInstr) -> Self {
        Instr {
            op: flat_instr.op,
            label: flat_instr.label.into(),
            dest: flat_instr.dest.into(),
            ty: flat_instr.ty.into(),
            value: flat_instr.value.into(),
            args: flat_instr.args.into(),
            instr_labels: flat_instr.instr_labels.into(),
            funcs: flat_instr.funcs.into(),
        }
    }
}

impl From<FuncArg> for FlatFuncArg {
    fn from(func_arg: FuncArg) -> Self {
        Self {
            arg_name_idxes: func_arg.arg_name_idxes.into(),
            arg_type: func_arg.arg_type.into(),
        }
    }
}

impl From<FlatFuncArg> for FuncArg {
    fn from(flat_func_arg: FlatFuncArg) -> Self {
        Self {
            arg_name_idxes: flat_func_arg.arg_name_idxes.into(),
            arg_type: flat_func_arg
                .arg_type
                .try_into()
                .expect("Can't convert `FlatType::Null` into a `Type`"),
        }
    }
}

impl From<InstrView<'_>> for InstrStore {
    fn from(instr_view: InstrView) -> Self {
        let func_name = instr_view.func_name.into();
        let func_args: Vec<FuncArg> = instr_view
            .func_args
            .iter()
            .map(|func_arg| FuncArg::from(*func_arg))
            .collect();

        let func_ret_ty = instr_view.func_ret_ty.into();

        let var_store = instr_view.var_store.into();
        let args_idxes_store: Vec<(u32, u32)> = instr_view
            .arg_idxes_store
            .iter()
            .map(|arg_idxes| <(u32, u32)>::from(*arg_idxes))
            .collect();
        let labels_idxes_store: Vec<(u32, u32)> = instr_view
            .labels_idxes_store
            .iter()
            .map(|label_idxes| <(u32, u32)>::from(*label_idxes))
            .collect();

        let labels_store = instr_view.labels_store.into();
        let funcs_store = instr_view.funcs_store.into();
        let instrs: Vec<Instr> = instr_view
            .instrs
            .iter()
            .map(|flat_instr| Instr::from(*flat_instr))
            .collect();

        InstrStore {
            func_name,
            func_args,
            func_ret_ty,
            var_store,
            args_idxes_store,
            labels_idxes_store,
            labels_store,
            funcs_store,
            instrs,
        }
    }
}

/* -------------------------------------------------------------------------- */
/*                               Pretty-Printing                              */
/* -------------------------------------------------------------------------- */

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Type::Int => write!(f, "int"),
            Type::Bool => write!(f, "bool"),
        }
    }
}

impl Type {
    /// Converts a `Type` to its string representation
    pub fn as_str(&self) -> &str {
        match self {
            Type::Int => "int",
            Type::Bool => "bool",
        }
    }
}

impl fmt::Display for BrilValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BrilValue::IntVal(n) => write!(f, "{}", n),
            BrilValue::BoolVal(b) => {
                write!(f, "{}", bool::from(*b))
            }
        }
    }
}
