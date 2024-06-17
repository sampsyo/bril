use crate::rt;
use bril_rs as bril;
use core::mem;
use cranelift_codegen::entity::EntityRef;
use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_codegen::ir::InstBuilder;
use cranelift_codegen::settings::Configurable;
use cranelift_codegen::{ir, isa, settings};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{default_libcall_names, Module};
use cranelift_object::{ObjectBuilder, ObjectModule};
use enum_map::{enum_map, Enum, EnumMap};
use std::collections::HashMap;
use std::fs;
use std::sync::Arc;

/// Runtime functions used by ordinary Bril instructions.
#[derive(Debug, Enum)]
#[allow(clippy::enum_variant_names)]
enum RTFunc {
    PrintInt,
    PrintBool,
    PrintFloat,
    PrintSep,
    PrintEnd,
    Alloc,
    Free,
}

impl RTFunc {
    fn sig(
        &self,
        pointer_type: ir::Type,
        call_conv: cranelift_codegen::isa::CallConv,
    ) -> ir::Signature {
        match self {
            Self::PrintInt => ir::Signature {
                params: vec![ir::AbiParam::new(ir::types::I64)],
                returns: vec![],
                call_conv,
            },
            Self::PrintBool => ir::Signature {
                params: vec![ir::AbiParam::new(ir::types::I8)],
                returns: vec![],
                call_conv,
            },
            Self::PrintFloat => ir::Signature {
                params: vec![ir::AbiParam::new(ir::types::F64)],
                returns: vec![],
                call_conv,
            },
            Self::PrintSep => ir::Signature {
                params: vec![],
                returns: vec![],
                call_conv,
            },
            Self::PrintEnd => ir::Signature {
                params: vec![],
                returns: vec![],
                call_conv,
            },
            Self::Alloc => ir::Signature {
                params: vec![
                    ir::AbiParam::new(ir::types::I64),
                    ir::AbiParam::new(ir::types::I64),
                ],
                returns: vec![ir::AbiParam::new(pointer_type)],
                call_conv,
            },
            Self::Free => ir::Signature {
                params: vec![ir::AbiParam::new(pointer_type)],
                returns: vec![],
                call_conv,
            },
        }
    }

    fn name(&self) -> &'static str {
        match self {
            Self::PrintInt => "_bril_print_int",
            Self::PrintBool => "_bril_print_bool",
            Self::PrintFloat => "_bril_print_float",
            Self::PrintSep => "_bril_print_sep",
            Self::PrintEnd => "_bril_print_end",
            Self::Alloc => "_bril_alloc",
            Self::Free => "_bril_free",
        }
    }

    fn rt_impl(&self) -> *const u8 {
        match self {
            RTFunc::PrintInt => rt::print_int as *const u8,
            RTFunc::PrintBool => rt::print_bool as *const u8,
            RTFunc::PrintFloat => rt::print_float as *const u8,
            RTFunc::PrintSep => rt::print_sep as *const u8,
            RTFunc::PrintEnd => rt::print_end as *const u8,
            RTFunc::Alloc => rt::mem_alloc as *const u8,
            RTFunc::Free => rt::mem_free as *const u8,
        }
    }
}

/// Runtime functions used in the native `main` function, which dispatches to the proper Bril
/// `main` function.
#[derive(Debug, Enum)]
#[allow(clippy::enum_variant_names)]
enum RTSetupFunc {
    ParseInt,
    ParseBool,
    ParseFloat,
}

impl RTSetupFunc {
    fn sig(
        &self,
        pointer_type: ir::Type,
        call_conv: cranelift_codegen::isa::CallConv,
    ) -> ir::Signature {
        match self {
            Self::ParseInt => ir::Signature {
                params: vec![
                    ir::AbiParam::new(pointer_type),
                    ir::AbiParam::new(ir::types::I64),
                ],
                returns: vec![ir::AbiParam::new(ir::types::I64)],
                call_conv,
            },
            Self::ParseBool => ir::Signature {
                params: vec![
                    ir::AbiParam::new(pointer_type),
                    ir::AbiParam::new(ir::types::I64),
                ],
                returns: vec![ir::AbiParam::new(ir::types::I8)],
                call_conv,
            },
            Self::ParseFloat => ir::Signature {
                params: vec![
                    ir::AbiParam::new(pointer_type),
                    ir::AbiParam::new(ir::types::I64),
                ],
                returns: vec![ir::AbiParam::new(ir::types::F64)],
                call_conv,
            },
        }
    }

    fn name(&self) -> &'static str {
        match self {
            Self::ParseInt => "_bril_parse_int",
            Self::ParseBool => "_bril_parse_bool",
            Self::ParseFloat => "_bril_parse_float",
        }
    }
}

/// Translate a Bril type into a CLIF type.
fn translate_type(typ: &bril::Type, pointer_type: ir::Type) -> ir::Type {
    match typ {
        bril::Type::Int => ir::types::I64,
        bril::Type::Bool => ir::types::I8,
        bril::Type::Float => ir::types::F64,
        bril::Type::Char => ir::types::I32,
        bril::Type::Pointer(_) => pointer_type,
    }
}

/// Generate a CLIF signature for a Bril function.
fn translate_sig(func: &bril::Function, pointer_type: ir::Type) -> ir::Signature {
    let mut sig = ir::Signature::new(isa::CallConv::Fast);
    if let Some(ret) = &func.return_type {
        sig.returns
            .push(ir::AbiParam::new(translate_type(ret, pointer_type)));
    }
    for arg in &func.args {
        sig.params.push(ir::AbiParam::new(translate_type(
            &arg.arg_type,
            pointer_type,
        )));
    }
    sig
}

/// Get all the variables defined in a function (and their types), including the arguments.
fn all_vars(func: &bril::Function) -> HashMap<&String, &bril::Type> {
    func.instrs
        .iter()
        .filter_map(|inst| match inst {
            bril::Code::Instruction(op) => match op {
                bril::Instruction::Constant {
                    dest,
                    op: _,
                    const_type: typ,
                    value: _,
                    ..
                } => Some((dest, typ)),
                bril::Instruction::Value {
                    args: _,
                    dest,
                    funcs: _,
                    labels: _,
                    op: _,
                    op_type: typ,
                    ..
                } => Some((dest, typ)),
                _ => None,
            },
            _ => None,
        })
        .chain(func.args.iter().map(|arg| (&arg.name, &arg.arg_type)))
        .collect()
}

/// Emit Cranelift code to load a Bril value from memory.
fn emit_load(
    pointer_type: ir::Type,
    builder: &mut FunctionBuilder,
    typ: &bril::Type,
    ptr: ir::Value,
) -> ir::Value {
    let mem_type = translate_type(typ, pointer_type);
    builder
        .ins()
        .load(mem_type, ir::MemFlags::trusted(), ptr, 0)
}

/// Emit cranelift code to store a Bril value to memory.
fn emit_store(builder: &mut FunctionBuilder, ptr: ir::Value, val: ir::Value) {
    builder.ins().store(ir::MemFlags::trusted(), val, ptr, 0);
}

/// An environment for translating Bril into CLIF.
struct CompileEnv<'a> {
    vars: HashMap<&'a String, Variable>,
    var_types: HashMap<&'a String, &'a bril::Type>,
    rt_refs: EnumMap<RTFunc, ir::FuncRef>,
    blocks: HashMap<&'a String, ir::Block>,
    func_refs: HashMap<&'a String, ir::FuncRef>,
    pointer_type: ir::Type,
}

impl CompileEnv<'_> {
    /// Get the element size of the pointed-to type in bytes. `typ` must be a Bril pointer type.
    fn pointee_bytes(&self, typ: &bril::Type) -> u32 {
        let pointee_type = match typ {
            bril::Type::Pointer(t) => t,
            _ => panic!("alloc for non-pointer type"),
        };
        translate_type(pointee_type, self.pointer_type).bytes()
    }

    /// Generate a CLIF icmp instruction.
    fn gen_icmp(
        &self,
        builder: &mut FunctionBuilder,
        args: &[String],
        dest: &String,
        cc: ir::condcodes::IntCC,
    ) {
        let lhs = builder.use_var(self.vars[&args[0]]);
        let rhs = builder.use_var(self.vars[&args[1]]);
        let res = builder.ins().icmp(cc, lhs, rhs);
        builder.def_var(self.vars[dest], res);
    }

    /// Generate a CLIF fcmp instruction.
    fn gen_fcmp(
        &self,
        builder: &mut FunctionBuilder,
        args: &[String],
        dest: &String,
        cc: ir::condcodes::FloatCC,
    ) {
        let lhs = builder.use_var(self.vars[&args[0]]);
        let rhs = builder.use_var(self.vars[&args[1]]);
        let res = builder.ins().fcmp(cc, lhs, rhs);
        builder.def_var(self.vars[dest], res);
    }

    /// Generate a CLIF binary operator.
    fn gen_binary(
        &self,
        builder: &mut FunctionBuilder,
        args: &[String],
        dest: &String,
        dest_type: &bril::Type,
        op: ir::Opcode,
    ) {
        let lhs = builder.use_var(self.vars[&args[0]]);
        let rhs = builder.use_var(self.vars[&args[1]]);
        let typ = translate_type(dest_type, self.pointer_type);
        let (inst, dfg) = builder.ins().Binary(op, typ, lhs, rhs);
        let res = dfg.first_result(inst);
        builder.def_var(self.vars[dest], res);
    }

    /// Implement a Bril `print` instruction in CLIF.
    fn gen_print(&self, args: &[String], builder: &mut FunctionBuilder) {
        let mut first = true;
        for arg in args {
            // Separate printed values.
            if first {
                first = false;
            } else {
                builder.ins().call(self.rt_refs[RTFunc::PrintSep], &[]);
            }

            // Print each value according to its type.
            let arg_val = builder.use_var(self.vars[arg]);
            let print_func = match self.var_types[arg] {
                bril::Type::Int => RTFunc::PrintInt,
                bril::Type::Bool => RTFunc::PrintBool,
                bril::Type::Float => RTFunc::PrintFloat,
                bril::Type::Char => unimplemented!(),
                bril::Type::Pointer(_) => unimplemented!(),
            };
            let print_ref = self.rt_refs[print_func];
            builder.ins().call(print_ref, &[arg_val]);
        }
        builder.ins().call(self.rt_refs[RTFunc::PrintEnd], &[]);
    }

    /// Implement a Bril constant as a CLIF constant assignment.
    fn compile_const(
        &self,
        builder: &mut FunctionBuilder,
        typ: &bril::Type,
        lit: &bril::Literal,
    ) -> ir::Value {
        match typ {
            bril::Type::Int => {
                let val = match lit {
                    bril::Literal::Int(i) => *i,
                    _ => panic!("incorrect literal type for int"),
                };
                builder.ins().iconst(ir::types::I64, val)
            }
            bril::Type::Bool => {
                let val = match lit {
                    bril::Literal::Bool(b) => *b,
                    _ => panic!("incorrect literal type for bool"),
                };
                builder.ins().iconst(ir::types::I8, if val { 1 } else { 0 })
            }
            bril::Type::Float => {
                let val = match lit {
                    bril::Literal::Float(f) => *f,
                    bril::Literal::Int(i) => *i as f64,
                    _ => panic!("incorrect literal type for float"),
                };
                builder.ins().f64const(val)
            }
            bril::Type::Char => {
                let val = match lit {
                    bril::Literal::Char(c) => *c,
                    _ => panic!("incorrect literal type for char"),
                };
                builder.ins().iconst(ir::types::I32, val as i64)
            }
            bril::Type::Pointer(_) => panic!("pointer literals not allowed"),
        }
    }

    /// Translate Bril opcodes that have CLIF equivalents.
    fn translate_op(op: bril::ValueOps) -> ir::Opcode {
        match op {
            bril::ValueOps::Add => ir::Opcode::Iadd,
            bril::ValueOps::Sub => ir::Opcode::Isub,
            bril::ValueOps::Mul => ir::Opcode::Imul,
            bril::ValueOps::Div => ir::Opcode::Sdiv,
            bril::ValueOps::And => ir::Opcode::Band,
            bril::ValueOps::Or => ir::Opcode::Bor,
            bril::ValueOps::Fadd => ir::Opcode::Fadd,
            bril::ValueOps::Fsub => ir::Opcode::Fsub,
            bril::ValueOps::Fmul => ir::Opcode::Fmul,
            bril::ValueOps::Fdiv => ir::Opcode::Fdiv,
            _ => panic!("not a translatable opcode: {op}"),
        }
    }

    /// Translate Bril opcodes that correspond to CLIF integer comparisons.
    fn translate_intcc(op: bril::ValueOps) -> IntCC {
        match op {
            bril::ValueOps::Lt => IntCC::SignedLessThan,
            bril::ValueOps::Le => IntCC::SignedLessThanOrEqual,
            bril::ValueOps::Eq => IntCC::Equal,
            bril::ValueOps::Ge => IntCC::SignedGreaterThanOrEqual,
            bril::ValueOps::Gt => IntCC::SignedGreaterThan,
            _ => panic!("not a comparison opcode: {op}"),
        }
    }

    /// Translate Bril opcodes that correspond to CLIF floating point comparisons.
    fn translate_floatcc(op: bril::ValueOps) -> FloatCC {
        match op {
            bril::ValueOps::Flt => FloatCC::LessThan,
            bril::ValueOps::Fle => FloatCC::LessThanOrEqual,
            bril::ValueOps::Feq => FloatCC::Equal,
            bril::ValueOps::Fge => FloatCC::GreaterThanOrEqual,
            bril::ValueOps::Fgt => FloatCC::GreaterThan,
            _ => panic!("not a comparison opcode: {op}"),
        }
    }

    /// Compile one Bril instruction into CLIF.
    fn compile_inst(&self, inst: &bril::Instruction, builder: &mut FunctionBuilder) {
        match inst {
            bril::Instruction::Constant {
                dest,
                op: _,
                const_type: typ,
                value,
                ..
            } => {
                let val = self.compile_const(builder, typ, value);
                builder.def_var(self.vars[dest], val);
            }
            bril::Instruction::Effect {
                args,
                funcs,
                labels,
                op,
                ..
            } => match op {
                bril::EffectOps::Print => self.gen_print(args, builder),
                bril::EffectOps::Jump => {
                    builder.ins().jump(self.blocks[&labels[0]], &[]);
                }
                bril::EffectOps::Branch => {
                    let arg = builder.use_var(self.vars[&args[0]]);
                    let true_block = self.blocks[&labels[0]];
                    let false_block = self.blocks[&labels[1]];
                    builder.ins().brif(arg, true_block, &[], false_block, &[]);
                }
                bril::EffectOps::Call => {
                    let func_ref = self.func_refs[&funcs[0]];
                    let arg_vals: Vec<ir::Value> = args
                        .iter()
                        .map(|arg| builder.use_var(self.vars[arg]))
                        .collect();
                    builder.ins().call(func_ref, &arg_vals);
                }
                bril::EffectOps::Return => {
                    if !args.is_empty() {
                        let arg = builder.use_var(self.vars[&args[0]]);
                        builder.ins().return_(&[arg]);
                    } else {
                        builder.ins().return_(&[]);
                    }
                }
                bril::EffectOps::Nop => {}
                bril::EffectOps::Store => {
                    let ptr_arg = builder.use_var(self.vars[&args[0]]);
                    let val_arg = builder.use_var(self.vars[&args[1]]);
                    emit_store(builder, ptr_arg, val_arg);
                }
                bril::EffectOps::Free => {
                    let ptr_arg = builder.use_var(self.vars[&args[0]]);
                    builder.ins().call(self.rt_refs[RTFunc::Free], &[ptr_arg]);
                }
                bril::EffectOps::Speculate | bril::EffectOps::Commit | bril::EffectOps::Guard => {
                    unimplemented!()
                }
            },
            bril::Instruction::Value {
                args,
                dest,
                funcs,
                labels: _,
                op,
                op_type,
                ..
            } => match op {
                bril::ValueOps::Add
                | bril::ValueOps::Sub
                | bril::ValueOps::Mul
                | bril::ValueOps::Div
                | bril::ValueOps::And
                | bril::ValueOps::Or => {
                    self.gen_binary(builder, args, dest, op_type, Self::translate_op(*op));
                }
                bril::ValueOps::Lt
                | bril::ValueOps::Le
                | bril::ValueOps::Eq
                | bril::ValueOps::Ge
                | bril::ValueOps::Gt => {
                    self.gen_icmp(builder, args, dest, Self::translate_intcc(*op))
                }
                bril::ValueOps::Not => {
                    let arg = builder.use_var(self.vars[&args[0]]);

                    // Logical "not." The IR only has bitwise not.
                    let zero = builder.ins().iconst(ir::types::I8, 0);
                    let one = builder.ins().iconst(ir::types::I8, 1);
                    let res = builder.ins().select(arg, zero, one);
                    builder.def_var(self.vars[dest], res);
                }
                bril::ValueOps::Call => {
                    let func_ref = self.func_refs[&funcs[0]];
                    let arg_vals: Vec<ir::Value> = args
                        .iter()
                        .map(|arg| builder.use_var(self.vars[arg]))
                        .collect();
                    let inst = builder.ins().call(func_ref, &arg_vals);
                    let res = builder.inst_results(inst)[0];
                    builder.def_var(self.vars[dest], res);
                }
                bril::ValueOps::Id => {
                    let arg = builder.use_var(self.vars[&args[0]]);
                    builder.def_var(self.vars[dest], arg);
                }

                // Floating point extension.
                bril::ValueOps::Fadd
                | bril::ValueOps::Fsub
                | bril::ValueOps::Fmul
                | bril::ValueOps::Fdiv => {
                    self.gen_binary(builder, args, dest, op_type, Self::translate_op(*op));
                }
                bril::ValueOps::Flt
                | bril::ValueOps::Fle
                | bril::ValueOps::Feq
                | bril::ValueOps::Fge
                | bril::ValueOps::Fgt => {
                    self.gen_fcmp(builder, args, dest, Self::translate_floatcc(*op))
                }

                // Memory extension.
                bril::ValueOps::Alloc => {
                    // The number of elements to allocate comes from the program.
                    let count_val = builder.use_var(self.vars[&args[0]]);

                    // The bytes per element depends on the type.
                    let elem_bytes = self.pointee_bytes(op_type);
                    let bytes_val = builder.ins().iconst(ir::types::I64, elem_bytes as i64);

                    // Call the allocate function.
                    let inst = builder
                        .ins()
                        .call(self.rt_refs[RTFunc::Alloc], &[count_val, bytes_val]);
                    let res = builder.inst_results(inst)[0];
                    builder.def_var(self.vars[dest], res);
                }
                bril::ValueOps::Load => {
                    let ptr = builder.use_var(self.vars[&args[0]]);
                    let val = emit_load(self.pointer_type, builder, op_type, ptr);
                    builder.def_var(self.vars[dest], val);
                }
                bril::ValueOps::PtrAdd => {
                    let orig_ptr = builder.use_var(self.vars[&args[0]]);
                    let amt = builder.use_var(self.vars[&args[1]]);

                    let size = self.pointee_bytes(op_type);
                    let offset_val = builder.ins().imul_imm(amt, size as i64);

                    let res = builder.ins().iadd(orig_ptr, offset_val);
                    builder.def_var(self.vars[dest], res);
                }
                bril::ValueOps::Phi
                | bril::ValueOps::Ceq
                | bril::ValueOps::Clt
                | bril::ValueOps::Cgt
                | bril::ValueOps::Cle
                | bril::ValueOps::Cge
                | bril::ValueOps::Char2int
                | bril::ValueOps::Int2char => unimplemented!(),
            },
        }
    }

    /// Is a given Bril instruction a basic block terminator?
    fn is_term(inst: &bril::Instruction) -> bool {
        if let bril::Instruction::Effect {
            args: _,
            funcs: _,
            labels: _,
            op,
            ..
        } = inst
        {
            matches!(
                op,
                bril::EffectOps::Branch | bril::EffectOps::Jump | bril::EffectOps::Return
            )
        } else {
            false
        }
    }

    /// Emit the body of a Bril function into a CLIF function.
    fn compile_body(&self, insts: &[bril::Code], builder: &mut FunctionBuilder) {
        let mut terminated = false; // Entry block is open.
        for code in insts {
            match code {
                bril::Code::Instruction(inst) => {
                    // If a normal instruction immediately follows a terminator, we need a new (anonymous) block.
                    if terminated {
                        let block = builder.create_block();
                        builder.switch_to_block(block);
                        terminated = false;
                    }

                    // Compile one instruction.
                    self.compile_inst(inst, builder);

                    if Self::is_term(inst) {
                        terminated = true;
                    }
                }
                bril::Code::Label { label, .. } => {
                    let new_block = self.blocks[label];

                    // If the previous block was missing a terminator (fall-through), insert a
                    // jump to the new block.
                    if !terminated {
                        builder.ins().jump(new_block, &[]);
                    }
                    terminated = false;

                    builder.switch_to_block(new_block);
                }
            }
        }

        // Implicit return in the last block.
        if !terminated {
            builder.ins().return_(&[]);
        }
    }
}

/// A compiler from an entire Bril program to a CLIF module.
pub struct Translator<M: Module> {
    rt_funcs: EnumMap<RTFunc, cranelift_module::FuncId>,
    module: M,
    context: cranelift_codegen::Context,
    funcs: HashMap<String, cranelift_module::FuncId>,
}

impl<M: Module> Translator<M> {
    /// Declare all our runtime functions in a CLIF module.
    fn declare_rt(module: &mut M) -> EnumMap<RTFunc, cranelift_module::FuncId> {
        enum_map! {
            rtfunc =>
                module
                    .declare_function(
                        rtfunc.name(),
                        cranelift_module::Linkage::Import,
                        &rtfunc.sig(module.isa().pointer_type(), module.isa().default_call_conv()),
                    )
                    .unwrap()
        }
    }

    fn declare_func(&mut self, func: &bril::Function) -> cranelift_module::FuncId {
        // The Bril `main` function gets a different internal name, and we call it from a new
        // proper main function that gets argv/argc.
        let name = if func.name == "main" {
            "__bril_main"
        } else {
            &func.name
        };

        let sig = translate_sig(func, self.module.isa().pointer_type());
        self.module
            .declare_function(name, cranelift_module::Linkage::Local, &sig)
            .unwrap()
    }

    fn enter_func(&mut self, func: &bril::Function, func_id: cranelift_module::FuncId) {
        let sig = translate_sig(func, self.module.isa().pointer_type());
        self.context.func =
            ir::Function::with_name_signature(ir::UserFuncName::user(0, func_id.as_u32()), sig);
    }

    fn finish_func(&mut self, func_id: cranelift_module::FuncId, dump: bool) {
        // Print the IR, if requested.
        if dump {
            println!("{}", self.context.func.display());
        }

        // Add to the module.
        self.module
            .define_function(func_id, &mut self.context)
            .unwrap();
        self.context.clear();
    }

    fn compile_func(&mut self, func: &bril::Function) {
        let mut fn_builder_ctx = FunctionBuilderContext::new();
        let mut builder = FunctionBuilder::new(&mut self.context.func, &mut fn_builder_ctx);

        // Declare runtime functions.
        let rt_refs = self
            .rt_funcs
            .map(|_, id| self.module.declare_func_in_func(id, builder.func));

        // Declare all variables (including for function parameters).
        let var_types = all_vars(func);
        let vars: HashMap<&String, Variable> = var_types
            .iter()
            .enumerate()
            .map(|(i, (name, typ))| {
                let var = Variable::new(i);
                builder.declare_var(var, translate_type(typ, self.module.isa().pointer_type()));
                (*name, var)
            })
            .collect();

        // Create blocks for every label.
        let blocks: HashMap<&String, ir::Block> = func
            .instrs
            .iter()
            .filter_map(|code| match code {
                bril::Code::Label { label, .. } => {
                    let block = builder.create_block();
                    Some((label, block))
                }
                _ => None,
            })
            .collect();

        // "Import" all the functions we may need to call.
        // TODO We could do this only for the functions we actually use...
        let func_refs: HashMap<&String, ir::FuncRef> = self
            .funcs
            .iter()
            .map(|(name, id)| (name, self.module.declare_func_in_func(*id, builder.func)))
            .collect();

        let env = CompileEnv {
            vars,
            var_types,
            rt_refs,
            blocks,
            func_refs,
            pointer_type: self.module.isa().pointer_type(),
        };

        // Define variables for function arguments in the entry block.
        let entry_block = builder.create_block();
        builder.switch_to_block(entry_block);
        builder.append_block_params_for_function_params(entry_block);
        for (i, arg) in func.args.iter().enumerate() {
            let param = builder.block_params(entry_block)[i];
            builder.def_var(env.vars[&arg.name], param);
        }

        // Insert instructions.
        env.compile_body(&func.instrs, &mut builder);

        builder.seal_all_blocks();
        builder.finalize();
    }

    /// Generate a C-style `main` function that parses command-line arguments and then calls the
    /// Bril `main` function.
    pub fn add_c_main(&mut self, args: &[bril::Argument], dump: bool) -> cranelift_module::FuncId {
        // Declare `main` with argc/argv parameters.
        let pointer_type = self.module.isa().pointer_type();
        let sig = ir::Signature {
            params: vec![
                ir::AbiParam::new(pointer_type),
                ir::AbiParam::new(pointer_type),
            ],
            returns: vec![ir::AbiParam::new(pointer_type)],
            call_conv: self.module.isa().default_call_conv(),
        };
        let main_id = self
            .module
            .declare_function("main", cranelift_module::Linkage::Export, &sig)
            .unwrap();

        self.context.func =
            ir::Function::with_name_signature(ir::UserFuncName::user(0, main_id.as_u32()), sig);

        // Declare `main`-specific setup runtime functions.
        let call_conv = self.module.isa().default_call_conv();
        let rt_setup_refs: EnumMap<RTSetupFunc, ir::FuncRef> = enum_map! {
            rt_setup_func => {
                let func_id = self
                    .module
                    .declare_function(
                        rt_setup_func.name(),
                        cranelift_module::Linkage::Import,
                        &rt_setup_func.sig(pointer_type, call_conv),
                    )
                    .unwrap();
                self
                    .module
                    .declare_func_in_func(func_id, &mut self.context.func)
            }
        };

        let mut fn_builder_ctx = FunctionBuilderContext::new();
        let mut builder = FunctionBuilder::new(&mut self.context.func, &mut fn_builder_ctx);

        let block = builder.create_block();
        builder.switch_to_block(block);
        builder.seal_block(block);
        builder.append_block_params_for_function_params(block);

        // Parse each argument.
        let argv_arg = builder.block_params(block)[1]; // argc, argv
        let arg_vals: Vec<ir::Value> = args
            .iter()
            .enumerate()
            .map(|(i, arg)| {
                let parse_ref = rt_setup_refs[match arg.arg_type {
                    bril::Type::Int => RTSetupFunc::ParseInt,
                    bril::Type::Bool => RTSetupFunc::ParseBool,
                    bril::Type::Float => RTSetupFunc::ParseFloat,
                    bril::Type::Char => unimplemented!(),
                    bril::Type::Pointer(_) => unimplemented!("can't print pointers"),
                }];
                let idx_arg = builder.ins().iconst(ir::types::I64, (i + 1) as i64); // skip argv[0]
                let inst = builder.ins().call(parse_ref, &[argv_arg, idx_arg]);
                builder.inst_results(inst)[0]
            })
            .collect();

        // Call the "real" main function.
        let real_main_id = self.funcs["main"];
        let real_main_ref = self.module.declare_func_in_func(real_main_id, builder.func);
        builder.ins().call(real_main_ref, &arg_vals);

        // Return 0 from `main`.
        let zero = builder.ins().iconst(self.module.isa().pointer_type(), 0);
        builder.ins().return_(&[zero]);
        builder.finalize();

        // Add to the module.
        if dump {
            println!("{}", self.context.func.display());
        }
        self.module
            .define_function(main_id, &mut self.context)
            .unwrap();
        self.context.clear();

        main_id
    }

    /// Add a function that wraps a Bril function to invoke it with arguments that come from
    /// memory. The new function takes a single pointer as an argument, which points to an array of
    /// pointers to the arguments.
    pub fn add_mem_wrapper(
        &mut self,
        name: &str,
        args: &[bril::Argument],
        dump: bool,
    ) -> cranelift_module::FuncId {
        // Declare wrapper function.
        let pointer_type = self.module.isa().pointer_type();
        let sig = ir::Signature {
            params: vec![ir::AbiParam::new(pointer_type)],
            returns: vec![],
            call_conv: self.module.isa().default_call_conv(),
        };
        let wrapped_name = format!("{name}_wrapper");
        let wrapper_id = self
            .module
            .declare_function(&wrapped_name, cranelift_module::Linkage::Export, &sig)
            .unwrap();

        self.context.func =
            ir::Function::with_name_signature(ir::UserFuncName::user(0, wrapper_id.as_u32()), sig);
        let mut fn_builder_ctx = FunctionBuilderContext::new();
        let mut builder = FunctionBuilder::new(&mut self.context.func, &mut fn_builder_ctx);

        let block = builder.create_block();
        builder.switch_to_block(block);
        builder.seal_block(block);
        builder.append_block_params_for_function_params(block);

        // Load every argument from memory.
        let base_ptr = builder.block_params(block)[0];
        let ptr_size = pointer_type.bytes();
        let arg_vals: Vec<ir::Value> = args
            .iter()
            .enumerate()
            .map(|(i, arg)| {
                // Load the pointer.
                let offset = (ptr_size * (i as u32)) as i32;
                let arg_ptr =
                    builder
                        .ins()
                        .load(pointer_type, ir::MemFlags::trusted(), base_ptr, offset);

                // Load the argument value.
                emit_load(pointer_type, &mut builder, &arg.arg_type, arg_ptr)
            })
            .collect();

        // Call the "real" main function.
        let real_func_id = self.funcs[name];
        let real_func_ref = self.module.declare_func_in_func(real_func_id, builder.func);
        builder.ins().call(real_func_ref, &arg_vals);

        builder.ins().return_(&[]);

        // Add to the module.
        if dump {
            println!("{}", self.context.func.display());
        }
        self.module
            .define_function(wrapper_id, &mut self.context)
            .unwrap();
        self.context.clear();

        wrapper_id
    }

    pub fn compile_prog(&mut self, prog: &bril::Program, dump: bool) {
        // Declare all functions.
        for func in &prog.functions {
            let id = self.declare_func(func);
            self.funcs.insert(func.name.to_owned(), id);
        }

        // Define all functions.
        for func in &prog.functions {
            let id = self.funcs[&func.name];
            self.enter_func(func, id);
            self.compile_func(func);
            self.finish_func(id, dump);
        }
    }
}

/// AOT compiler that generates `.o` files.
impl Translator<ObjectModule> {
    /// Configure a Cranelift target ISA object.
    fn get_isa(
        target: Option<String>,
        pic: bool,
        opt_level: &str,
    ) -> Arc<dyn cranelift_codegen::isa::TargetIsa> {
        let mut flag_builder = settings::builder();
        flag_builder
            .set("opt_level", opt_level)
            .expect("invalid opt level");
        if pic {
            flag_builder.set("is_pic", "true").unwrap();
        }
        let isa_builder = if let Some(targ) = target {
            cranelift_codegen::isa::lookup_by_name(&targ).expect("invalid target")
        } else {
            cranelift_native::builder().unwrap()
        };
        isa_builder
            .finish(settings::Flags::new(flag_builder))
            .unwrap()
    }

    pub fn new(target: Option<String>, opt_level: &str) -> Self {
        // Make an object module.
        let isa = Self::get_isa(target, true, opt_level);
        let mut module =
            ObjectModule::new(ObjectBuilder::new(isa, "foo", default_libcall_names()).unwrap());

        Self {
            rt_funcs: Self::declare_rt(&mut module),
            module,
            context: cranelift_codegen::Context::new(),
            funcs: HashMap::new(),
        }
    }

    pub fn emit(self, output: &str) {
        let prod = self.module.finish();
        let objdata = prod.emit().expect("emission failed");
        fs::write(output, objdata).expect("failed to write .o file");
    }
}

/// A JIT compiler.
impl Translator<JITModule> {
    // `cranelift_jit` does not yet support PIC on AArch64:
    // https://github.com/bytecodealliance/wasmtime/issues/2735
    // The default initialization path for `JITBuilder` is hard-coded to use PIC, so we manually
    // disable it here. Once this is fully supported in `cranelift_jit`, we can switch to the
    // generic versin below unconditionally.
    #[cfg(target_arch = "aarch64")]
    fn jit_builder() -> JITBuilder {
        let mut flag_builder = settings::builder();
        flag_builder.set("use_colocated_libcalls", "false").unwrap();
        flag_builder.set("is_pic", "false").unwrap(); // PIC unsupported on ARM.
        let isa_builder = cranelift_native::builder().unwrap();
        let isa = isa_builder
            .finish(settings::Flags::new(flag_builder))
            .unwrap();
        JITBuilder::with_isa(isa, cranelift_module::default_libcall_names())
    }

    // The normal way to set up a JIT builder.
    #[cfg(not(target_arch = "aarch64"))]
    pub fn jit_builder() -> JITBuilder {
        JITBuilder::new(cranelift_module::default_libcall_names()).unwrap()
    }

    pub fn new() -> Self {
        // Set up the JIT.
        let mut builder = Self::jit_builder();

        // Provide runtime functions.
        enum_map! {
            rtfunc => {
                let f: RTFunc = rtfunc;
                builder.symbol(f.name(), f.rt_impl());
            }
        };

        let mut module = JITModule::new(builder);

        Self {
            rt_funcs: Self::declare_rt(&mut module),
            context: module.make_context(),
            module,
            funcs: HashMap::new(),
        }
    }

    /// Obtain an entry-point code pointer. The pointer remains valid as long as the translator
    /// itself (and therefore the `JITModule`) lives.
    fn get_func_ptr(&mut self, func_id: cranelift_module::FuncId) -> *const u8 {
        self.module.clear_context(&mut self.context);
        self.module.finalize_definitions().unwrap();

        self.module.get_finalized_function(func_id)
    }

    fn val_ptrs(vals: &[bril::Literal]) -> Vec<*const u8> {
        vals.iter()
            .map(|lit| match lit {
                bril::Literal::Int(i) => i as *const i64 as *const u8,
                bril::Literal::Bool(b) => b as *const bool as *const u8,
                bril::Literal::Float(f) => f as *const f64 as *const u8,
                bril::Literal::Char(c) => c as *const char as *const u8,
            })
            .collect()
    }

    /// Run a JITted wrapper function.
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn run(&mut self, func_id: cranelift_module::FuncId, args: &[bril::Literal]) {
        let func_ptr = self.get_func_ptr(func_id);
        let arg_ptrs = Self::val_ptrs(args);
        let func = mem::transmute::<*const u8, fn(*const *const u8) -> ()>(func_ptr);
        func(arg_ptrs.as_ptr());
    }
}

impl Default for Translator<JITModule> {
    fn default() -> Self {
        Self::new()
    }
}

pub fn find_func<'a>(funcs: &'a [bril::Function], name: &str) -> &'a bril::Function {
    funcs.iter().find(|f| f.name == name).unwrap()
}
