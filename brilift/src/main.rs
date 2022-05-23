use argh::FromArgs;
use bril_rs as bril;
use cranelift_codegen::entity::EntityRef;
use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::InstBuilder;
use cranelift_codegen::{ir, isa, settings};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{default_libcall_names, Module};
use cranelift_object::{ObjectBuilder, ObjectModule};
use std::collections::HashMap;
use std::fs;

struct RTSigs {
    print_int: ir::Signature,
}

struct RTIds {
    print_int: cranelift_module::FuncId,
}

struct RTRefs {
    print_int: ir::FuncRef,
}

fn tr_type(typ: &bril::Type) -> ir::Type {
    match typ {
        bril::Type::Int => ir::types::I64,
        bril::Type::Bool => ir::types::B1,
    }
}

fn tr_sig(func: &bril::Function) -> ir::Signature {
    let mut sig = ir::Signature::new(isa::CallConv::SystemV);
    if let Some(ret) = &func.return_type {
        sig.returns.push(ir::AbiParam::new(tr_type(ret)));
    }
    for arg in &func.args {
        sig.params.push(ir::AbiParam::new(tr_type(&arg.arg_type)));
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
                } => Some((dest, typ)),
                bril::Instruction::Value {
                    args: _,
                    dest,
                    funcs: _,
                    labels: _,
                    op: _,
                    op_type: typ,
                } => Some((dest, typ)),
                _ => None,
            },
            _ => None,
        })
        .chain(func.args.iter().map(|arg| (&arg.name, &arg.arg_type)))
        .collect()
}

// TODO Should really be a trait with two different structs that implement it?
struct Translator<M: Module> {
    rt_funcs: RTIds,
    module: M,
    context: cranelift_codegen::Context,
    funcs: HashMap<String, cranelift_module::FuncId>,
}

// TODO Should this be a constant or something?
fn get_rt_sigs() -> RTSigs {
    RTSigs {
        print_int: ir::Signature {
            params: vec![ir::AbiParam::new(ir::types::I64)],
            returns: vec![],
            call_conv: isa::CallConv::SystemV,
        },
    }
}

fn declare_rt<M: Module>(module: &mut M) -> RTIds {
    // TODO Maybe these should be hash tables or something?
    let rt_sigs = get_rt_sigs();
    RTIds {
        print_int: {
            module
                .declare_function(
                    "print_int",
                    cranelift_module::Linkage::Import,
                    &rt_sigs.print_int,
                )
                .unwrap()
        },
    }
}

impl Translator<ObjectModule> {
    fn new(target: Option<String>) -> Self {
        // Make an object module.
        let flag_builder = settings::builder();
        let isa = if let Some(targ) = target {
            let isa_builder =
                cranelift_codegen::isa::lookup_by_name(&targ).expect("invalid target");
            isa_builder
                .finish(settings::Flags::new(flag_builder))
                .unwrap()
        } else {
            let isa_builder = cranelift_native::builder().unwrap();
            isa_builder
                .finish(settings::Flags::new(flag_builder))
                .unwrap()
        };
        let mut module =
            ObjectModule::new(ObjectBuilder::new(isa, "foo", default_libcall_names()).unwrap());

        Self {
            rt_funcs: declare_rt(&mut module),
            module,
            context: cranelift_codegen::Context::new(),
            funcs: HashMap::new(),
        }
    }

    fn emit(self, output: &str) {
        let prod = self.module.finish();
        let objdata = prod.emit().expect("emission failed");
        fs::write(output, objdata).expect("failed to write .o file");
    }
}

impl Translator<JITModule> {
    fn new() -> Self {
        // Cranelift JIT scaffolding.
        let builder = JITBuilder::new(cranelift_module::default_libcall_names()).unwrap();
        let mut module = JITModule::new(builder);

        Self {
            rt_funcs: declare_rt(&mut module),
            context: module.make_context(),
            module,
            funcs: HashMap::new(),
        }
    }

    fn compile(mut self) -> *const u8 {
        self.module.clear_context(&mut self.context);
        self.module.finalize_definitions();

        // TODO Compile all functions.
        let id = self.funcs.get("main").unwrap();
        self.module.get_finalized_function(*id)
    }
}

fn is_term(inst: &bril::Instruction) -> bool {
    if let bril::Instruction::Effect {
        args: _,
        funcs: _,
        labels: _,
        op,
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

fn gen_icmp(
    builder: &mut FunctionBuilder,
    vars: &HashMap<String, Variable>,
    args: &[String],
    dest: &String,
    cc: IntCC,
) {
    let lhs = builder.use_var(*vars.get(&args[0]).unwrap());
    let rhs = builder.use_var(*vars.get(&args[1]).unwrap());
    let res = builder.ins().icmp(cc, lhs, rhs);
    builder.def_var(*vars.get(dest).unwrap(), res);
}

fn gen_binary(
    builder: &mut FunctionBuilder,
    vars: &HashMap<String, Variable>,
    args: &[String],
    dest: &String,
    dest_type: &bril::Type,
    op: ir::Opcode,
) {
    let lhs = builder.use_var(*vars.get(&args[0]).unwrap());
    let rhs = builder.use_var(*vars.get(&args[1]).unwrap());
    let typ = tr_type(dest_type);
    let (inst, dfg) = builder.ins().Binary(op, typ, lhs, rhs);
    let res = dfg.first_result(inst);
    builder.def_var(*vars.get(dest).unwrap(), res);
}

fn compile_inst(
    inst: &bril::Instruction,
    builder: &mut FunctionBuilder,
    vars: &HashMap<String, Variable>,
    rt_refs: &RTRefs,
    blocks: &HashMap<String, ir::Block>,
    func_refs: &HashMap<String, ir::FuncRef>,
) {
    match inst {
        bril::Instruction::Constant {
            dest,
            op: _,
            const_type: _,
            value,
        } => {
            let var = vars.get(dest).unwrap();
            let val = match value {
                bril::Literal::Int(i) => builder.ins().iconst(ir::types::I64, *i),
                bril::Literal::Bool(b) => builder.ins().bconst(ir::types::B1, *b),
            };
            builder.def_var(*var, val);
        }
        bril::Instruction::Effect {
            args,
            funcs,
            labels,
            op,
        } => {
            match op {
                bril::EffectOps::Print => {
                    // TODO Target should depend on the type.
                    // TODO Deal with multiple args somehow.
                    let arg = builder.use_var(*vars.get(&args[0]).unwrap());
                    builder.ins().call(rt_refs.print_int, &[arg]);
                }
                bril::EffectOps::Jump => {
                    let block = *blocks.get(&labels[0]).unwrap();
                    builder.ins().jump(block, &[]);
                }
                bril::EffectOps::Branch => {
                    let arg = builder.use_var(*vars.get(&args[0]).unwrap());
                    let true_block = *blocks.get(&labels[0]).unwrap();
                    let false_block = *blocks.get(&labels[1]).unwrap();
                    builder.ins().brnz(arg, true_block, &[]);
                    builder.ins().jump(false_block, &[]);
                }
                bril::EffectOps::Call => {
                    let func_ref = *func_refs.get(&funcs[0]).unwrap();
                    let arg_vals: Vec<ir::Value> = args
                        .iter()
                        .map(|arg| builder.use_var(*vars.get(arg).unwrap()))
                        .collect();
                    builder.ins().call(func_ref, &arg_vals);
                }
                bril::EffectOps::Return => {
                    if !args.is_empty() {
                        let arg = builder.use_var(*vars.get(&args[0]).unwrap());
                        builder.ins().return_(&[arg]);
                    } else {
                        builder.ins().return_(&[]);
                    }
                }
                bril::EffectOps::Nop => {}
            }
        }
        bril::Instruction::Value {
            args,
            dest,
            funcs,
            labels: _,
            op,
            op_type,
        } => match op {
            bril::ValueOps::Add => gen_binary(builder, vars, args, dest, op_type, ir::Opcode::Iadd),
            bril::ValueOps::Sub => gen_binary(builder, vars, args, dest, op_type, ir::Opcode::Isub),
            bril::ValueOps::Mul => gen_binary(builder, vars, args, dest, op_type, ir::Opcode::Imul),
            bril::ValueOps::Div => gen_binary(builder, vars, args, dest, op_type, ir::Opcode::Sdiv),
            bril::ValueOps::Lt => gen_icmp(builder, vars, args, dest, IntCC::SignedLessThan),
            bril::ValueOps::Le => gen_icmp(builder, vars, args, dest, IntCC::SignedLessThanOrEqual),
            bril::ValueOps::Eq => gen_icmp(builder, vars, args, dest, IntCC::Equal),
            bril::ValueOps::Ge => {
                gen_icmp(builder, vars, args, dest, IntCC::SignedGreaterThanOrEqual)
            }
            bril::ValueOps::Gt => gen_icmp(builder, vars, args, dest, IntCC::SignedGreaterThan),
            bril::ValueOps::And => gen_binary(builder, vars, args, dest, op_type, ir::Opcode::Band),
            bril::ValueOps::Or => gen_binary(builder, vars, args, dest, op_type, ir::Opcode::Bor),
            bril::ValueOps::Not => {
                let arg = builder.use_var(*vars.get(&args[0]).unwrap());
                let res = builder.ins().bnot(arg);
                builder.def_var(*vars.get(dest).unwrap(), res);
            }
            bril::ValueOps::Call => {
                let func_ref = *func_refs.get(&funcs[0]).unwrap();
                let arg_vals: Vec<ir::Value> = args
                    .iter()
                    .map(|arg| builder.use_var(*vars.get(arg).unwrap()))
                    .collect();
                let inst = builder.ins().call(func_ref, &arg_vals);
                let res = builder.inst_results(inst)[0];
                builder.def_var(*vars.get(dest).unwrap(), res);
            }
            bril::ValueOps::Id => {
                let arg = builder.use_var(*vars.get(&args[0]).unwrap());
                builder.def_var(*vars.get(dest).unwrap(), arg);
            }
        },
    }
}

impl<M: Module> Translator<M> {
    fn declare_func(&mut self, func: &bril::Function) -> cranelift_module::FuncId {
        let sig = tr_sig(func);
        self.module
            .declare_function(&func.name, cranelift_module::Linkage::Export, &sig)
            .unwrap()
    }

    fn enter_func(&mut self, func: &bril::Function, func_id: cranelift_module::FuncId) {
        let sig = tr_sig(func);
        self.context.func =
            ir::Function::with_name_signature(ir::ExternalName::user(0, func_id.as_u32()), sig);
    }

    fn finish_func(&mut self, func_id: cranelift_module::FuncId, dump: bool) {
        // Print the IR, if requested.
        if dump {
            println!("{}", self.context.func.display());
        }

        // Verify the function (in debug mode).
        #[cfg(debug_assertions)]
        {
            let flags = settings::Flags::new(settings::builder());
            let res = cranelift_codegen::verifier::verify_function(&self.context.func, &flags);
            if let Err(errors) = res {
                panic!("{}", errors);
            }
        }

        // Add to the module.
        self.module
            .define_function(func_id, &mut self.context)
            .unwrap();
    }

    fn emit_func(&mut self, func: bril::Function) {
        let mut fn_builder_ctx = FunctionBuilderContext::new();
        let mut builder = FunctionBuilder::new(&mut self.context.func, &mut fn_builder_ctx);

        // Declare runtime functions.
        let rt_refs = RTRefs {
            print_int: self
                .module
                .declare_func_in_func(self.rt_funcs.print_int, builder.func),
        };

        // Declare all variables (including for function parameters).
        let mut vars = HashMap::<String, Variable>::new();
        for (i, (name, typ)) in all_vars(&func).iter().enumerate() {
            let var = Variable::new(i);
            builder.declare_var(var, tr_type(typ));
            vars.insert(name.to_string(), var);
        }

        // Create blocks for every label.
        let mut blocks = HashMap::<String, ir::Block>::new();
        for code in &func.instrs {
            if let bril::Code::Label { label } = code {
                let block = builder.create_block();
                blocks.insert(label.to_string(), block);
            }
        }

        // "Import" all the functions we may need to call.
        // TODO We could do this only for the functions we actually use...
        let func_refs: HashMap<String, ir::FuncRef> = self
            .funcs
            .iter()
            .map(|(name, id)| {
                (
                    name.to_owned(),
                    self.module.declare_func_in_func(*id, builder.func),
                )
            })
            .collect();

        // Define variables for function arguments in the entry block.
        let entry_block = builder.create_block();
        builder.switch_to_block(entry_block);
        builder.append_block_params_for_function_params(entry_block);
        for (i, arg) in func.args.iter().enumerate() {
            let param = builder.block_params(entry_block)[i];
            builder.def_var(vars[&arg.name], param);
        }

        // Insert instructions.
        let mut terminated = false; // Entry block is open.
        for code in &func.instrs {
            match code {
                bril::Code::Instruction(inst) => {
                    // If a normal instruction immediately follows a terminator, we need a new (anonymous) block.
                    if terminated {
                        let block = builder.create_block();
                        builder.switch_to_block(block);
                        terminated = false;
                    }

                    // Compile one instruction.
                    compile_inst(inst, &mut builder, &vars, &rt_refs, &blocks, &func_refs);

                    if is_term(inst) {
                        terminated = true;
                    }
                }
                bril::Code::Label { label } => {
                    let new_block = *blocks.get(label).unwrap();

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

        builder.seal_all_blocks();
        builder.finalize();
    }

    fn compile_prog(&mut self, prog: bril::Program, dump: bool) {
        // Declare all functions.
        for func in &prog.functions {
            let id = self.declare_func(func);
            self.funcs.insert(func.name.to_owned(), id);
        }

        // Define all functions.
        for func in prog.functions {
            let id = *self.funcs.get(&func.name).unwrap();
            self.enter_func(&func, id);
            self.emit_func(func);
            self.finish_func(id, dump);
        }
    }
}

#[derive(FromArgs)]
#[argh(description = "Bril compiler")]
struct Args {
    #[argh(switch, short = 'j', description = "JIT and run")]
    jit: bool,

    #[argh(option, short = 't', description = "target triple")]
    target: Option<String>,

    #[argh(
        option,
        short = 'o',
        description = "output file",
        default = "String::from(\"bril.o\")"
    )]
    output: String,

    #[argh(switch, short = 'd', description = "dump CLIF IR")]
    dump_ir: bool,
}

fn main() {
    let args: Args = argh::from_env();

    // Load the Bril program from stdin.
    let prog = bril::load_program();

    if args.jit {
        let mut trans = Translator::<JITModule>::new();
        trans.compile_prog(prog, args.dump_ir);
        trans.compile();
    } else {
        let mut trans = Translator::<ObjectModule>::new(args.target);
        trans.compile_prog(prog, args.dump_ir);
        trans.emit(&args.output);
    }
}
