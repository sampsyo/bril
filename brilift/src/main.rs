use argh::FromArgs;
use bril_rs as bril;
use cranelift_codegen::entity::EntityRef;
use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::InstBuilder;
use cranelift_codegen::settings::Configurable;
use cranelift_codegen::{ir, isa, settings};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{default_libcall_names, FuncOrDataId, Module};
use cranelift_object::{ObjectBuilder, ObjectModule};
use std::collections::HashMap;
use std::fs;

trait RTFuncUtils {
    fn sig(
        &self,
        pointer_type: ir::Type,
        call_conv: cranelift_codegen::isa::CallConv,
    ) -> ir::Signature;

    fn name(&self) -> &'static str;

    fn get_func_id<M: Module>(
        &self,
        module: &mut M,
        call_conv: cranelift_codegen::isa::CallConv,
    ) -> cranelift_module::FuncId {
        match module.get_name(self.name()) {
            Some(FuncOrDataId::Func(f)) => f,
            Some(_) => panic!("Expected {} to be a function, not data", self.name()),
            None => module
                .declare_function(
                    self.name(),
                    cranelift_module::Linkage::Import,
                    &self.sig(module.isa().pointer_type(), call_conv),
                )
                .unwrap(),
        }
    }

    fn get_func_ref<M: Module>(
        &self,
        module: &mut M,
        f: &mut ir::Function,
        call_conv: cranelift_codegen::isa::CallConv,
    ) -> ir::FuncRef {
        let f_id = self.get_func_id(module, call_conv);
        module.declare_func_in_func(f_id, f)
    }
}

/// The RTFuncUtils gives you this function for those structs, however, sometimes you just want to get the reference for an already declared function based off of the string name.
fn get_already_declared_func_ref<M: Module>(
    name: &str,
    module: &mut M,
    f: &mut ir::Function,
) -> ir::FuncRef {
    match module.get_name(name) {
        Some(FuncOrDataId::Func(f_id)) => module.declare_func_in_func(f_id, f),
        _ => panic!(),
    }
}

#[derive(Debug)]
#[allow(clippy::enum_variant_names)]
enum RTFunc {
    PrintInt,
    PrintBool,
    PrintSep,
    PrintEnd,
}

impl RTFuncUtils for RTFunc {
    fn sig(
        &self,
        _pointer_type: ir::Type,
        call_conv: cranelift_codegen::isa::CallConv,
    ) -> ir::Signature {
        match self {
            Self::PrintInt => ir::Signature {
                params: vec![ir::AbiParam::new(ir::types::I64)],
                returns: vec![],
                call_conv,
            },
            Self::PrintBool => ir::Signature {
                params: vec![ir::AbiParam::new(ir::types::B1)],
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
        }
    }

    fn name(&self) -> &'static str {
        match self {
            Self::PrintInt => "_bril_print_int",
            Self::PrintBool => "_bril_print_bool",
            Self::PrintSep => "_bril_print_sep",
            Self::PrintEnd => "_bril_print_end",
        }
    }
}

/// Runtime functions used in the native `main` function, which dispatches to the proper Bril
/// `main` function.
#[derive(Debug)]
enum RTSetupFunc {
    ParseInt,
    ParseBool,
}

impl RTFuncUtils for RTSetupFunc {
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
                returns: vec![ir::AbiParam::new(ir::types::B1)],
                call_conv,
            },
        }
    }

    fn name(&self) -> &'static str {
        match self {
            Self::ParseInt => "_bril_parse_int",
            Self::ParseBool => "_bril_parse_bool",
        }
    }
}

/// Translate a Bril type into a CLIF type.
fn translate_type(typ: &bril::Type) -> ir::Type {
    match typ {
        bril::Type::Int => ir::types::I64,
        bril::Type::Bool => ir::types::B1,
    }
}

/// Generate a CLIF signature for a Bril function.
fn translate_sig(func: &bril::Function) -> ir::Signature {
    let mut sig = ir::Signature::new(isa::CallConv::Fast);
    if let Some(ret) = &func.return_type {
        sig.returns.push(ir::AbiParam::new(translate_type(ret)));
    }
    for arg in &func.args {
        sig.params
            .push(ir::AbiParam::new(translate_type(&arg.arg_type)));
    }
    sig
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
        _ => panic!("not a translatable opcode: {}", op),
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
        _ => panic!("not a comparison opcode: {}", op),
    }
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
    module: M,
    context: cranelift_codegen::Context,
    funcs: HashMap<String, cranelift_module::FuncId>,
}

/// Configure a Cranelift target ISA object.
fn get_isa(
    target: Option<String>,
    pic: bool,
    opt_level: &str,
) -> Box<dyn cranelift_codegen::isa::TargetIsa> {
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

/// AOT compiler that generates `.o` files.
impl Translator<ObjectModule> {
    fn new(target: Option<String>, opt_level: &str) -> Self {
        // Make an object module.
        let isa = get_isa(target, true, opt_level);
        let module =
            ObjectModule::new(ObjectBuilder::new(isa, "foo", default_libcall_names()).unwrap());

        Self {
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

/// JIT compiler that totally does not work yet.
impl Translator<JITModule> {
    fn new() -> Self {
        // Cranelift JIT scaffolding.
        let builder = JITBuilder::new(cranelift_module::default_libcall_names()).unwrap();
        let module = JITModule::new(builder);

        Self {
            context: module.make_context(),
            module,
            funcs: HashMap::new(),
        }
    }

    fn compile(mut self) -> *const u8 {
        self.module.clear_context(&mut self.context);
        self.module.finalize_definitions();

        // TODO Compile all functions.
        let id = self.funcs["main"];
        self.module.get_finalized_function(id)
    }
}

/// Is a given Bril instruction a basic block terminator?
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

/// Generate a CLIF icmp instruction.
fn gen_icmp(
    builder: &mut FunctionBuilder,
    vars: &HashMap<String, Variable>,
    args: &[String],
    dest: &String,
    cc: IntCC,
) {
    let lhs = builder.use_var(vars[&args[0]]);
    let rhs = builder.use_var(vars[&args[1]]);
    let res = builder.ins().icmp(cc, lhs, rhs);
    builder.def_var(vars[dest], res);
}

/// Generate a CLIF binary operator.
fn gen_binary(
    builder: &mut FunctionBuilder,
    vars: &HashMap<String, Variable>,
    args: &[String],
    dest: &String,
    dest_type: &bril::Type,
    op: ir::Opcode,
) {
    let lhs = builder.use_var(vars[&args[0]]);
    let rhs = builder.use_var(vars[&args[1]]);
    let typ = translate_type(dest_type);
    let (inst, dfg) = builder.ins().Binary(op, typ, lhs, rhs);
    let res = dfg.first_result(inst);
    builder.def_var(vars[dest], res);
}

/// An environment for translating Bril into CLIF.
struct CompileEnv<'a, M: Module> {
    module: &'a mut M,
    vars: HashMap<String, Variable>,
    var_types: HashMap<&'a String, &'a bril::Type>,
    blocks: HashMap<String, ir::Block>,
}

impl<'a, M: Module> CompileEnv<'a, M> {
    /// Implement a Bril `print` instruction in CLIF.
    fn gen_print(&mut self, args: &[String], builder: &mut FunctionBuilder) {
        let call_conv = self.module.isa().default_call_conv();
        let mut first = true;
        for arg in args {
            // Separate printed values.
            if first {
                first = false;
            } else {
                let f = RTFunc::PrintSep.get_func_ref(&mut self.module, builder.func, call_conv);
                builder.ins().call(f, &[]);
            }

            // Print each value according to its type.
            let arg_val = builder.use_var(self.vars[arg]);
            let print_func = match self.var_types[arg] {
                bril::Type::Int => RTFunc::PrintInt,
                bril::Type::Bool => RTFunc::PrintBool,
            };
            let print_ref = print_func.get_func_ref(&mut self.module, builder.func, call_conv);
            builder.ins().call(print_ref, &[arg_val]);
        }
        let f = RTFunc::PrintEnd.get_func_ref(&mut self.module, builder.func, call_conv);
        builder.ins().call(f, &[]);
    }

    /// Compile one Bril instruction into CLIF.
    fn compile_inst(&mut self, inst: &bril::Instruction, builder: &mut FunctionBuilder) {
        match inst {
            bril::Instruction::Constant {
                dest,
                op: _,
                const_type: _,
                value,
            } => {
                let val = match value {
                    bril::Literal::Int(i) => builder.ins().iconst(ir::types::I64, *i),
                    bril::Literal::Bool(b) => builder.ins().bconst(ir::types::B1, *b),
                };
                builder.def_var(self.vars[dest], val);
            }
            bril::Instruction::Effect {
                args,
                funcs,
                labels,
                op,
            } => match op {
                bril::EffectOps::Print => self.gen_print(args, builder),
                bril::EffectOps::Jump => {
                    builder.ins().jump(self.blocks[&labels[0]], &[]);
                }
                bril::EffectOps::Branch => {
                    let arg = builder.use_var(self.vars[&args[0]]);
                    let true_block = self.blocks[&labels[0]];
                    let false_block = self.blocks[&labels[1]];
                    builder.ins().brnz(arg, true_block, &[]);
                    builder.ins().jump(false_block, &[]);
                }
                bril::EffectOps::Call => {
                    let func_ref =
                        get_already_declared_func_ref(&funcs[0], self.module, builder.func);
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
            },
            bril::Instruction::Value {
                args,
                dest,
                funcs,
                labels: _,
                op,
                op_type,
            } => match op {
                bril::ValueOps::Add
                | bril::ValueOps::Sub
                | bril::ValueOps::Mul
                | bril::ValueOps::Div
                | bril::ValueOps::And
                | bril::ValueOps::Or => {
                    gen_binary(builder, &self.vars, args, dest, op_type, translate_op(*op));
                }
                bril::ValueOps::Lt
                | bril::ValueOps::Le
                | bril::ValueOps::Eq
                | bril::ValueOps::Ge
                | bril::ValueOps::Gt => {
                    gen_icmp(builder, &self.vars, args, dest, translate_intcc(*op))
                }
                bril::ValueOps::Not => {
                    let arg = builder.use_var(self.vars[&args[0]]);
                    let res = builder.ins().bnot(arg);
                    builder.def_var(self.vars[dest], res);
                }
                bril::ValueOps::Call => {
                    let func_ref =
                        get_already_declared_func_ref(&funcs[0], self.module, builder.func);
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
            },
        }
    }
    fn compile_body(&mut self, insts: &[bril::Code], builder: &mut FunctionBuilder) {
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

                    if is_term(inst) {
                        terminated = true;
                    }
                }
                bril::Code::Label { label } => {
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

impl<M: Module> Translator<M> {
    fn declare_func(&mut self, func: &bril::Function) -> cranelift_module::FuncId {
        // The Bril `main` function gets a different internal name, and we call it from a new
        // proper main function that gets argv/argc.
        let name = if func.name == "main" {
            "__bril_main"
        } else {
            &func.name
        };

        let sig = translate_sig(func);
        self.module
            .declare_function(name, cranelift_module::Linkage::Local, &sig)
            .unwrap()
    }

    fn enter_func(&mut self, func: &bril::Function, func_id: cranelift_module::FuncId) {
        let sig = translate_sig(func);
        self.context.func =
            ir::Function::with_name_signature(ir::ExternalName::user(0, func_id.as_u32()), sig);
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

    fn compile_func(&mut self, func: bril::Function) {
        let mut fn_builder_ctx = FunctionBuilderContext::new();
        let mut builder = FunctionBuilder::new(&mut self.context.func, &mut fn_builder_ctx);

        // Declare all variables (including for function parameters).
        let var_types = all_vars(&func);
        let vars: HashMap<String, Variable> = var_types
            .iter()
            .enumerate()
            .map(|(i, (name, typ))| {
                let var = Variable::new(i);
                builder.declare_var(var, translate_type(typ));
                (name.to_string(), var)
            })
            .collect();

        // Create blocks for every label.
        let blocks: HashMap<String, ir::Block> = func
            .instrs
            .iter()
            .filter_map(|code| match code {
                bril::Code::Label { label } => {
                    let block = builder.create_block();
                    Some((label.to_string(), block))
                }
                _ => None,
            })
            .collect();

        let mut env = CompileEnv {
            module: &mut self.module,
            vars,
            var_types,
            blocks,
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

    /// Generate a proper `main` function that calls the Bril `main` function.
    fn add_main(&mut self, args: &[bril::Argument], dump: bool) {
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
            ir::Function::with_name_signature(ir::ExternalName::user(0, main_id.as_u32()), sig);

        let call_conv = self.module.isa().default_call_conv();

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
                let parse_ref = match arg.arg_type {
                    bril::Type::Int => RTSetupFunc::ParseInt,
                    bril::Type::Bool => RTSetupFunc::ParseBool,
                }
                .get_func_ref(&mut self.module, builder.func, call_conv);
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
    }

    fn compile_prog(&mut self, prog: bril::Program, dump: bool, wrap_main: bool) {
        // Declare all functions.
        for func in &prog.functions {
            let id = self.declare_func(func);
            self.funcs.insert(func.name.to_owned(), id);
        }

        // Define all functions.
        for func in prog.functions {
            // If it's main, (maybe) wrap it in an entry function.
            if wrap_main && func.name == "main" {
                self.add_main(&func.args, dump);
            }

            // Compile every function.
            let id = self.funcs[&func.name];
            self.enter_func(&func, id);
            self.compile_func(func);
            self.finish_func(id, dump);
        }
    }
}

#[derive(FromArgs)]
#[argh(description = "Bril compiler")]
struct Args {
    #[argh(switch, short = 'j', description = "JIT and run (doesn't work)")]
    jit: bool,

    #[argh(option, short = 't', description = "target triple")]
    target: Option<String>,

    #[argh(
        option,
        short = 'o',
        description = "output object file",
        default = "String::from(\"bril.o\")"
    )]
    output: String,

    #[argh(switch, short = 'd', description = "dump CLIF IR")]
    dump_ir: bool,

    #[argh(switch, short = 'v', description = "verbose logging")]
    verbose: bool,

    #[argh(
        option,
        short = 'O',
        description = "optimization level (none, speed, or speed_and_size)",
        default = "String::from(\"none\")"
    )]
    opt_level: String,
}

fn main() {
    let args: Args = argh::from_env();

    // Set up logging.
    simplelog::TermLogger::init(
        if args.verbose {
            simplelog::LevelFilter::Debug
        } else {
            simplelog::LevelFilter::Warn
        },
        simplelog::Config::default(),
        simplelog::TerminalMode::Mixed,
        simplelog::ColorChoice::Auto,
    )
    .unwrap();

    // Load the Bril program from stdin.
    let prog = bril::load_program();

    if args.jit {
        let mut trans = Translator::<JITModule>::new();
        trans.compile_prog(prog, args.dump_ir, false);
        trans.compile();
    } else {
        let mut trans = Translator::<ObjectModule>::new(args.target, &args.opt_level);
        trans.compile_prog(prog, args.dump_ir, true);
        trans.emit(&args.output);
    }
}
