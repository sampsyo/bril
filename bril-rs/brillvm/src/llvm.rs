use std::collections::HashMap;

use inkwell::{
    basic_block::BasicBlock,
    builder::Builder,
    context::Context,
    module::Module,
    types::{BasicMetadataTypeEnum, FunctionType},
    values::{BasicValueEnum, FunctionValue, IntValue, PointerValue},
    IntPredicate,
};

use bril_rs::{
    Argument, ConstOps, EffectOps, Function, Instruction, Literal, Program, Type, ValueOps,
};

fn build_functiontype<'a>(
    context: &'a Context,
    args: &Vec<&Type>,
    return_ty: &Option<Type>,
) -> FunctionType<'a> {
    let param_types: Vec<BasicMetadataTypeEnum> = args
        .iter()
        .map(|t| match t {
            Type::Int => context.i64_type().into(),
            Type::Bool => context.bool_type().into(),
            /* Type::Float => context.f64_type().into(), */
        })
        .collect();
    match return_ty {
        None => context.void_type().fn_type(&param_types, false),
        Some(Type::Int) => context.i64_type().fn_type(&param_types, false),
        Some(Type::Bool) => context.bool_type().fn_type(&param_types, false),
        /* Some(Type::Float) => context.f64_type().fn_type(&param_types, false), */
    }
}

// Type information is needed for cases like Bool which is modelled as an int and is as far as I can tell indistinguishable.
#[derive(Debug, Clone)]
struct WrappedPointer<'a> {
    ty: Type,
    ptr: PointerValue<'a>,
}

impl<'a> WrappedPointer<'a> {
    fn new(builder: &'a Builder, context: &'a Context, name: &String, ty: &Type) -> Self {
        Self {
            ty: ty.clone(),
            ptr: match ty {
                Type::Int => builder.build_alloca(context.i64_type(), name),
                Type::Bool => builder.build_alloca(context.bool_type(), name),
            },
        }
    }
}

struct Heap<'a, 'b> {
    // Map variable names in Bril to their type and location on the stack.
    map: HashMap<&'b String, WrappedPointer<'a>>,
}

impl<'a, 'b> Heap<'a, 'b> {
    fn new() -> Self {
        Heap {
            map: HashMap::new(),
        }
    }

    fn get_or_create(
        &mut self,
        builder: &'a Builder,
        context: &'a Context,
        name: &'b String,
        ty: &Type,
    ) -> WrappedPointer<'a> {
        self.map
            .entry(name)
            .or_insert_with(|| WrappedPointer::new(builder, context, name, ty))
            .clone()
    }

    fn get(&self, name: &String) -> WrappedPointer<'a> {
        self.map.get(name).unwrap().clone()
    }
}

impl<'a, 'b> Default for Heap<'a, 'b> {
    fn default() -> Self {
        Self::new()
    }
}

struct Fresh {
    count: u64,
}

impl Fresh {
    fn new() -> Self {
        Self { count: 0 }
    }

    fn fresh_label(&mut self) -> String {
        let l = format!("label{}", self.count);
        self.count += 1;
        l
    }

    fn fresh_var(&mut self) -> String {
        let v = format!("var{}", self.count);
        self.count += 1;
        v
    }
}

impl Default for Fresh {
    fn default() -> Self {
        Self::new()
    }
}

// For when you know the types of variables(And thus can create them correctly if there is an issue)
fn build_load_op_store<'a, 'b>(
    builder: &'a Builder,
    context: &'a Context,
    heap: &mut Heap<'a, 'b>,
    fresh: &mut Fresh,
    op: impl Fn(Vec<BasicValueEnum<'a>>) -> BasicValueEnum<'a>,
    mut names: Vec<&'b String>,
    mut types: Vec<Type>,
) {
    builder.build_store(
        heap.get_or_create(
            builder,
            context,
            names.pop().unwrap(),
            &types.pop().unwrap(),
        )
        .ptr,
        op(names
            .iter()
            .zip(types.iter())
            .map(|(n, t)| {
                builder.build_load(
                    heap.get_or_create(builder, context, n, t).ptr,
                    &fresh.fresh_var(),
                )
            })
            .collect()),
    );
}

// When you don't statically know all of the types and you don't want to ask questions
// This is much more falliable than `build_load_op_store` so use that when you can
fn build_load_op_store_flexible<'a, 'b>(
    builder: &'a Builder,
    context: &'a Context,
    heap: &mut Heap<'a, 'b>,
    fresh: &mut Fresh,
    op: impl Fn(Vec<BasicValueEnum<'a>>) -> BasicValueEnum<'a>,
    mut names: Vec<&'b String>,
    ty: &Type,
) {
    builder.build_store(
        heap.get_or_create(builder, context, names.pop().unwrap(), ty)
            .ptr,
        op(names
            .iter()
            .map(|n| builder.build_load(heap.get(n).ptr, &fresh.fresh_var()))
            .collect()),
    );
}

// For when you know the types of variables(And thus can create them correctly if there is an issue)
fn build_load_op_effect<'a, 'b>(
    builder: &'a Builder,
    context: &'a Context,
    heap: &mut Heap<'a, 'b>,
    fresh: &mut Fresh,
    op: impl Fn(Vec<BasicValueEnum<'a>>),
    names: Vec<&'b String>,
    types: Vec<Type>,
) {
    op(names
        .iter()
        .zip(types.iter())
        .map(|(n, t)| {
            builder.build_load(
                heap.get_or_create(builder, context, n, t).ptr,
                &fresh.fresh_var(),
            )
        })
        .collect());
}

// When you don't statically know all of the types and you don't want to ask questions
// This is much more falliable than `build_load_op_effect` so use that when you can
fn build_load_op_effect_flexible<'a, 'b>(
    builder: &'a Builder,
    heap: &mut Heap<'a, 'b>,
    fresh: &mut Fresh,
    op: impl Fn(Vec<BasicValueEnum<'a>>),
    names: Vec<&'b String>,
) {
    op(names
        .iter()
        .map(|n| builder.build_load(heap.get(n).ptr, &fresh.fresh_var()))
        .collect());
}

fn block_map_get<'a, 'b>(
    context: &'a Context,
    llvm_func: FunctionValue<'a>,
    block_map: &mut HashMap<String, BasicBlock<'a>>,
    name: &String,
) -> BasicBlock<'a> {
    block_map
        .entry(name.clone())
        .or_insert_with(|| context.append_basic_block(llvm_func, name))
        .clone()
}

fn build_instruction<'a, 'b>(
    i: &'b Instruction,
    context: &'a Context,
    module: &'a Module,
    builder: &'a Builder,
    heap: &mut Heap<'a, 'b>,
    block_map: &mut HashMap<String, BasicBlock<'a>>,
    llvm_func: FunctionValue<'a>,
    fresh: &mut Fresh,
) {
    match i {
        Instruction::Constant {
            dest,
            op: ConstOps::Const,
            const_type: Type::Int,
            value: Literal::Int(i),
        } => {
            builder.build_store(
                heap.get_or_create(builder, context, dest, &Type::Int).ptr,
                context.i64_type().const_int(*i as u64, true),
            );
        }
        Instruction::Constant {
            dest,
            op: ConstOps::Const,
            const_type: Type::Bool,
            value: Literal::Bool(b),
        } => {
            builder.build_store(
                heap.get_or_create(builder, context, dest, &Type::Bool).ptr,
                context.bool_type().const_int(*b as u64, false),
            );
        }
        Instruction::Constant {
            dest: _,
            op: _,
            const_type: _,
            value: _,
        } => unimplemented!(),
        Instruction::Value {
            args,
            dest,
            funcs: _,
            labels: _,
            op: ValueOps::Add,
            op_type: _,
        } => {
            let ret_name = fresh.fresh_var();
            build_load_op_store(
                builder,
                context,
                heap,
                fresh,
                |v| {
                    builder
                        .build_int_add::<IntValue>(
                            v[0].try_into().unwrap(),
                            v[1].try_into().unwrap(),
                            &ret_name,
                        )
                        .into()
                },
                vec![&args[0], &args[1], dest],
                vec![Type::Int, Type::Int, Type::Int],
            );
        }
        Instruction::Value {
            args,
            dest,
            funcs: _,
            labels: _,
            op: ValueOps::Sub,
            op_type: _,
        } => {
            let ret_name = fresh.fresh_var();
            build_load_op_store(
                builder,
                context,
                heap,
                fresh,
                |v| {
                    builder
                        .build_int_sub::<IntValue>(
                            v[0].try_into().unwrap(),
                            v[1].try_into().unwrap(),
                            &ret_name,
                        )
                        .into()
                },
                vec![&args[0], &args[1], dest],
                vec![Type::Int, Type::Int, Type::Int],
            );
        }
        Instruction::Value {
            args,
            dest,
            funcs: _,
            labels: _,
            op: ValueOps::Mul,
            op_type: _,
        } => {
            let ret_name = fresh.fresh_var();
            build_load_op_store(
                builder,
                context,
                heap,
                fresh,
                |v| {
                    builder
                        .build_int_mul::<IntValue>(
                            v[0].try_into().unwrap(),
                            v[1].try_into().unwrap(),
                            &ret_name,
                        )
                        .into()
                },
                vec![&args[0], &args[1], dest],
                vec![Type::Int, Type::Int, Type::Int],
            );
        }
        Instruction::Value {
            args,
            dest,
            funcs: _,
            labels: _,
            op: ValueOps::Div,
            op_type: _,
        } => {
            let ret_name = fresh.fresh_var();
            build_load_op_store(
                builder,
                context,
                heap,
                fresh,
                |v| {
                    builder
                        .build_int_signed_div::<IntValue>(
                            v[0].try_into().unwrap(),
                            v[1].try_into().unwrap(),
                            &ret_name,
                        )
                        .into()
                },
                vec![&args[0], &args[1], dest],
                vec![Type::Int, Type::Int, Type::Int],
            );
        }
        Instruction::Value {
            args,
            dest,
            funcs: _,
            labels: _,
            op: ValueOps::Eq,
            op_type: _,
        } => {
            let ret_name = fresh.fresh_var();
            build_load_op_store(
                builder,
                context,
                heap,
                fresh,
                |v| {
                    builder
                        .build_int_compare::<IntValue>(
                            IntPredicate::EQ,
                            v[0].try_into().unwrap(),
                            v[1].try_into().unwrap(),
                            &ret_name,
                        )
                        .into()
                },
                vec![&args[0], &args[1], dest],
                vec![Type::Int, Type::Int, Type::Bool],
            );
        }
        Instruction::Value {
            args,
            dest,
            funcs: _,
            labels: _,
            op: ValueOps::Lt,
            op_type: _,
        } => {
            let ret_name = fresh.fresh_var();
            build_load_op_store(
                builder,
                context,
                heap,
                fresh,
                |v| {
                    builder
                        .build_int_compare::<IntValue>(
                            IntPredicate::SLT,
                            v[0].try_into().unwrap(),
                            v[1].try_into().unwrap(),
                            &ret_name,
                        )
                        .into()
                },
                vec![&args[0], &args[1], dest],
                vec![Type::Int, Type::Int, Type::Bool],
            );
        }
        Instruction::Value {
            args,
            dest,
            funcs: _,
            labels: _,
            op: ValueOps::Gt,
            op_type: _,
        } => {
            let ret_name = fresh.fresh_var();
            build_load_op_store(
                builder,
                context,
                heap,
                fresh,
                |v| {
                    builder
                        .build_int_compare::<IntValue>(
                            IntPredicate::SGT,
                            v[0].try_into().unwrap(),
                            v[1].try_into().unwrap(),
                            &ret_name,
                        )
                        .into()
                },
                vec![&args[0], &args[1], dest],
                vec![Type::Int, Type::Int, Type::Bool],
            );
        }
        Instruction::Value {
            args,
            dest,
            funcs: _,
            labels: _,
            op: ValueOps::Le,
            op_type: _,
        } => {
            let ret_name = fresh.fresh_var();
            build_load_op_store(
                builder,
                context,
                heap,
                fresh,
                |v| {
                    builder
                        .build_int_compare::<IntValue>(
                            IntPredicate::SLE,
                            v[0].try_into().unwrap(),
                            v[1].try_into().unwrap(),
                            &ret_name,
                        )
                        .into()
                },
                vec![&args[0], &args[1], dest],
                vec![Type::Int, Type::Int, Type::Bool],
            );
        }
        Instruction::Value {
            args,
            dest,
            funcs: _,
            labels: _,
            op: ValueOps::Ge,
            op_type: _,
        } => {
            let ret_name = fresh.fresh_var();
            build_load_op_store(
                builder,
                context,
                heap,
                fresh,
                |v| {
                    builder
                        .build_int_compare::<IntValue>(
                            IntPredicate::SGE,
                            v[0].try_into().unwrap(),
                            v[1].try_into().unwrap(),
                            &ret_name,
                        )
                        .into()
                },
                vec![&args[0], &args[1], dest],
                vec![Type::Int, Type::Int, Type::Bool],
            );
        }
        Instruction::Value {
            args,
            dest,
            funcs: _,
            labels: _,
            op: ValueOps::Not,
            op_type: _,
        } => {
            let ret_name = fresh.fresh_var();
            build_load_op_store(
                builder,
                context,
                heap,
                fresh,
                |v| {
                    builder
                        .build_not::<IntValue>(v[0].try_into().unwrap(), &ret_name)
                        .into()
                },
                vec![&args[0], dest],
                vec![Type::Bool, Type::Bool],
            );
        }
        Instruction::Value {
            args,
            dest,
            funcs: _,
            labels: _,
            op: ValueOps::And,
            op_type: _,
        } => {
            let ret_name = fresh.fresh_var();
            build_load_op_store(
                builder,
                context,
                heap,
                fresh,
                |v| {
                    builder
                        .build_and::<IntValue>(
                            v[0].try_into().unwrap(),
                            v[1].try_into().unwrap(),
                            &ret_name,
                        )
                        .into()
                },
                vec![&args[0], &args[1], dest],
                vec![Type::Bool, Type::Bool, Type::Bool],
            );
        }
        Instruction::Value {
            args,
            dest,
            funcs: _,
            labels: _,
            op: ValueOps::Or,
            op_type: _,
        } => {
            let ret_name = fresh.fresh_var();
            build_load_op_store(
                builder,
                context,
                heap,
                fresh,
                |v| {
                    builder
                        .build_or::<IntValue>(
                            v[0].try_into().unwrap(),
                            v[1].try_into().unwrap(),
                            &ret_name,
                        )
                        .into()
                },
                vec![&args[0], &args[1], dest],
                vec![Type::Bool, Type::Bool, Type::Bool],
            );
        }
        Instruction::Value {
            args,
            dest,
            funcs,
            labels: _,
            op: ValueOps::Call,
            op_type,
        } => {
            let function = module.get_function(&funcs[0]).unwrap();
            let ret_name = fresh.fresh_var();
            build_load_op_store_flexible(
                builder,
                context,
                heap,
                fresh,
                |v| {
                    builder
                        .build_call(
                            function,
                            v.iter()
                                .map(|val| (*val).into())
                                .collect::<Vec<_>>()
                                .as_slice(),
                            &ret_name,
                        )
                        .try_as_basic_value()
                        .left()
                        .unwrap()
                },
                args.iter().chain([dest]).collect(),
                op_type,
            );
        }
        Instruction::Value {
            args,
            dest,
            funcs: _,
            labels: _,
            op: ValueOps::Id,
            op_type,
        } => build_load_op_store(
            builder,
            context,
            heap,
            fresh,
            |v| v[0],
            args.iter().chain([dest]).collect(),
            vec![op_type.clone(), op_type.clone()],
        ),
        Instruction::Effect {
            args,
            funcs: _,
            labels: _,
            op: EffectOps::Return,
        } => {
            if llvm_func.get_name().to_str().unwrap() == "main" {
                builder.build_return(Some(&context.i64_type().const_int(0 as u64, true)));
            } else if args.is_empty() {
                builder.build_return(None);
            } else {
                builder.build_return(Some(
                    &builder.build_load(heap.get(&args[0]).ptr, &fresh.fresh_var()),
                ));
            }
        }
        Instruction::Effect {
            args,
            funcs,
            labels: _,
            op: EffectOps::Call,
        } => {
            let function = module.get_function(&funcs[0]).unwrap();
            let ret_name = fresh.fresh_var();
            build_load_op_effect_flexible(
                builder,
                heap,
                fresh,
                |v| {
                    builder.build_call(
                        function,
                        v.iter()
                            .map(|val| (*val).into())
                            .collect::<Vec<_>>()
                            .as_slice(),
                        &ret_name,
                    );
                },
                args.iter().collect(),
            );
        }
        Instruction::Effect {
            args: _,
            funcs: _,
            labels: _,
            op: EffectOps::Nop,
        } => {}
        Instruction::Effect {
            args,
            funcs: _,
            labels: _,
            op: EffectOps::Print,
        } => {
            let print_int = module.get_function("_bril_print_int").unwrap();
            let print_bool = module.get_function("_bril_print_bool").unwrap();
            /*             let print_float = module.get_function("_bril_print_float").unwrap(); */
            let print_sep = module.get_function("_bril_print_sep").unwrap();
            let print_end = module.get_function("_bril_print_end").unwrap();
            /*            let ret_name = fresh.fresh_var(); */
            let len = args.len();

            args.iter().enumerate().for_each(|(i, a)| {
                let wrapped_ptr = heap.get(a);
                let v = builder.build_load(wrapped_ptr.ptr, &fresh.fresh_var());
                match wrapped_ptr.ty {
                    Type::Int => {
                        builder.build_call(print_int, &[v.into()], "print_int");
                    }
                    Type::Bool => {
                        builder.build_call(print_bool, &[v.into()], "print_bool");
                    }
                };
                if i < len - 1 {
                    builder.build_call(print_sep, &[], "print_sep");
                }
            });
            builder.build_call(print_end, &[], "print_end");
        }
        Instruction::Effect {
            args: _,
            funcs: _,
            labels,
            op: EffectOps::Jump,
        } => {
            builder.build_unconditional_branch(block_map_get(
                context, llvm_func, block_map, &labels[0],
            ));
        }
        Instruction::Effect {
            args,
            funcs: _,
            labels,
            op: EffectOps::Branch,
        } => {
            let then_block = block_map_get(context, llvm_func, block_map, &labels[0]);
            let else_block = block_map_get(context, llvm_func, block_map, &labels[1]);
            build_load_op_effect(
                builder,
                context,
                heap,
                fresh,
                |v| {
                    builder.build_conditional_branch(
                        v[0].try_into().unwrap(),
                        then_block,
                        else_block,
                    );
                },
                args.iter().collect(),
                vec![Type::Bool],
            )
        }
    }
}

fn is_terminating_instr(i: &Option<Instruction>) -> bool {
    match i {
        Some(Instruction::Effect {
            args: _,
            funcs: _,
            labels: _,
            op: EffectOps::Branch | EffectOps::Jump | EffectOps::Return,
        }) => true,
        _ => false,
    }
}

pub fn create_module_from_program<'a>(
    context: &'a Context,
    Program { functions, .. }: &Program,
    _module_name: &str,
    runtime_path: &str,
) -> Module<'a> {
    let module = Module::parse_bitcode_from_path(runtime_path, context).unwrap();

    /* let module = context.create_module(module_name); */
    let builder = context.create_builder();
    let mut fresh = Fresh::new();
    // Add all functions to the module and save the pieces for the next part
    let funcs: Vec<_> = functions
        .iter()
        .map(
            |Function {
                 args,
                 instrs,
                 name,
                 return_type,
             }| {
                let ty = build_functiontype(
                    context,
                    &args
                        .iter()
                        .map(|Argument { arg_type, .. }| arg_type)
                        .collect(),
                    if name == "main" {
                        &Some(Type::Int)
                    } else {
                        return_type
                    },
                );
                let llvm_func = module.add_function(name, ty, None);
                args.iter().zip(llvm_func.get_param_iter()).for_each(
                    |(Argument { name, .. }, bve)| match bve {
                        inkwell::values::BasicValueEnum::IntValue(i) => i.set_name(name),
                        inkwell::values::BasicValueEnum::FloatValue(f) => f.set_name(name),
                        inkwell::values::BasicValueEnum::ArrayValue(_)
                        | inkwell::values::BasicValueEnum::PointerValue(_)
                        | inkwell::values::BasicValueEnum::StructValue(_)
                        | inkwell::values::BasicValueEnum::VectorValue(_) => unimplemented!(),
                    },
                );
                (llvm_func, args, instrs)
            },
        )
        .collect(); // Important to collect, need to add all functions first

    // Now actually build each function
    funcs.into_iter().for_each(|(llvm_func, args, instrs)| {
        if !instrs.is_empty() {
            let mut heap = Heap::new();
            let mut block_map = HashMap::new();

            let mut block = Some(context.append_basic_block(llvm_func, &fresh.fresh_label()));

            builder.position_at_end(block.unwrap());

            llvm_func.get_param_iter().enumerate().for_each(|(i, arg)| {
                let Argument { name, arg_type } = &args[i];
                let ptr = heap.get_or_create(&builder, context, name, arg_type).ptr;
                builder.build_store(ptr, arg);
            });

            let mut instrs_iter = instrs.iter();
            let mut last_instr = None;
            while let Some(i) = instrs_iter.next() {
                match i {
                    bril_rs::Code::Label { label, .. } => {
                        let new_block = block_map_get(context, llvm_func, &mut block_map, label);
                        if block != None && !is_terminating_instr(&last_instr) {
                            builder.build_unconditional_branch(block_map_get(
                                context,
                                llvm_func,
                                &mut block_map,
                                label,
                            ));
                        }
                        block = Some(new_block);
                        last_instr = None;
                    }
                    bril_rs::Code::Instruction(i) => {
                        if !is_terminating_instr(&last_instr) {
                            if let None = block {
                                block = Some(block_map_get(
                                    context,
                                    llvm_func,
                                    &mut block_map,
                                    &fresh.fresh_var(),
                                ));
                            }
                            builder.position_at_end(block.unwrap());
                            build_instruction(
                                i,
                                context,
                                &module,
                                &builder,
                                &mut heap,
                                &mut block_map,
                                llvm_func,
                                &mut fresh,
                            );
                            last_instr = Some(i.clone());
                        }
                    }
                }
            }

            // Make sure every function is terminated with a return if not already
            if !is_terminating_instr(&last_instr) {
                builder.build_return(Some(&context.i64_type().const_int(0 as u64, true)));
            }
        }
    });
    module
}
