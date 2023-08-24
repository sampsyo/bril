use std::collections::HashMap;

use inkwell::{
    basic_block::BasicBlock,
    builder::Builder,
    context::Context,
    module::Module,
    types::{BasicMetadataTypeEnum, BasicType, BasicTypeEnum, FunctionType, PointerType},
    values::{BasicValue, BasicValueEnum, FloatValue, FunctionValue, IntValue, PointerValue},
    AddressSpace, FloatPredicate, IntPredicate,
};

use bril_rs::{
    Argument, Code, ConstOps, EffectOps, Function, Instruction, Literal, Program, Type, ValueOps,
};

/// A helper function for performing operations over LLVM types
fn llvm_type_map<'ctx, A, F>(context: &'ctx Context, ty: &Type, mut fn_map: F) -> A
where
    F: for<'a> FnMut(BasicTypeEnum<'ctx>) -> A,
{
    match ty {
        Type::Int => fn_map(context.i64_type().into()),
        Type::Bool => fn_map(context.bool_type().into()),
        Type::Float => fn_map(context.f64_type().into()),
        Type::Pointer(_) => fn_map(build_pointertype(context, ty).into()),
    }
}

fn unwrap_bril_ptrtype(ty: &Type) -> &Type {
    match ty {
        Type::Pointer(ty) => ty,
        _ => unreachable!(),
    }
}

/// Converts a Bril Pointer type into an LLVM Pointer type
fn build_pointertype<'a>(context: &'a Context, ty: &Type) -> PointerType<'a> {
    llvm_type_map(context, unwrap_bril_ptrtype(ty), |t| {
        t.ptr_type(AddressSpace::default())
    })
}

/// Converts a Bril function signature into an LLVM function type
fn build_functiontype<'a>(
    context: &'a Context,
    args: &[&Type],
    return_ty: &Option<Type>,
) -> FunctionType<'a> {
    let param_types: Vec<BasicMetadataTypeEnum> = args
        .iter()
        .map(|t| llvm_type_map(context, t, Into::into))
        .collect();
    #[allow(clippy::option_if_let_else)] // I think this is more readable
    match return_ty {
        None => context.void_type().fn_type(&param_types, false),
        Some(t) => llvm_type_map(context, t, |t| t.fn_type(&param_types, false)),
    }
}

fn build_load<'a>(
    context: &'a Context,
    builder: &'a Builder,
    ptr: &WrappedPointer<'a>,
    name: &str,
) -> BasicValueEnum<'a> {
    llvm_type_map(context, &ptr.ty, |pointee_ty| {
        builder.build_load(pointee_ty, ptr.ptr, name)
    })
}

// Type information is needed for cases like Bool which is modelled as an int and is as far as I can tell indistinguishable.
#[derive(Debug, Clone)]
struct WrappedPointer<'a> {
    ty: Type,
    ptr: PointerValue<'a>,
}

impl<'a> WrappedPointer<'a> {
    fn new(builder: &'a Builder, context: &'a Context, name: &str, ty: &Type) -> Self {
        Self {
            ty: ty.clone(),
            ptr: llvm_type_map(context, ty, |ty| builder.build_alloca(ty, name)),
        }
    }
}

#[derive(Default)]
struct Heap<'a, 'b> {
    // Map variable names in Bril to their type and location on the stack.
    map: HashMap<&'b String, WrappedPointer<'a>>,
}

impl<'a, 'b> Heap<'a, 'b> {
    fn new() -> Self {
        Self::default()
    }

    fn add(
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

#[derive(Default)]
struct Fresh {
    count: u64,
}

impl Fresh {
    fn new() -> Self {
        Self::default()
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

// This handles the builder boilerplate of creating loads for the arguments of a function and the the corresponding store of the result.
fn build_op<'a, 'b>(
    context: &'a Context,
    builder: &'a Builder,
    heap: &mut Heap<'a, 'b>,
    fresh: &mut Fresh,
    op: impl Fn(Vec<BasicValueEnum<'a>>) -> BasicValueEnum<'a>,
    args: &'b [String],
    dest: &'b String,
) {
    builder.build_store(
        heap.get(dest).ptr,
        op(args
            .iter()
            .map(|n| build_load(context, builder, &heap.get(n), &fresh.fresh_var()))
            .collect()),
    );
}

// Like `build_op` but where there is no return value
fn build_effect_op<'a, 'b>(
    context: &'a Context,
    builder: &'a Builder,
    heap: &mut Heap<'a, 'b>,
    fresh: &mut Fresh,
    op: impl Fn(Vec<BasicValueEnum<'a>>),
    args: &'b [String],
) {
    op(args
        .iter()
        .map(|n| build_load(context, builder, &heap.get(n), &fresh.fresh_var()))
        .collect());
}

// Handles the map of labels to LLVM Basicblocks and creates a new one when it doesn't exist
fn block_map_get<'a>(
    context: &'a Context,
    llvm_func: FunctionValue<'a>,
    block_map: &mut HashMap<String, BasicBlock<'a>>,
    name: &str,
) -> BasicBlock<'a> {
    *block_map
        .entry(name.to_owned())
        .or_insert_with(|| context.append_basic_block(llvm_func, name))
}

// The workhorse of converting a Bril Instruction to an LLVM Instruction
#[allow(clippy::too_many_arguments)]
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
        // Special case where Bril casts integers to floats
        Instruction::Constant {
            dest,
            op: ConstOps::Const,
            const_type: Type::Float,
            value: Literal::Int(i),
        } => {
            #[allow(clippy::cast_precision_loss)]
            builder.build_store(
                heap.get(dest).ptr,
                context.f64_type().const_float(*i as f64),
            );
        }
        Instruction::Constant {
            dest,
            op: ConstOps::Const,
            const_type: _,
            value: Literal::Int(i),
        } => {
            #[allow(clippy::cast_sign_loss)]
            builder.build_store(
                heap.get(dest).ptr,
                context.i64_type().const_int(*i as u64, true),
            );
        }
        Instruction::Constant {
            dest,
            op: ConstOps::Const,
            const_type: _,
            value: Literal::Bool(b),
        } => {
            builder.build_store(
                heap.get(dest).ptr,
                context.bool_type().const_int((*b).into(), false),
            );
        }
        Instruction::Constant {
            dest,
            op: ConstOps::Const,
            const_type: _,
            value: Literal::Float(f),
        } => {
            builder.build_store(heap.get(dest).ptr, context.f64_type().const_float(*f));
        }
        Instruction::Value {
            args,
            dest,
            funcs: _,
            labels: _,
            op: ValueOps::Add,
            op_type: _,
        } => {
            let ret_name = fresh.fresh_var();
            build_op(
                context,
                builder,
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
                args,
                dest,
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
            build_op(
                context,
                builder,
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
                args,
                dest,
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
            build_op(
                context,
                builder,
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
                args,
                dest,
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
            build_op(
                context,
                builder,
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
                args,
                dest,
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
            build_op(
                context,
                builder,
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
                args,
                dest,
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
            build_op(
                context,
                builder,
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
                args,
                dest,
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
            build_op(
                context,
                builder,
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
                args,
                dest,
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
            build_op(
                context,
                builder,
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
                args,
                dest,
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
            build_op(
                context,
                builder,
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
                args,
                dest,
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
            build_op(
                context,
                builder,
                heap,
                fresh,
                |v| {
                    builder
                        .build_not::<IntValue>(v[0].try_into().unwrap(), &ret_name)
                        .into()
                },
                args,
                dest,
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
            build_op(
                context,
                builder,
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
                args,
                dest,
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
            build_op(
                context,
                builder,
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
                args,
                dest,
            );
        }
        Instruction::Value {
            args,
            dest,
            funcs,
            labels: _,
            op: ValueOps::Call,
            op_type: _,
        } => {
            let func_name = if funcs[0] == "main" {
                "_main"
            } else {
                &funcs[0]
            };
            let function = module.get_function(func_name).unwrap();
            let ret_name = fresh.fresh_var();
            build_op(
                context,
                builder,
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
                args,
                dest,
            );
        }
        Instruction::Value {
            args,
            dest,
            funcs: _,
            labels: _,
            op: ValueOps::Id,
            op_type: _,
        } => build_op(context, builder, heap, fresh, |v| v[0], args, dest),
        Instruction::Value {
            args,
            dest,
            funcs: _,
            labels: _,
            op: ValueOps::Fadd,
            op_type: _,
        } => {
            let ret_name = fresh.fresh_var();
            build_op(
                context,
                builder,
                heap,
                fresh,
                |v| {
                    builder
                        .build_float_add::<FloatValue>(
                            v[0].try_into().unwrap(),
                            v[1].try_into().unwrap(),
                            &ret_name,
                        )
                        .into()
                },
                args,
                dest,
            );
        }
        Instruction::Value {
            args,
            dest,
            funcs: _,
            labels: _,
            op: ValueOps::Fsub,
            op_type: _,
        } => {
            let ret_name = fresh.fresh_var();
            build_op(
                context,
                builder,
                heap,
                fresh,
                |v| {
                    builder
                        .build_float_sub::<FloatValue>(
                            v[0].try_into().unwrap(),
                            v[1].try_into().unwrap(),
                            &ret_name,
                        )
                        .into()
                },
                args,
                dest,
            );
        }
        Instruction::Value {
            args,
            dest,
            funcs: _,
            labels: _,
            op: ValueOps::Fmul,
            op_type: _,
        } => {
            let ret_name = fresh.fresh_var();
            build_op(
                context,
                builder,
                heap,
                fresh,
                |v| {
                    builder
                        .build_float_mul::<FloatValue>(
                            v[0].try_into().unwrap(),
                            v[1].try_into().unwrap(),
                            &ret_name,
                        )
                        .into()
                },
                args,
                dest,
            );
        }
        Instruction::Value {
            args,
            dest,
            funcs: _,
            labels: _,
            op: ValueOps::Fdiv,
            op_type: _,
        } => {
            let ret_name = fresh.fresh_var();
            build_op(
                context,
                builder,
                heap,
                fresh,
                |v| {
                    builder
                        .build_float_div::<FloatValue>(
                            v[0].try_into().unwrap(),
                            v[1].try_into().unwrap(),
                            &ret_name,
                        )
                        .into()
                },
                args,
                dest,
            );
        }
        Instruction::Value {
            args,
            dest,
            funcs: _,
            labels: _,
            op: ValueOps::Feq,
            op_type: _,
        } => {
            let ret_name = fresh.fresh_var();
            build_op(
                context,
                builder,
                heap,
                fresh,
                |v| {
                    builder
                        .build_float_compare::<FloatValue>(
                            FloatPredicate::OEQ,
                            v[0].try_into().unwrap(),
                            v[1].try_into().unwrap(),
                            &ret_name,
                        )
                        .into()
                },
                args,
                dest,
            );
        }
        Instruction::Value {
            args,
            dest,
            funcs: _,
            labels: _,
            op: ValueOps::Flt,
            op_type: _,
        } => {
            let ret_name = fresh.fresh_var();
            build_op(
                context,
                builder,
                heap,
                fresh,
                |v| {
                    builder
                        .build_float_compare::<FloatValue>(
                            FloatPredicate::OLT,
                            v[0].try_into().unwrap(),
                            v[1].try_into().unwrap(),
                            &ret_name,
                        )
                        .into()
                },
                args,
                dest,
            );
        }
        Instruction::Value {
            args,
            dest,
            funcs: _,
            labels: _,
            op: ValueOps::Fgt,
            op_type: _,
        } => {
            let ret_name = fresh.fresh_var();
            build_op(
                context,
                builder,
                heap,
                fresh,
                |v| {
                    builder
                        .build_float_compare::<FloatValue>(
                            FloatPredicate::OGT,
                            v[0].try_into().unwrap(),
                            v[1].try_into().unwrap(),
                            &ret_name,
                        )
                        .into()
                },
                args,
                dest,
            );
        }
        Instruction::Value {
            args,
            dest,
            funcs: _,
            labels: _,
            op: ValueOps::Fle,
            op_type: _,
        } => {
            let ret_name = fresh.fresh_var();
            build_op(
                context,
                builder,
                heap,
                fresh,
                |v| {
                    builder
                        .build_float_compare::<FloatValue>(
                            FloatPredicate::OLE,
                            v[0].try_into().unwrap(),
                            v[1].try_into().unwrap(),
                            &ret_name,
                        )
                        .into()
                },
                args,
                dest,
            );
        }
        Instruction::Value {
            args,
            dest,
            funcs: _,
            labels: _,
            op: ValueOps::Fge,
            op_type: _,
        } => {
            let ret_name = fresh.fresh_var();
            build_op(
                context,
                builder,
                heap,
                fresh,
                |v| {
                    builder
                        .build_float_compare::<FloatValue>(
                            FloatPredicate::OGE,
                            v[0].try_into().unwrap(),
                            v[1].try_into().unwrap(),
                            &ret_name,
                        )
                        .into()
                },
                args,
                dest,
            );
        }
        Instruction::Effect {
            args,
            funcs: _,
            labels: _,
            op: EffectOps::Return,
        } => {
            if args.is_empty() {
                builder.build_return(None);
            } else {
                builder.build_return(Some(&build_load(
                    context,
                    builder,
                    &heap.get(&args[0]),
                    &fresh.fresh_var(),
                )));
            }
        }
        Instruction::Effect {
            args,
            funcs,
            labels: _,
            op: EffectOps::Call,
        } => {
            let func_name = if funcs[0] == "main" {
                "_main"
            } else {
                &funcs[0]
            };
            let function = module.get_function(func_name).unwrap();
            let ret_name = fresh.fresh_var();
            build_effect_op(
                context,
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
                args,
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
            let print_float = module.get_function("_bril_print_float").unwrap();
            let print_sep = module.get_function("_bril_print_sep").unwrap();
            let print_end = module.get_function("_bril_print_end").unwrap();
            /*            let ret_name = fresh.fresh_var(); */
            let len = args.len();

            args.iter().enumerate().for_each(|(i, a)| {
                let wrapped_ptr = heap.get(a);
                let v = build_load(context, builder, &wrapped_ptr, &fresh.fresh_var());
                match wrapped_ptr.ty {
                    Type::Int => {
                        builder.build_call(print_int, &[v.into()], "print_int");
                    }
                    Type::Bool => {
                        builder.build_call(
                            print_bool,
                            &[builder
                                .build_int_cast::<IntValue>(
                                    v.try_into().unwrap(),
                                    context.bool_type(),
                                    "bool_cast",
                                )
                                .into()],
                            "print_bool",
                        );
                    }
                    Type::Float => {
                        builder.build_call(print_float, &[v.into()], "print_float");
                    }
                    Type::Pointer(_) => {
                        unreachable!()
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
            build_effect_op(
                context,
                builder,
                heap,
                fresh,
                |v| {
                    builder.build_conditional_branch(
                        v[0].try_into().unwrap(),
                        then_block,
                        else_block,
                    );
                },
                args,
            );
        }
        Instruction::Value {
            args,
            dest,
            funcs: _,
            labels,
            op: ValueOps::Phi,
            op_type,
        } => {
            let name = fresh.fresh_var();
            let blocks = labels
                .iter()
                .map(|l| block_map_get(context, llvm_func, block_map, l))
                .collect::<Vec<_>>();

            let phi = builder.build_phi(
                build_pointertype(context, &Type::Pointer(Box::new(op_type.clone()))),
                &name,
            );

            let pointers = args.iter().map(|a| heap.get(a).ptr).collect::<Vec<_>>();

            // The phi node is a little non-standard since we can't load in values from the stack before the phi instruction. Instead, the phi instruction will be over stack locations which will then be loaded into the corresponding output location.
            phi.add_incoming(
                pointers
                    .iter()
                    .zip(blocks.iter())
                    .map(|(val, block)| (val as &dyn BasicValue, *block))
                    .collect::<Vec<_>>()
                    .as_slice(),
            );

            builder.build_store(
                heap.get(dest).ptr,
                build_load(
                    context,
                    builder,
                    &WrappedPointer {
                        ty: op_type.clone(),
                        ptr: phi.as_basic_value().into_pointer_value(),
                    },
                    &fresh.fresh_var(),
                ),
            );
        }
        Instruction::Value {
            args,
            dest,
            funcs: _,
            labels: _,
            op: ValueOps::Alloc,
            op_type,
        } => {
            let alloc_name = fresh.fresh_var();
            let ty = unwrap_bril_ptrtype(op_type);
            build_op(
                context,
                builder,
                heap,
                fresh,
                |v| {
                    llvm_type_map(context, ty, |ty| {
                        builder
                            .build_array_malloc(ty, v[0].try_into().unwrap(), &alloc_name)
                            .unwrap()
                            .into()
                    })
                },
                args,
                dest,
            );
        }
        Instruction::Value {
            args,
            dest,
            funcs: _,
            labels: _,
            op: ValueOps::Load,
            op_type,
        } => {
            let name = fresh.fresh_var();
            llvm_type_map(context, op_type, |pointee_ty| {
                build_op(
                    context,
                    builder,
                    heap,
                    fresh,
                    |v| builder.build_load(pointee_ty, v[0].try_into().unwrap(), &name),
                    args,
                    dest,
                );
            });
        }
        Instruction::Value {
            args,
            dest,
            funcs: _,
            labels: _,
            op: ValueOps::PtrAdd,
            op_type,
        } => {
            let name = fresh.fresh_var();
            let op_type = unwrap_bril_ptrtype(op_type);
            build_op(
                context,
                builder,
                heap,
                fresh,
                |v| unsafe {
                    llvm_type_map(context, op_type, |pointee_ty| {
                        builder
                            .build_gep(
                                pointee_ty,
                                v[0].try_into().unwrap(),
                                &[v[1].try_into().unwrap()],
                                &name,
                            )
                            .into()
                    })
                },
                args,
                dest,
            );
        }
        Instruction::Effect {
            args,
            funcs: _,
            labels: _,
            op: EffectOps::Store,
        } => {
            build_effect_op(
                context,
                builder,
                heap,
                fresh,
                |v| {
                    builder.build_store(v[0].try_into().unwrap(), v[1]);
                },
                args,
            );
        }
        Instruction::Effect {
            args,
            funcs: _,
            labels: _,
            op: EffectOps::Free,
        } => {
            build_effect_op(
                context,
                builder,
                heap,
                fresh,
                |v| {
                    builder.build_free(v[0].try_into().unwrap());
                },
                args,
            );
        }
    }
}

// Check for instructions that end a block
const fn is_terminating_instr(i: &Option<Instruction>) -> bool {
    matches!(
        i,
        Some(Instruction::Effect {
            args: _,
            funcs: _,
            labels: _,
            op: EffectOps::Branch | EffectOps::Jump | EffectOps::Return,
        })
    )
}

/// Given a Bril program, create an LLVM module from it
/// The `runtime_module` is the module containing the runtime library
/// # Panics
/// Panics if the program is invalid
#[must_use]
pub fn create_module_from_program<'a>(
    context: &'a Context,
    Program { functions, .. }: &Program,
    runtime_module: Module<'a>,
) -> Module<'a> {
    let builder = context.create_builder();

    // "Global" counter for creating labels/temp variable names
    let mut fresh = Fresh::new();

    // Add all functions to the module, initialize all variables in the heap, and setup for the second phase
    #[allow(clippy::needless_collect)]
    let funcs: Vec<_> = functions
        .iter()
        .map(
            |Function {
                 args,
                 instrs,
                 name,
                 return_type,
             }| {
                // Setup function in module
                let ty = build_functiontype(
                    context,
                    &args
                        .iter()
                        .map(|Argument { arg_type, .. }| arg_type)
                        .collect::<Vec<_>>(),
                    return_type,
                );

                let func_name = if name == "main" { "_main" } else { name };

                let llvm_func = runtime_module.add_function(func_name, ty, None);
                args.iter().zip(llvm_func.get_param_iter()).for_each(
                    |(Argument { name, .. }, bve)| match bve {
                        inkwell::values::BasicValueEnum::IntValue(i) => i.set_name(name),
                        inkwell::values::BasicValueEnum::FloatValue(f) => f.set_name(name),
                        inkwell::values::BasicValueEnum::PointerValue(p) => p.set_name(name),
                        inkwell::values::BasicValueEnum::ArrayValue(_)
                        | inkwell::values::BasicValueEnum::StructValue(_)
                        | inkwell::values::BasicValueEnum::VectorValue(_) => unreachable!(),
                    },
                );

                // For each function, we also need to push all variables onto the stack
                let mut heap = Heap::new();
                let block = context.append_basic_block(llvm_func, &fresh.fresh_label());
                builder.position_at_end(block);

                llvm_func.get_param_iter().enumerate().for_each(|(i, arg)| {
                    let Argument { name, arg_type } = &args[i];
                    let ptr = heap.add(&builder, context, name, arg_type).ptr;
                    builder.build_store(ptr, arg);
                });

                instrs.iter().for_each(|i| match i {
                    Code::Label { .. } | Code::Instruction(Instruction::Effect { .. }) => {}
                    Code::Instruction(Instruction::Constant {
                        dest, const_type, ..
                    }) => {
                        heap.add(&builder, context, dest, const_type);
                    }
                    Code::Instruction(Instruction::Value { dest, op_type, .. }) => {
                        heap.add(&builder, context, dest, op_type);
                    }
                });

                (llvm_func, instrs, block, heap)
            },
        )
        .collect(); // Important to collect, can't be done lazily because we need all functions to be loaded in before a call instruction of a function is processed.

    // Now actually build each function
    funcs
        .into_iter()
        .for_each(|(llvm_func, instrs, mut block, mut heap)| {
            let mut last_instr = None;

            // If their are actually instructions, proceed
            if !instrs.is_empty() {
                builder.position_at_end(block);

                // Maps labels to llvm blocks for jumps
                let mut block_map = HashMap::new();
                instrs.iter().for_each(|i| match i {
                    bril_rs::Code::Label { label, .. } => {
                        let new_block = block_map_get(context, llvm_func, &mut block_map, label);

                        // Check if wee need to insert a jump since all llvm blocks must be terminated
                        if !is_terminating_instr(&last_instr) {
                            builder.build_unconditional_branch(block_map_get(
                                context,
                                llvm_func,
                                &mut block_map,
                                label,
                            ));
                        }

                        // Start a new block
                        block = new_block;
                        builder.position_at_end(block);
                        last_instr = None;
                    }
                    bril_rs::Code::Instruction(i) => {
                        // Check if we are in a basic block that has already been terminated
                        // If so, we just keep skipping unreachable instructions until we hit a new block or run out of instructions
                        if !is_terminating_instr(&last_instr) {
                            build_instruction(
                                i,
                                context,
                                &runtime_module,
                                &builder,
                                &mut heap,
                                &mut block_map,
                                llvm_func,
                                &mut fresh,
                            );
                            last_instr = Some(i.clone());
                        }
                    }
                });
            }

            // Make sure every function is terminated with a return if not already
            if !is_terminating_instr(&last_instr) {
                builder.build_return(None);
            }
        });

    // Add new main function to act as a entry point to the function.
    // Sets up arguments for a _main call
    // and always returns zero
    let entry_func_type = context.i32_type().fn_type(
        &[
            context.i32_type().into(),
            context
                .i8_type()
                .ptr_type(AddressSpace::default())
                .ptr_type(AddressSpace::default())
                .into(),
        ],
        false,
    );
    let entry_func = runtime_module.add_function("main", entry_func_type, None);
    entry_func.get_nth_param(0).unwrap().set_name("argc");
    entry_func.get_nth_param(1).unwrap().set_name("argv");

    let entry_block = context.append_basic_block(entry_func, &fresh.fresh_label());
    builder.position_at_end(entry_block);

    let mut heap = Heap::new();

    if let Some(function) = runtime_module.get_function("_main") {
        let Function { args, .. } = functions
            .iter()
            .find(|Function { name, .. }| name == "main")
            .unwrap();

        let argv = entry_func.get_nth_param(1).unwrap().into_pointer_value();

        let parse_int = runtime_module.get_function("_bril_parse_int").unwrap();
        let parse_bool = runtime_module.get_function("_bril_parse_bool").unwrap();
        let parse_float = runtime_module.get_function("_bril_parse_float").unwrap();

        function.get_param_iter().enumerate().for_each(|(i, _)| {
            let Argument { name, arg_type } = &args[i];
            let ptr = heap.add(&builder, context, name, arg_type).ptr;
            let arg_str = builder.build_load(
                context.i8_type().ptr_type(AddressSpace::default()),
                unsafe {
                    builder.build_in_bounds_gep(
                        context
                            .i8_type()
                            .ptr_type(AddressSpace::default())
                            .ptr_type(AddressSpace::default()),
                        argv,
                        &[context.i64_type().const_int((i + 1) as u64, true)],
                        "calculate offset",
                    )
                },
                "load arg",
            );
            let arg = match arg_type {
                Type::Int => builder
                    .build_call(parse_int, &[arg_str.into()], "parse_int")
                    .try_as_basic_value()
                    .unwrap_left(),
                Type::Bool => builder
                    .build_call(parse_bool, &[arg_str.into()], "parse_bool")
                    .try_as_basic_value()
                    .unwrap_left(),
                Type::Float => builder
                    .build_call(parse_float, &[arg_str.into()], "parse_float")
                    .try_as_basic_value()
                    .unwrap_left(),
                Type::Pointer(_) => unreachable!(),
            };
            builder.build_store(ptr, arg);
        });

        build_effect_op(
            context,
            &builder,
            &mut heap,
            &mut fresh,
            |v| {
                builder.build_call(
                    function,
                    v.iter()
                        .map(|val| (*val).into())
                        .collect::<Vec<_>>()
                        .as_slice(),
                    "call main",
                );
            },
            &args
                .iter()
                .map(|Argument { name, .. }| name.clone())
                .collect::<Vec<String>>(),
        );
    }
    builder.build_return(Some(&context.i32_type().const_int(0, true)));

    // Return the module
    runtime_module
}
