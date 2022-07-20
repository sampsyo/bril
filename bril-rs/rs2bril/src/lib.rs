#![warn(clippy::all, clippy::pedantic, clippy::nursery, clippy::cargo)]
#![warn(missing_docs)]
#![doc = include_str!("../README.md")]
#![allow(clippy::too_many_lines)]
#![allow(clippy::cognitive_complexity)]

#[doc(hidden)]
pub mod cli;

use bril_rs::{
    Argument, Code, ConstOps, EffectOps, Function, Instruction, Literal, Position, Program, Type,
    ValueOps,
};

use syn::punctuated::Punctuated;
use syn::{
    BinOp, Block, Expr, ExprArray, ExprAssign, ExprAssignOp, ExprBinary, ExprBlock, ExprCall,
    ExprIf, ExprIndex, ExprLit, ExprMacro, ExprParen, ExprPath, ExprReference, ExprRepeat,
    ExprReturn, ExprUnary, ExprWhile, File, FnArg, Ident, Item, ItemFn, Lit, Local, Macro, Pat,
    PatIdent, PatType, Path, PathSegment, ReturnType, Signature, Stmt, Type as SType, TypeArray,
    TypePath, TypeReference, TypeSlice, UnOp,
};

use proc_macro2::Span;

use std::collections::HashMap;

// References, Dereference, Mutability, and Visibility are all silently ignored
// Most other things are rejected as they can't be handled

struct State {
    is_pos: bool,
    temp_var_count: u64,
    ident_type_map: HashMap<String, Type>,
    func_context_map: HashMap<String, (HashMap<String, Type>, Option<Type>)>,
}

impl State {
    fn new(is_pos: bool) -> Self {
        Self {
            is_pos,
            temp_var_count: 0,
            ident_type_map: HashMap::new(),
            func_context_map: HashMap::new(),
        }
    }

    fn fresh_var(&mut self, ty: Type) -> String {
        let num = self.temp_var_count;
        self.temp_var_count += 1;
        let name = format!("tmp{num}");
        self.add_type_for_ident(name.clone(), ty);
        name
    }

    fn fresh_label(&mut self) -> String {
        let num = self.temp_var_count;
        self.temp_var_count += 1;
        let name = format!("label{num}");
        name
    }

    fn starting_new_function(&mut self, name: &String) {
        self.ident_type_map = self.func_context_map.get(name).unwrap().0.clone();
    }

    fn add_type_for_ident(&mut self, ident: String, ty: Type) {
        self.ident_type_map.insert(ident, ty);
    }

    fn get_type_for_ident(&self, ident: &String) -> Type {
        self.ident_type_map.get(ident).unwrap().clone()
    }

    fn get_ret_type_for_func(&self, func: &String) -> Option<Type> {
        self.func_context_map.get(func).unwrap().1.clone()
    }
}

fn from_span_to_position(span: Span) -> Position {
    let position = span.start();
    Position {
        col: position.column as u64,
        row: position.line as u64,
    }
}

fn from_pat_to_string(pat: Pat) -> String {
    match pat {
        Pat::Ident(PatIdent {
            attrs,
            by_ref: _,
            mutability: _,
            ident,
            subpat,
        }) => {
            assert!(
                attrs.is_empty(),
                "can't handle attributes in function arguments"
            );
            assert!(
                subpat.is_none(),
                "can't handle subpat in function arguments"
            );
            ident.to_string()
        }
        _ => panic!("can't handle non-ident pattern"),
    }
}

fn from_type_to_type(ty: SType) -> Type {
    match ty {
        SType::Array(TypeArray { elem, .. }) | SType::Slice(TypeSlice { elem, .. }) => {
            Type::Pointer(Box::new(from_type_to_type(*elem)))
        }
        SType::Reference(TypeReference { elem, .. }) => from_type_to_type(*elem),
        SType::Path(TypePath { qself: None, path }) if path.get_ident().is_some() => {
            match path.get_ident().unwrap().to_string().as_str() {
                "i64" => Type::Int,
                "f64" => Type::Float,
                "bool" => Type::Bool,
                x => panic!("can't handle type ident {x}"),
            }
        }
        _ => panic!("can't handle type {ty:?}"),
    }
}

fn from_fnarg_to_argument(f: FnArg, state: &mut State) -> Argument {
    match f {
        FnArg::Receiver(_) => panic!("can't handle self in function arguments"),
        FnArg::Typed(PatType {
            attrs,
            pat,
            colon_token: _,
            ty,
        }) => {
            assert!(
                attrs.is_empty(),
                "can't handle attributes in function arguments"
            );

            let name = from_pat_to_string(*pat);
            let arg_type = from_type_to_type(*ty);

            state.add_type_for_ident(name.clone(), arg_type.clone());

            Argument { name, arg_type }
        }
    }
}

fn from_signature_to_function(
    Signature {
        constness,
        asyncness,
        unsafety,
        abi,
        fn_token: _,
        ident,
        generics,
        paren_token: _,
        inputs,
        variadic,
        output,
    }: Signature,
    state: &mut State,
) -> Function {
    assert!(constness.is_none(), "can't handle const in Rust function");

    assert!(asyncness.is_none(), "can't handle async in Rust function");

    assert!(unsafety.is_none(), "can't handle unsafety in Rust function");

    assert!(abi.is_none(), "can't handle abi in Rust function");

    assert!(
        generics.params.is_empty(),
        "can't handle generics in Rust function"
    );

    assert!(variadic.is_none(), "can't handle variadic in Rust function");

    let return_type = match output {
        ReturnType::Default => None,
        ReturnType::Type(_, ty) => Some(from_type_to_type(*ty)),
    };

    let args: Vec<Argument> = inputs
        .into_iter()
        .map(|i| from_fnarg_to_argument(i, state))
        .collect();

    state.func_context_map.insert(
        ident.to_string(),
        (
            args.iter()
                .map(|a| (a.name.clone(), a.arg_type.clone()))
                .collect(),
            return_type.clone(),
        ),
    );

    Function {
        name: ident.to_string(),
        pos: if state.is_pos {
            Some(from_span_to_position(ident.span()))
        } else {
            None
        },
        instrs: Vec::new(),
        args,
        return_type,
    }
}

fn array_init_helper(
    vars: Vec<String>,
    mut code: Vec<Code>,
    state: &mut State,
) -> (Option<String>, Vec<Code>) {
    let op_type = Type::Pointer(Box::new(state.get_type_for_ident(&vars[0])));
    let pointer = state.fresh_var(op_type.clone());
    let size = state.fresh_var(Type::Int);
    code.push(Code::Instruction(Instruction::Constant {
        dest: size.clone(),
        op: ConstOps::Const,
        pos: None,
        const_type: Type::Int,
        value: Literal::Int(code.len() as i64),
    }));
    code.push(Code::Instruction(Instruction::Value {
        args: vec![size],
        dest: pointer.clone(),
        funcs: Vec::new(),
        labels: Vec::new(),
        op: ValueOps::Alloc,
        pos: None,
        op_type: op_type.clone(),
    }));
    vars.into_iter().enumerate().for_each(|(i, v)| {
        let idx = state.fresh_var(Type::Int);
        code.push(Code::Instruction(Instruction::Constant {
            dest: idx.clone(),
            op: ConstOps::Const,
            pos: None,
            const_type: Type::Int,
            value: Literal::Int(i as i64),
        }));
        let index_pointer = state.fresh_var(op_type.clone());
        code.push(Code::Instruction(Instruction::Value {
            args: vec![pointer.clone(), idx],
            dest: index_pointer.clone(),
            funcs: Vec::new(),
            labels: Vec::new(),
            op: ValueOps::PtrAdd,
            pos: None,
            op_type: op_type.clone(),
        }));
        code.push(Code::Instruction(Instruction::Effect {
            args: vec![index_pointer, v],
            funcs: Vec::new(),
            labels: Vec::new(),
            op: EffectOps::Store,
            pos: None,
        }));
    });
    (Some(pointer), code)
}

fn from_expr_to_bril(expr: Expr, state: &mut State) -> (Option<String>, Vec<Code>) {
    match expr {
        Expr::Array(ExprArray {
            attrs,
            bracket_token: _,
            elems,
        }) if attrs.is_empty() => {
            let (vars, vec_code): (Vec<String>, Vec<Vec<Code>>) = elems
                .into_iter()
                .map(|e| {
                    let (a, c) = from_expr_to_bril(e, state);
                    (a.unwrap(), c)
                })
                .unzip();
            let code: Vec<Code> = vec_code.into_iter().flatten().collect();
            array_init_helper(vars, code, state)
        }
        Expr::Assign(ExprAssign {
            attrs,
            left,
            eq_token,
            right,
        }) if attrs.is_empty() => {
            let (arg, mut code) = from_expr_to_bril(*right, state);
            match *left {
                Expr::Path(ExprPath {
                    attrs,
                    qself: None,
                    path,
                }) if attrs.is_empty() && path.get_ident().is_some() => {
                    let dest = path.get_ident().unwrap().to_string();
                    let op_type = state.get_type_for_ident(&dest);
                    code.push(Code::Instruction(Instruction::Value {
                        args: vec![arg.unwrap()],
                        dest,
                        funcs: Vec::new(),
                        labels: Vec::new(),
                        op: ValueOps::Id,
                        pos: if state.is_pos {
                            Some(from_span_to_position(eq_token.spans[0]))
                        } else {
                            None
                        },
                        op_type,
                    }));
                    (None, code)
                }
                Expr::Index(ExprIndex {
                    attrs,
                    expr,
                    bracket_token,
                    index,
                }) if attrs.is_empty() => {
                    let (arg1, mut code1) = from_expr_to_bril(*expr, state);
                    let (arg2, mut code2) = from_expr_to_bril(*index, state);
                    code1.append(&mut code2);
                    code1.append(&mut code);
                    let op_type =
                        Type::Pointer(Box::new(state.get_type_for_ident(&arg.clone().unwrap())));
                    let dest = state.fresh_var(op_type.clone());
                    code1.push(Code::Instruction(Instruction::Value {
                        args: vec![arg1.unwrap(), arg2.unwrap()],
                        dest: dest.clone(),
                        funcs: Vec::new(),
                        labels: Vec::new(),
                        op: ValueOps::PtrAdd,
                        pos: if state.is_pos {
                            Some(from_span_to_position(bracket_token.span))
                        } else {
                            None
                        },
                        op_type,
                    }));
                    code1.push(Code::Instruction(Instruction::Effect {
                        args: vec![dest, arg.unwrap()],
                        funcs: Vec::new(),
                        labels: Vec::new(),
                        op: EffectOps::Store,
                        pos: if state.is_pos {
                            Some(from_span_to_position(eq_token.span))
                        } else {
                            None
                        },
                    }));
                    (None, code1)
                }
                _ => panic!("can't handle left hand assignment: {left:?}"),
            }
        }
        Expr::AssignOp(ExprAssignOp {
            attrs,
            left,
            op,
            right,
        }) if attrs.is_empty() => {
            let (assign_arg, mut assign_code) = from_expr_to_bril(*left, state);
            let mut segments = Punctuated::new();
            segments.push(PathSegment::from(Ident::new(
                assign_arg.clone().unwrap().as_str(),
                Span::call_site(), // THis probably makes no sense actually. But I'm not sure where to get the span yet
            )));
            let (arg, mut code) = from_expr_to_bril(
                Expr::Binary(ExprBinary {
                    attrs: Vec::new(),
                    left: Box::new(Expr::Path(ExprPath {
                        attrs: Vec::new(),
                        qself: None,
                        path: Path {
                            leading_colon: None,
                            segments,
                        },
                    })),
                    op: match op {
                        BinOp::AddEq(x) => BinOp::Add(syn::Token![+](x.spans[0])),
                        BinOp::SubEq(x) => BinOp::Sub(syn::Token![-](x.spans[0])),
                        BinOp::MulEq(x) => BinOp::Mul(syn::Token![*](x.spans[0])),
                        BinOp::DivEq(x) => BinOp::Div(syn::Token![/](x.spans[0])),
                        _ => panic!("can't handle Assignment Op {op:?}"),
                    },
                    right,
                }),
                state,
            );
            let op_type = state.get_type_for_ident(arg.as_ref().unwrap());
            assign_code.append(&mut code);
            assign_code.push(Code::Instruction(Instruction::Value {
                args: vec![arg.unwrap()],
                dest: assign_arg.unwrap(),
                funcs: Vec::new(),
                labels: Vec::new(),
                op: ValueOps::Id,
                pos: None,
                op_type,
            }));
            (None, assign_code)
        }
        Expr::Binary(ExprBinary {
            attrs,
            left,
            op,
            right,
        }) if attrs.is_empty() => {
            let (arg1, mut code1) = from_expr_to_bril(*left, state);
            let (arg2, mut code2) = from_expr_to_bril(*right, state);
            code1.append(&mut code2);

            let (value_op, op_type, span) =
                match (op, state.get_type_for_ident(arg1.as_ref().unwrap())) {
                    (BinOp::Add(x), Type::Int) => (ValueOps::Add, Type::Int, x.spans[0]),
                    (BinOp::Add(x), Type::Float) => (ValueOps::Fadd, Type::Float, x.spans[0]),
                    (BinOp::Sub(x), Type::Int) => (ValueOps::Sub, Type::Int, x.spans[0]),
                    (BinOp::Sub(x), Type::Float) => (ValueOps::Fsub, Type::Float, x.spans[0]),
                    (BinOp::Mul(x), Type::Int) => (ValueOps::Mul, Type::Int, x.spans[0]),
                    (BinOp::Mul(x), Type::Float) => (ValueOps::Fmul, Type::Float, x.spans[0]),
                    (BinOp::Div(x), Type::Int) => (ValueOps::Div, Type::Int, x.spans[0]),
                    (BinOp::Div(x), Type::Float) => (ValueOps::Fdiv, Type::Float, x.spans[0]),
                    (BinOp::And(x), _) => (ValueOps::And, Type::Bool, x.spans[0]),
                    (BinOp::Or(x), _) => (ValueOps::Or, Type::Bool, x.spans[0]),
                    (BinOp::Eq(x), Type::Int) => (ValueOps::Eq, Type::Bool, x.spans[0]),
                    (BinOp::Eq(x), Type::Float) => (ValueOps::Feq, Type::Bool, x.spans[0]),
                    (BinOp::Lt(x), Type::Int) => (ValueOps::Lt, Type::Bool, x.spans[0]),
                    (BinOp::Lt(x), Type::Float) => (ValueOps::Flt, Type::Bool, x.spans[0]),
                    (BinOp::Le(x), Type::Int) => (ValueOps::Le, Type::Bool, x.spans[0]),
                    (BinOp::Le(x), Type::Float) => (ValueOps::Fle, Type::Bool, x.spans[0]),
                    (BinOp::Ge(x), Type::Int) => (ValueOps::Ge, Type::Bool, x.spans[0]),
                    (BinOp::Ge(x), Type::Float) => (ValueOps::Fge, Type::Bool, x.spans[0]),
                    (BinOp::Gt(x), Type::Int) => (ValueOps::Gt, Type::Bool, x.spans[0]),
                    (BinOp::Gt(x), Type::Float) => (ValueOps::Fgt, Type::Bool, x.spans[0]),
                    (_, _) => unimplemented!(),
                };

            let dest = state.fresh_var(op_type.clone());

            code1.push(Code::Instruction(Instruction::Value {
                args: vec![arg1.unwrap(), arg2.unwrap()],
                dest: dest.clone(),
                funcs: Vec::new(),
                labels: Vec::new(),
                op: value_op,
                pos: if state.is_pos {
                    Some(from_span_to_position(span))
                } else {
                    None
                },
                op_type,
            }));
            (Some(dest), code1)
        }
        Expr::Block(ExprBlock {
            attrs,
            label: None,
            block,
        }) if attrs.is_empty() => (None, from_block_to_vec_code(block, state)),
        Expr::Call(ExprCall {
            attrs,
            func,
            paren_token,
            args,
        }) if attrs.is_empty() => {
            let f = match *func {
                Expr::Path(ExprPath {
                    attrs,
                    qself: None,
                    path,
                }) if attrs.is_empty() && path.get_ident().is_some() => {
                    path.get_ident().unwrap().to_string()
                }
                _ => panic!("can't handle non-single path as function: {func:?}"),
            };
            let (vars, vec_code): (Vec<String>, Vec<Vec<Code>>) = args
                .into_iter()
                .map(|e| {
                    let (a, c) = from_expr_to_bril(e, state);
                    (a.unwrap(), c)
                })
                .unzip();
            let mut code: Vec<Code> = vec_code.into_iter().flatten().collect();
            match state.get_ret_type_for_func(&f) {
                None => {
                    code.push(Code::Instruction(Instruction::Effect {
                        args: vars,
                        funcs: vec![f],
                        labels: Vec::new(),
                        op: EffectOps::Call,
                        pos: if state.is_pos {
                            Some(from_span_to_position(paren_token.span))
                        } else {
                            None
                        },
                    }));
                    (None, code)
                }
                Some(ret) => {
                    let dest = state.fresh_var(ret.clone());
                    code.push(Code::Instruction(Instruction::Value {
                        args: vars,
                        dest: dest.clone(),
                        funcs: vec![f],
                        labels: Vec::new(),
                        op: ValueOps::Call,
                        pos: if state.is_pos {
                            Some(from_span_to_position(paren_token.span))
                        } else {
                            None
                        },
                        op_type: ret,
                    }));
                    (Some(dest), code)
                }
            }
        }
        Expr::ForLoop(_) => todo!(),
        Expr::If(ExprIf {
            attrs,
            if_token: _,
            cond,
            then_branch,
            else_branch,
        }) if attrs.is_empty() => {
            let (cond_var, mut code) = from_expr_to_bril(*cond, state);
            let then_label = state.fresh_label();
            let else_label = state.fresh_label();
            let end_label = state.fresh_label();
            code.push(Code::Instruction(Instruction::Effect {
                args: vec![cond_var.unwrap()],
                funcs: Vec::new(),
                labels: vec![then_label.clone(), else_label.clone()],
                op: EffectOps::Branch,
                pos: None,
            }));
            code.push(Code::Label {
                label: then_label,
                pos: None,
            });

            code.append(&mut from_block_to_vec_code(then_branch, state));

            code.push(Code::Instruction(Instruction::Effect {
                args: Vec::new(),
                funcs: Vec::new(),
                labels: vec![end_label.clone()],
                op: EffectOps::Jump,
                pos: None,
            }));
            code.push(Code::Label {
                label: else_label,
                pos: None,
            });

            if let Some((_, else_branch)) = else_branch {
                if let (None, mut else_code) = from_expr_to_bril(*else_branch, state) {
                    code.append(&mut else_code);
                } else {
                    panic!("panic in else branch");
                }
            }

            code.push(Code::Instruction(Instruction::Effect {
                args: Vec::new(),
                funcs: Vec::new(),
                labels: vec![end_label.clone()],
                op: EffectOps::Jump,
                pos: None,
            }));
            code.push(Code::Label {
                label: end_label,
                pos: None,
            });
            (None, code)
        }
        Expr::Index(ExprIndex {
            attrs,
            expr,
            bracket_token,
            index,
        }) if attrs.is_empty() => {
            let (arg1, mut code1) = from_expr_to_bril(*expr, state);
            let (arg2, mut code2) = from_expr_to_bril(*index, state);
            code1.append(&mut code2);
            let pointer_type = state.get_type_for_ident(&arg1.clone().unwrap());
            let load_type = match pointer_type.clone() {
                Type::Pointer(t) => *t,
                _ => panic!("can't index into non-pointer type"),
            };
            let dest = state.fresh_var(pointer_type.clone());
            code1.push(Code::Instruction(Instruction::Value {
                args: vec![arg1.unwrap(), arg2.unwrap()],
                dest: dest.clone(),
                funcs: Vec::new(),
                labels: Vec::new(),
                op: ValueOps::PtrAdd,
                pos: if state.is_pos {
                    Some(from_span_to_position(bracket_token.span))
                } else {
                    None
                },
                op_type: pointer_type,
            }));
            let load_dest = state.fresh_var(load_type.clone());
            code1.push(Code::Instruction(Instruction::Value {
                args: vec![dest],
                dest: load_dest.clone(),
                funcs: Vec::new(),
                labels: Vec::new(),
                op: ValueOps::Load,
                pos: if state.is_pos {
                    Some(from_span_to_position(bracket_token.span))
                } else {
                    None
                },
                op_type: load_type,
            }));
            (Some(load_dest), code1)
        }
        Expr::Lit(ExprLit { attrs, lit }) if attrs.is_empty() => match lit {
            Lit::Int(x) => {
                let dest = state.fresh_var(Type::Int);
                (
                    Some(dest.clone()),
                    vec![Code::Instruction(Instruction::Constant {
                        dest,
                        op: ConstOps::Const,
                        pos: if state.is_pos {
                            Some(from_span_to_position(x.span()))
                        } else {
                            None
                        },
                        const_type: Type::Int,
                        value: Literal::Int(x.base10_parse::<i64>().unwrap()),
                    })],
                )
            }
            Lit::Float(x) => {
                let dest = state.fresh_var(Type::Float);
                (
                    Some(dest.clone()),
                    vec![Code::Instruction(Instruction::Constant {
                        dest,
                        op: ConstOps::Const,
                        pos: if state.is_pos {
                            Some(from_span_to_position(x.span()))
                        } else {
                            None
                        },
                        const_type: Type::Float,
                        value: Literal::Float(x.base10_parse::<f64>().unwrap()),
                    })],
                )
            }
            Lit::Bool(x) => {
                let dest = state.fresh_var(Type::Bool);
                (
                    Some(dest.clone()),
                    vec![Code::Instruction(Instruction::Constant {
                        dest,
                        op: ConstOps::Const,
                        pos: if state.is_pos {
                            Some(from_span_to_position(x.span))
                        } else {
                            None
                        },
                        const_type: Type::Bool,
                        value: Literal::Bool(x.value()),
                    })],
                )
            }
            _ => panic!("can't handle literal: {lit:?}"),
        },
        Expr::Macro(ExprMacro {
            attrs,
            mac:
                Macro {
                    path,
                    bang_token,
                    delimiter: _,
                    tokens,
                },
        }) if attrs.is_empty()
            && path.get_ident().map(std::string::ToString::to_string)
                == Some("println".to_string()) =>
        {
            let mut t = tokens.into_iter();
            // to remove format string
            t.next();
            let args = t
                .filter_map(|i| {
                    if i.to_string() == *"," {
                        None
                    } else {
                        Some(i.to_string())
                    }
                })
                .collect();
            (
                None,
                vec![Code::Instruction(Instruction::Effect {
                    args,
                    funcs: Vec::new(),
                    labels: Vec::new(),
                    op: EffectOps::Print,
                    pos: if state.is_pos {
                        Some(from_span_to_position(bang_token.span))
                    } else {
                        None
                    },
                })],
            )
        }
        Expr::Paren(ExprParen {
            attrs,
            paren_token: _,
            expr,
        }) if attrs.is_empty() => from_expr_to_bril(*expr, state),
        Expr::Path(ExprPath {
            attrs,
            qself: None,
            path,
        }) if attrs.is_empty() && path.get_ident().is_some() => {
            (Some(path.get_ident().unwrap().to_string()), Vec::new())
        }
        Expr::Reference(ExprReference {
            attrs,
            and_token: _,
            raw: _,
            mutability: _,
            expr,
        }) if attrs.is_empty() => from_expr_to_bril(*expr, state),
        Expr::Repeat(ExprRepeat {
            attrs,
            bracket_token: _,
            expr,
            semi_token: _,
            len,
        }) if attrs.is_empty() => {
            let (var, code) = from_expr_to_bril(*expr, state);

            let array_len = match *len {
                Expr::Lit(ExprLit {
                    attrs,
                    lit: Lit::Int(i),
                }) if attrs.is_empty() => i.base10_parse::<usize>().unwrap(),
                _ => panic!("can't handle non-literal Int for repeated array length"),
            };

            let vars = std::iter::repeat(var.unwrap()).take(array_len).collect();

            array_init_helper(vars, code, state)
        }
        Expr::Return(ExprReturn {
            attrs,
            return_token,
            expr,
        }) if attrs.is_empty() => {
            let (args, mut code) = match expr {
                Some(e) => {
                    let (a, c) = from_expr_to_bril(*e, state);
                    (vec![a.unwrap()], c)
                }
                None => (Vec::new(), Vec::new()),
            };
            code.push(Code::Instruction(Instruction::Effect {
                args,
                funcs: Vec::new(),
                labels: Vec::new(),
                op: EffectOps::Return,
                pos: if state.is_pos {
                    Some(from_span_to_position(return_token.span))
                } else {
                    None
                },
            }));
            (None, code)
        }
        Expr::Unary(ExprUnary { attrs, op, expr }) if attrs.is_empty() => {
            let (arg, mut code) = from_expr_to_bril(*expr, state);

            let mut args = vec![arg.clone().unwrap()];

            let (op, pos, op_type) = match op {
                UnOp::Deref(x) => (
                    ValueOps::Id,
                    if state.is_pos {
                        Some(from_span_to_position(x.spans[0]))
                    } else {
                        None
                    },
                    state.get_type_for_ident(&arg.unwrap()),
                ),
                UnOp::Not(x) => (
                    ValueOps::Not,
                    if state.is_pos {
                        Some(from_span_to_position(x.spans[0]))
                    } else {
                        None
                    },
                    Type::Bool,
                ),
                UnOp::Neg(x) => {
                    let ty = state.get_type_for_ident(&arg.unwrap());
                    (
                        match ty {
                            Type::Int => {
                                let tmp = state.fresh_var(ty.clone());
                                code.push(Code::Instruction(Instruction::Constant {
                                    op: ConstOps::Const,
                                    dest: tmp.clone(),
                                    const_type: ty.clone(),
                                    value: Literal::Int(-1),
                                    pos: None,
                                }));
                                args.push(tmp);
                                ValueOps::Mul
                            }
                            Type::Float => {
                                let tmp = state.fresh_var(ty.clone());
                                code.push(Code::Instruction(Instruction::Constant {
                                    op: ConstOps::Const,
                                    dest: tmp.clone(),
                                    const_type: ty.clone(),
                                    value: Literal::Float(-1.0),
                                    pos: None,
                                }));
                                args.push(tmp);
                                ValueOps::Fmul
                            }
                            _ => panic!("can't handle negation of non-int/float"),
                        },
                        if state.is_pos {
                            Some(from_span_to_position(x.spans[0]))
                        } else {
                            None
                        },
                        ty,
                    )
                }
            };

            let dest = state.fresh_var(op_type.clone());

            code.push(Code::Instruction(Instruction::Value {
                dest: dest.clone(),
                args,
                funcs: Vec::new(),
                labels: Vec::new(),
                op,
                pos,
                op_type,
            }));
            (Some(dest), code)
        }
        Expr::While(ExprWhile {
            attrs,
            label: None,
            while_token: _,
            cond,
            body,
        }) if attrs.is_empty() => {
            let start_label = state.fresh_label();
            let then_label = state.fresh_label();
            let end_label = state.fresh_label();
            let (cond_var, mut cond_code) = from_expr_to_bril(*cond, state);
            let mut code = vec![Code::Label {
                label: start_label.clone(),
                pos: None,
            }];
            code.append(&mut cond_code);
            code.push(Code::Instruction(Instruction::Effect {
                args: vec![cond_var.unwrap()],
                funcs: Vec::new(),
                labels: vec![then_label.clone(), end_label.clone()],
                op: EffectOps::Branch,
                pos: None,
            }));
            code.push(Code::Label {
                label: then_label,
                pos: None,
            });

            code.append(&mut from_block_to_vec_code(body, state));

            code.push(Code::Instruction(Instruction::Effect {
                args: Vec::new(),
                funcs: Vec::new(),
                labels: vec![start_label],
                op: EffectOps::Jump,
                pos: None,
            }));

            code.push(Code::Label {
                label: end_label,
                pos: None,
            });
            (None, code)
        }
        e => panic!("can't handle expression: {e:?}"),
    }
}

fn from_stmt_to_vec_code(s: Stmt, state: &mut State) -> Vec<Code> {
    match s {
        Stmt::Item(_) => panic!("can't handle item in function body"),
        Stmt::Local(Local {
            attrs,
            let_token,
            pat,
            init,
            semi_token: _,
        }) => {
            assert!(attrs.is_empty(), "can't handle attributes in function body");
            assert!(init.is_some(), "must initialize all assignments");
            match pat {
                Pat::Type(PatType {
                    attrs,
                    pat,
                    colon_token: _,
                    ty,
                }) if attrs.is_empty() => {
                    let op_type = from_type_to_type(*ty);
                    let dest = from_pat_to_string(*pat);
                    state.add_type_for_ident(dest.clone(), op_type.clone());
                    let (_, expr) = init.unwrap();
                    let (arg, mut code) = from_expr_to_bril(*expr, state);
                    code.push(Code::Instruction(Instruction::Value {
                        args: vec![arg.unwrap()],
                        dest,
                        funcs: Vec::new(),
                        labels: Vec::new(),
                        op: ValueOps::Id,
                        pos: if state.is_pos {
                            Some(from_span_to_position(let_token.span))
                        } else {
                            None
                        },
                        op_type,
                    }));
                    code
                }
                // todo would be nice to infer the types of variables
                p @ Pat::Ident(_) => {
                    panic!("You probably forgot to add the type when declaring a variable: {p:?}")
                }
                p => panic!("can't handle pattern in let: {p:?}"),
            }
        }
        Stmt::Expr(e) | Stmt::Semi(e, _) => {
            let (_, code) = from_expr_to_bril(e, state);
            code
        }
    }
}

fn from_block_to_vec_code(Block { stmts, .. }: Block, state: &mut State) -> Vec<Code> {
    stmts
        .into_iter()
        .flat_map(|s| from_stmt_to_vec_code(s, state))
        .collect()
}

fn from_item_fn_to_empty_function(
    ItemFn {
        attrs,
        vis: _,
        sig,
        block,
    }: ItemFn,
    state: &mut State,
) -> (Function, Block) {
    assert!(attrs.is_empty(), "can't handle attributes in Rust function");

    let func = from_signature_to_function(sig, state);
    (func, *block)
}

fn from_empty_function_to_function(
    (mut func, block): (Function, Block),
    state: &mut State,
) -> Function {
    func.instrs = from_block_to_vec_code(block, state);
    func
}

#[doc(hidden)]
#[must_use]
pub fn from_file_to_program(
    File {
        shebang,
        attrs,
        items,
    }: File,
    is_pos: bool,
) -> Program {
    assert!(shebang.is_none(), "can't handle shebang items in Rust file");

    assert!(attrs.is_empty(), "can't handle attributes in Rust file");

    let mut state = State::new(is_pos);

    // The processing of Functions is separated into two parts to get global information like type signatures for functions before processing function bodies
    let sigs_processed: Vec<(Function, Block)> = items
        .into_iter()
        .map(|i| match i {
            Item::Fn(i) => from_item_fn_to_empty_function(i, &mut state),
            _ => panic!("can't handle non-fun item"),
        })
        .collect();

    //todo passing function_arguments is probably fucked at the moment
    // Use starting_new_function or something???
    // Pass function name to start context?

    Program {
        functions: sigs_processed
            .into_iter()
            .map(|f| {
                state.starting_new_function(&f.0.name);
                from_empty_function_to_function(f, &mut state)
            })
            .collect(),
    }
}
