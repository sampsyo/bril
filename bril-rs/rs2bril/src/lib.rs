#![warn(clippy::all, clippy::pedantic, clippy::nursery, clippy::cargo)]
#![warn(missing_docs)]
#![doc = include_str!("../README.md")]
#![allow(clippy::too_many_lines)]
#![allow(clippy::cognitive_complexity)]

#[doc(hidden)]
pub mod cli;

use bril_rs::{
    Argument, Code, ColRow, ConstOps, EffectOps, Function, Instruction, Literal, Position, Program,
    Type, ValueOps,
};

use syn::{
    BinOp, Block, Expr, ExprArray, ExprAssign, ExprBinary, ExprBlock, ExprCall, ExprCast, ExprIf,
    ExprIndex, ExprLet, ExprLit, ExprLoop, ExprMacro, ExprParen, ExprPath, ExprReference,
    ExprRepeat, ExprReturn, ExprUnary, ExprWhile, File, FnArg, Item, ItemFn, Lit, Local, Macro,
    Pat, PatIdent, PatType, Path, ReturnType, Signature, Stmt, StmtMacro, Type as SType, TypeArray,
    TypePath, TypeReference, TypeSlice, UnOp,
};

use proc_macro2::Span;

use std::collections::HashMap;

// References, Dereference, Mutability, and Visibility are all silently ignored
// Most other things are rejected as they can't be handled

struct State {
    is_pos: bool,
    src: Option<String>,
    temp_var_count: u64,
    ident_type_map: HashMap<String, Type>,
    func_context_map: HashMap<String, (HashMap<String, Type>, Option<Type>)>,
}

impl State {
    fn new(is_pos: bool, src: Option<String>) -> Self {
        Self {
            is_pos,
            src,
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

// A helper to get the span of any given Rust expression
fn from_expr_to_span(expr: &Expr) -> Span {
    match expr {
        Expr::Array(ExprArray {
            attrs: _,
            bracket_token,
            elems: _,
        })
        | Expr::Repeat(ExprRepeat {
            attrs: _,
            bracket_token,
            expr: _,
            semi_token: _,
            len: _,
        }) => bracket_token.span.join(),
        Expr::Assign(ExprAssign {
            attrs: _,
            left,
            eq_token: _,
            right,
        })
        | Expr::Binary(ExprBinary {
            attrs: _,
            left,
            op: _,
            right,
        }) => from_expr_to_span(left)
            .join(from_expr_to_span(right))
            .unwrap(),
        Expr::Block(ExprBlock {
            attrs: _,
            label: _,
            block,
        }) => block.brace_token.span.join(),
        Expr::Call(ExprCall {
            attrs: _,
            func,
            paren_token,
            args: _,
        }) => from_expr_to_span(func)
            .join(paren_token.span.join())
            .unwrap(),
        Expr::Cast(ExprCast {
            attrs: _,
            expr,
            as_token: _,
            ty: _,
        })
        | Expr::Reference(ExprReference {
            attrs: _,
            and_token: _,
            mutability: _,
            expr,
        }) => from_expr_to_span(expr),
        Expr::If(ExprIf {
            attrs: _,
            if_token,
            cond: _,
            then_branch: _,
            else_branch: Some((_, else_branch)),
        }) => if_token.span.join(from_expr_to_span(else_branch)).unwrap(),
        Expr::If(ExprIf {
            attrs: _,
            if_token,
            cond: _,
            then_branch,
            else_branch: None,
        }) => if_token
            .span
            .join(then_branch.brace_token.span.join())
            .unwrap(),
        Expr::Index(ExprIndex {
            attrs: _,
            expr,
            bracket_token,
            index: _,
        }) => from_expr_to_span(expr)
            .join(bracket_token.span.join())
            .unwrap(),
        Expr::Let(ExprLet {
            attrs: _,
            let_token,
            pat: _,
            eq_token: _,
            expr,
        }) => let_token.span.join(from_expr_to_span(expr)).unwrap(),
        Expr::Lit(ExprLit { attrs: _, lit }) => lit.span(),
        Expr::Loop(ExprLoop {
            attrs: _,
            label: _,
            loop_token,
            body,
        }) => loop_token.span.join(body.brace_token.span.join()).unwrap(),
        Expr::Macro(ExprMacro {
            attrs: _,
            mac:
                Macro {
                    path: _,
                    bang_token,
                    delimiter,
                    tokens: _,
                },
        }) => bang_token
            .span
            .join(match delimiter {
                syn::MacroDelimiter::Paren(p) => p.span.join(),
                syn::MacroDelimiter::Brace(b) => b.span.join(),
                syn::MacroDelimiter::Bracket(b) => b.span.join(),
            })
            .unwrap(),
        Expr::Paren(ExprParen {
            attrs: _,
            paren_token,
            expr: _,
        }) => paren_token.span.join(),
        Expr::Path(ExprPath {
            attrs: _,
            qself: _,
            path: Path {
                leading_colon: _,
                segments,
            },
        }) => segments
            .first()
            .unwrap()
            .ident
            .span()
            .join(segments.last().unwrap().ident.span())
            .unwrap(),
        Expr::Return(ExprReturn {
            attrs: _,
            return_token,
            expr: None,
        }) => return_token.span,
        Expr::Return(ExprReturn {
            attrs: _,
            return_token,
            expr: Some(expr),
        }) => return_token.span.join(from_expr_to_span(expr)).unwrap(),
        Expr::Unary(ExprUnary { attrs: _, op, expr }) => match op {
            UnOp::Deref(d) => d.span,
            UnOp::Not(n) => n.span,
            UnOp::Neg(n) => n.span,
            _ => unimplemented!("Non-exhaustive"),
        }
        .join(from_expr_to_span(expr))
        .unwrap(),
        Expr::While(ExprWhile {
            attrs: _,
            label: _,
            while_token,
            cond: _,
            body,
        }) => while_token.span.join(body.brace_token.span.join()).unwrap(),
        _ => todo!(),
    }
}

// A helper for converting Syn Span to Bril Position
fn from_span_to_position(
    starting_span: Span,
    ending_span: Option<Span>,
    src: Option<String>,
) -> Position {
    let start = starting_span.start();
    let end = ending_span.map_or(starting_span.end(), |s| s.end());
    Position {
        pos: ColRow {
            col: start.column as u64,
            row: start.line as u64,
        },
        pos_end: Some(ColRow {
            col: end.column as u64,
            row: end.line as u64,
        }),
        src,
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

// A helper for converting Syn Type to Bril Type
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

// A helper for converting Syn function arguments to Bril Argument
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
        fn_token,
        ident,
        generics,
        paren_token,
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
            Some(from_span_to_position(
                fn_token.span,
                Some(paren_token.span.join()),
                state.src.clone(),
            ))
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
        value: Literal::Int(i64::try_from(vars.len()).unwrap()),
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
            value: Literal::Int(i64::try_from(i).unwrap()),
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
    let pos = if state.is_pos {
        Some(from_span_to_position(
            from_expr_to_span(&expr),
            None,
            state.src.clone(),
        ))
    } else {
        None
    };
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
            eq_token: _,
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
                        pos,
                        op_type,
                    }));
                    (None, code)
                }
                Expr::Index(ExprIndex {
                    attrs,
                    expr,
                    bracket_token: _,
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
                        pos: pos.clone(),
                        op_type,
                    }));
                    code1.push(Code::Instruction(Instruction::Effect {
                        args: vec![dest, arg.unwrap()],
                        funcs: Vec::new(),
                        labels: Vec::new(),
                        op: EffectOps::Store,
                        pos,
                    }));
                    (None, code1)
                }
                _ => panic!("can't handle left hand assignment: {left:?}"),
            }
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

            let mut place_expression = None;

            let (value_op, op_type) = match (op, state.get_type_for_ident(arg1.as_ref().unwrap())) {
                (BinOp::Add(_), Type::Int) => (ValueOps::Add, Type::Int),
                (BinOp::Add(_), Type::Float) => (ValueOps::Fadd, Type::Float),
                (BinOp::Sub(_), Type::Int) => (ValueOps::Sub, Type::Int),
                (BinOp::Sub(_), Type::Float) => (ValueOps::Fsub, Type::Float),
                (BinOp::Mul(_), Type::Int) => (ValueOps::Mul, Type::Int),
                (BinOp::Mul(_), Type::Float) => (ValueOps::Fmul, Type::Float),
                (BinOp::Div(_), Type::Int) => (ValueOps::Div, Type::Int),
                (BinOp::Div(_), Type::Float) => (ValueOps::Fdiv, Type::Float),
                (BinOp::And(_), _) => (ValueOps::And, Type::Bool),
                (BinOp::Or(_), _) => (ValueOps::Or, Type::Bool),
                (BinOp::Eq(_), Type::Int) => (ValueOps::Eq, Type::Bool),
                (BinOp::Eq(_), Type::Float) => (ValueOps::Feq, Type::Bool),
                (BinOp::Lt(_), Type::Int) => (ValueOps::Lt, Type::Bool),
                (BinOp::Lt(_), Type::Float) => (ValueOps::Flt, Type::Bool),
                (BinOp::Le(_), Type::Int) => (ValueOps::Le, Type::Bool),
                (BinOp::Le(_), Type::Float) => (ValueOps::Fle, Type::Bool),
                (BinOp::Ge(_), Type::Int) => (ValueOps::Ge, Type::Bool),
                (BinOp::Ge(_), Type::Float) => (ValueOps::Fge, Type::Bool),
                (BinOp::Gt(_), Type::Int) => (ValueOps::Gt, Type::Bool),
                (BinOp::Gt(_), Type::Float) => (ValueOps::Fgt, Type::Bool),

                // For the assignment operations, the left hand side is being mutated and is restricted to being a rust "place expression"
                // So we need to set a specific destination
                // https://doc.rust-lang.org/reference/expressions.html#place-expressions-and-value-expressions
                (BinOp::AddAssign(_), Type::Int) => {
                    place_expression = arg1.clone();
                    (ValueOps::Add, Type::Int)
                }
                (BinOp::AddAssign(_), Type::Float) => {
                    place_expression = arg1.clone();
                    (ValueOps::Fadd, Type::Float)
                }
                (BinOp::SubAssign(_), Type::Int) => {
                    place_expression = arg1.clone();
                    (ValueOps::Sub, Type::Int)
                }
                (BinOp::SubAssign(_), Type::Float) => {
                    place_expression = arg1.clone();
                    (ValueOps::Fsub, Type::Float)
                }
                (BinOp::MulAssign(_), Type::Int) => {
                    place_expression = arg1.clone();
                    (ValueOps::Mul, Type::Int)
                }
                (BinOp::MulAssign(_), Type::Float) => {
                    place_expression = arg1.clone();
                    (ValueOps::Fmul, Type::Float)
                }
                (BinOp::DivAssign(_), Type::Int) => {
                    place_expression = arg1.clone();
                    (ValueOps::Div, Type::Int)
                }
                (BinOp::DivAssign(_), Type::Float) => {
                    place_expression = arg1.clone();
                    (ValueOps::Fdiv, Type::Float)
                }
                (_, _) => unimplemented!("{op:?}"),
            };

            let dest = place_expression.unwrap_or_else(|| state.fresh_var(op_type.clone()));

            code1.push(Code::Instruction(Instruction::Value {
                args: vec![arg1.unwrap(), arg2.unwrap()],
                dest: dest.clone(),
                funcs: Vec::new(),
                labels: Vec::new(),
                op: value_op,
                pos,
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
            paren_token: _,
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
            if f == "drop" {
                code.push(Code::Instruction(Instruction::Effect {
                    args: vars,
                    funcs: Vec::new(),
                    labels: Vec::new(),
                    op: EffectOps::Free,
                    pos,
                }));
                (None, code)
            } else {
                match state.get_ret_type_for_func(&f) {
                    None => {
                        code.push(Code::Instruction(Instruction::Effect {
                            args: vars,
                            funcs: vec![f],
                            labels: Vec::new(),
                            op: EffectOps::Call,
                            pos,
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
                            pos,
                            op_type: ret,
                        }));
                        (Some(dest), code)
                    }
                }
            }
        }
        Expr::Cast(ExprCast {
            attrs,
            expr,
            as_token: _,
            ty,
        }) if attrs.is_empty() => {
            if let SType::Path(TypePath { qself: None, path }) = *ty {
                // ignore casts to usize
                if path.get_ident().is_some() && path.get_ident().unwrap() == "usize" {
                    from_expr_to_bril(*expr, state)
                } else {
                    panic!("can't handle type in cast: {path:?}");
                }
            } else {
                panic!("can't handle type in cast: {ty:?}");
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
                pos: pos.clone(),
            }));
            code.push(Code::Label {
                label: then_label,
                pos: pos.clone(),
            });

            code.append(&mut from_block_to_vec_code(then_branch, state));

            code.push(Code::Instruction(Instruction::Effect {
                args: Vec::new(),
                funcs: Vec::new(),
                labels: vec![end_label.clone()],
                op: EffectOps::Jump,
                pos: pos.clone(),
            }));
            code.push(Code::Label {
                label: else_label,
                pos: pos.clone(),
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
                pos: pos.clone(),
            }));
            code.push(Code::Label {
                label: end_label,
                pos,
            });
            (None, code)
        }
        Expr::Index(ExprIndex {
            attrs,
            expr,
            bracket_token: _,
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
                pos: pos.clone(),
                op_type: pointer_type,
            }));
            let load_dest = state.fresh_var(load_type.clone());
            code1.push(Code::Instruction(Instruction::Value {
                args: vec![dest],
                dest: load_dest.clone(),
                funcs: Vec::new(),
                labels: Vec::new(),
                op: ValueOps::Load,
                pos,
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
                        pos,
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
                        pos,
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
                        pos,
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
                    bang_token: _,
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
                    pos,
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
            return_token: _,
            expr,
        }) if attrs.is_empty() => {
            let (args, mut code) = expr.map_or_else(
                || (Vec::new(), Vec::new()),
                |e| {
                    let (a, c) = from_expr_to_bril(*e, state);
                    (vec![a.unwrap()], c)
                },
            );
            code.push(Code::Instruction(Instruction::Effect {
                args,
                funcs: Vec::new(),
                labels: Vec::new(),
                op: EffectOps::Return,
                pos,
            }));
            (None, code)
        }
        Expr::Unary(ExprUnary { attrs, op, expr }) if attrs.is_empty() => {
            let (arg, mut code) = from_expr_to_bril(*expr, state);

            let mut args = vec![arg.clone().unwrap()];

            let (op, op_type) = match op {
                UnOp::Deref(_) => (ValueOps::Id, state.get_type_for_ident(&arg.unwrap())),
                UnOp::Not(_) => (ValueOps::Not, Type::Bool),
                UnOp::Neg(_) => {
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
                        ty,
                    )
                }
                _ => unimplemented!("Non-exhaustive"),
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
                pos: pos.clone(),
            }];
            code.append(&mut cond_code);
            code.push(Code::Instruction(Instruction::Effect {
                args: vec![cond_var.unwrap()],
                funcs: Vec::new(),
                labels: vec![then_label.clone(), end_label.clone()],
                op: EffectOps::Branch,
                pos: pos.clone(),
            }));
            code.push(Code::Label {
                label: then_label,
                pos: pos.clone(),
            });

            code.append(&mut from_block_to_vec_code(body, state));

            code.push(Code::Instruction(Instruction::Effect {
                args: Vec::new(),
                funcs: Vec::new(),
                labels: vec![start_label],
                op: EffectOps::Jump,
                pos: pos.clone(),
            }));

            code.push(Code::Label {
                label: end_label,
                pos,
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
            semi_token,
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
                    let expr = init.unwrap().expr;
                    let (arg, mut code) = from_expr_to_bril(*expr, state);
                    code.push(Code::Instruction(Instruction::Value {
                        args: vec![arg.unwrap()],
                        dest,
                        funcs: Vec::new(),
                        labels: Vec::new(),
                        op: ValueOps::Id,
                        pos: if state.is_pos {
                            Some(from_span_to_position(
                                let_token.span,
                                Some(semi_token.span),
                                state.src.clone(),
                            ))
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
        Stmt::Expr(e, _) => {
            let (_, code) = from_expr_to_bril(e, state);
            code
        }
        Stmt::Macro(StmtMacro {
            attrs,
            mac,
            semi_token: _,
        }) => {
            // Currently the only supported macro is println?
            // So we just dispatch StmtMacro as an ExprMacro
            let (_, code) = from_expr_to_bril(Expr::Macro(ExprMacro { attrs, mac }), state);
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
    src: Option<String>,
) -> Program {
    assert!(shebang.is_none(), "can't handle shebang items in Rust file");

    assert!(attrs.is_empty(), "can't handle attributes in Rust file");

    let mut state = State::new(is_pos, src);

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
