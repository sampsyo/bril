mod lib;

use lazy_static::lazy_static;
use std::collections::{HashMap, HashSet};
use std::io::{self, Read};
use std::sync::Mutex;

use lib::*;

fn parse_file() -> syn::File {
    let mut buffer = String::new();
    io::stdin().read_to_string(&mut buffer).unwrap();

    let tree: syn::File = syn::parse_str(&buffer[..]).unwrap();
    return tree;
}

lazy_static! {
    static ref UNIQUE_ID_COUNTER: Mutex<i32> = Mutex::new(0);
}

fn unique_id() -> String {
    let mut counter = UNIQUE_ID_COUNTER.lock().unwrap();
    *counter += 1;
    format!("_id_{}", *counter).to_string()
}

fn unique_label() -> String {
    let mut counter = UNIQUE_ID_COUNTER.lock().unwrap();
    *counter += 1;
    format!("_label_{}", *counter).to_string()
}

fn process_type(
    typ: &syn::Type,
    type_map: &HashMap<String, (Type, HashMap<String, usize>)>,
) -> Type {
    if typ == &syn::parse_quote!(i32) {
        Type::Int
    } else if typ == &syn::parse_quote!(bool) {
        Type::Bool
    } else if typ == &syn::parse_quote!(()) {
        Type::unit()
    } else {
        match typ {
            syn::Type::Tuple(tuple) => {
                let mut typs = Vec::new();
                for typ in tuple.elems.iter() {
                    typs.push(process_type(typ, type_map));
                }
                Type::Product(typs)
            }
            syn::Type::Path(syn::TypePath { path, .. }) => {
                let name = process_path(&path);
                match type_map.get(&name) {
                    Some((typ, _)) => typ.clone(),
                    None => panic!("unrecognized type: {}", name),
                }
            }
            _ => panic!("unrecognized type: {:?}", typ),
        }
    }
}

fn process_pat(pat: &syn::Pat) -> String {
    match pat {
        syn::Pat::Ident(id) => id.ident.to_string(),
        syn::Pat::Type(syn::PatType { pat, .. }) => process_pat(pat),
        _ => panic!("unsupported pattern: {:?}", pat),
    }
}

fn process_pat_typ(
    pat: &syn::Pat,
    type_map: &HashMap<String, (Type, HashMap<String, usize>)>,
) -> Type {
    match pat {
        syn::Pat::Type(syn::PatType { ty, .. }) => process_type(ty, type_map),
        _ => panic!("missing type annotation for {:?}", pat),
    }
}

fn process_path_parts(path: &syn::Path) -> Vec<String> {
    path.segments
        .iter()
        .map(|seg| seg.ident.to_string())
        .collect()
}

fn process_path(path: &syn::Path) -> String {
    match &process_path_parts(path)[..] {
        [ident] => ident.clone(),
        _ => panic!("expected path of length one, got {:?} instead", path),
    }
}

fn process_binop(binop: &syn::BinOp) -> (ValueOps, Type, Type) {
    match binop {
        syn::BinOp::Add(_) => (ValueOps::Add, Type::Int, Type::Int),
        syn::BinOp::Sub(_) => (ValueOps::Sub, Type::Int, Type::Int),
        syn::BinOp::Mul(_) => (ValueOps::Mul, Type::Int, Type::Int),
        syn::BinOp::Div(_) => (ValueOps::Div, Type::Int, Type::Int),
        syn::BinOp::And(_) => (ValueOps::And, Type::Bool, Type::Bool),
        syn::BinOp::Or(_) => (ValueOps::Or, Type::Bool, Type::Bool),
        syn::BinOp::Eq(_) => (ValueOps::Eq, Type::Int, Type::Bool),
        syn::BinOp::Lt(_) => (ValueOps::Lt, Type::Int, Type::Bool),
        syn::BinOp::Le(_) => (ValueOps::Le, Type::Int, Type::Bool),
        syn::BinOp::Gt(_) => (ValueOps::Gt, Type::Int, Type::Bool),
        syn::BinOp::Ge(_) => (ValueOps::Ge, Type::Int, Type::Bool),
        _ => panic!("unsupported binop: {:?}", binop),
    }
}

fn process_expr(
    name: &str,
    typ: &Type,
    expr: &syn::Expr,
    func_map: &HashMap<String, Function>,
    type_map: &HashMap<String, (Type, HashMap<String, usize>)>,
    return_type: &Option<Type>,
) -> Vec<Code> {
    match expr {
        syn::Expr::Path(syn::ExprPath { path, .. }) => match &process_path_parts(path)[..] {
            [var] => vec![Code::Instruction(Instruction::Value {
                op: ValueOps::Id,
                dest: name.to_string(),
                op_type: typ.clone(),
                args: vec![var.clone()],
                funcs: vec![],
                labels: vec![],
            })],
            [typ_name, constructor] => {
                let (typ, constructor_map) = match type_map.get(typ_name) {
                    Some(res) => res,
                    None => panic!("unknown type: {:?}", typ_name),
                };

                let index = match constructor_map.get(constructor) {
                    Some(index) => index,
                    None => panic!(
                        "undefined constructor {:?} for type {:?}",
                        constructor, typ_name
                    ),
                };

                let tmp = unique_id();

                let mut expr = Vec::new();
                expr.push(Code::Instruction(Instruction::Value {
                    op: ValueOps::Pack,
                    dest: tmp.clone(),
                    op_type: Type::unit(),
                    args: vec![],
                    funcs: vec![],
                    labels: vec![],
                }));
                expr.push(Code::Instruction(Instruction::Value {
                    op: ValueOps::Construct,
                    dest: name.to_string(),
                    op_type: typ.clone(),
                    args: vec![tmp, index.to_string()],
                    funcs: vec![],
                    labels: vec![],
                }));

                expr
            }
            _ => panic!("invalid path: {:?}", path),
        },
        syn::Expr::Paren(syn::ExprParen { expr, .. }) => {
            process_expr(name, typ, expr, func_map, type_map, return_type)
        }
        syn::Expr::Lit(syn::ExprLit {
            lit: syn::Lit::Int(lit),
            ..
        }) => {
            let val: i64 = lit.base10_parse().unwrap();
            vec![Code::Instruction(Instruction::Constant {
                op: ConstOps::Const,
                dest: name.to_string(),
                const_type: Type::Int,
                value: Literal::Int(val),
            })]
        }
        syn::Expr::Lit(syn::ExprLit {
            lit: syn::Lit::Bool(lit),
            ..
        }) => vec![Code::Instruction(Instruction::Constant {
            op: ConstOps::Const,
            dest: name.to_string(),
            const_type: Type::Bool,
            value: Literal::Bool(lit.value),
        })],
        syn::Expr::Assign(syn::ExprAssign { left, right, .. }) => match *left.clone() {
            syn::Expr::Type(syn::ExprType { expr, ty, .. }) => match *expr.clone() {
                syn::Expr::Path(syn::ExprPath { path, .. }) => {
                    let var = process_path(&path);
                    process_expr(
                        &var,
                        &process_type(&ty, &type_map),
                        &right,
                        &func_map,
                        &type_map,
                        &return_type,
                    )
                }
                _ => panic!("unsupported left hand side of assignment"),
            },
            _ => panic!("assignments must be annotated with types"),
        },
        syn::Expr::Unary(syn::ExprUnary {
            op: syn::UnOp::Not(_),
            expr: e,
            ..
        }) => {
            let tmp = unique_id();

            let mut expr = Vec::new();
            expr.append(&mut process_expr(
                &tmp,
                &Type::Bool,
                &e,
                &func_map,
                &type_map,
                &return_type,
            ));
            expr.push(Code::Instruction(Instruction::Value {
                op: ValueOps::Not,
                dest: name.to_string(),
                op_type: Type::Bool,
                args: vec![tmp],
                funcs: vec![],
                labels: vec![],
            }));

            expr
        }
        syn::Expr::Unary(syn::ExprUnary {
            op: syn::UnOp::Neg(_),
            expr: e,
            ..
        }) => {
            let tmp = unique_id();
            let neg_one = unique_id();

            let mut expr = Vec::new();
            expr.append(&mut process_expr(
                &tmp,
                &Type::Int,
                &e,
                &func_map,
                &type_map,
                &return_type,
            ));
            expr.push(Code::Instruction(Instruction::Constant {
                op: ConstOps::Const,
                dest: neg_one.clone(),
                const_type: Type::Int,
                value: Literal::Int(-1),
            }));
            expr.push(Code::Instruction(Instruction::Value {
                op: ValueOps::Mul,
                dest: name.to_string(),
                op_type: Type::Int,
                args: vec![tmp, neg_one],
                funcs: vec![],
                labels: vec![],
            }));

            expr
        }
        syn::Expr::Binary(syn::ExprBinary {
            left, op, right, ..
        }) => {
            let (binop, arg_typ, ret_typ) = process_binop(op);
            let tmp1 = unique_id();
            let tmp2 = unique_id();

            let mut expr = Vec::new();
            expr.append(&mut process_expr(
                &tmp1,
                &arg_typ,
                left,
                &func_map,
                &type_map,
                &return_type,
            ));
            expr.append(&mut process_expr(
                &tmp2,
                &arg_typ,
                right,
                &func_map,
                &type_map,
                &return_type,
            ));
            expr.push(Code::Instruction(Instruction::Value {
                op: binop,
                dest: name.to_string(),
                op_type: ret_typ,
                args: vec![tmp1, tmp2],
                funcs: vec![],
                labels: vec![],
            }));

            expr
        }
        syn::Expr::Call(syn::ExprCall { func, args, .. }) => match *func.clone() {
            syn::Expr::Path(syn::ExprPath { path, .. }) => match &process_path_parts(&path)[..] {
                [func_name] => {
                    let func_header = match func_map.get(func_name) {
                        Some(header) => header,
                        None => panic!("undefined function: {}", func_name),
                    };

                    let mut arg_list = Vec::new();
                    let mut expr = Vec::new();

                    if args.len() > 0 {
                        let header_args = func_header.args.clone();
                        if args.len() != header_args.len() {
                            panic!("incompatible function call arguments arity");
                        }
                        for (arg, sig) in args.iter().zip(header_args) {
                            let tmp = unique_id();
                            expr.append(&mut process_expr(
                                &tmp,
                                &sig.arg_type,
                                arg,
                                &func_map,
                                &type_map,
                                &return_type,
                            ));
                            arg_list.push(tmp);
                        }
                    }

                    let ret_type = func_header.return_type.clone().unwrap();

                    expr.push(if ret_type.is_unit() {
                        Code::Instruction(Instruction::Effect {
                            op: EffectOps::Call,
                            args: arg_list,
                            funcs: vec![func_name.clone()],
                            labels: vec![],
                        })
                    } else {
                        Code::Instruction(Instruction::Value {
                            op: ValueOps::Call,
                            dest: name.to_string(),
                            op_type: ret_type,
                            args: arg_list,
                            funcs: vec![func_name.clone()],
                            labels: vec![],
                        })
                    });

                    expr
                }
                [typ_name, constructor] => {
                    let (typ, constructor_map) = match type_map.get(typ_name) {
                        Some(res) => res,
                        None => panic!("unknown type: {:?}", typ_name),
                    };

                    let index = match constructor_map.get(constructor) {
                        Some(res) => res,
                        None => panic!(
                            "undefined constructor {:?} for type {:?}",
                            constructor, typ_name
                        ),
                    };

                    let typs = match typ {
                        Type::Sum(typs) => match &typs[index.clone()] {
                            Type::Product(typs) => typs,
                            _ => panic!("impossible???"),
                        },
                        _ => panic!("impossible??"),
                    };

                    if args.len() != typs.len() {
                        panic!("incorrect number of constructor arguments for constructor");
                    }

                    let tmp = unique_id();

                    let mut expr = Vec::new();
                    let mut arg_list = Vec::new();

                    for (arg, sig) in args.iter().zip(typs) {
                        let tmp = unique_id();
                        expr.append(&mut process_expr(
                            &tmp,
                            &sig,
                            &arg,
                            &func_map,
                            &type_map,
                            &return_type,
                        ));
                        arg_list.push(tmp);
                    }

                    expr.push(Code::Instruction(Instruction::Value {
                        op: ValueOps::Pack,
                        dest: tmp.clone(),
                        op_type: Type::Product(typs.clone()),
                        args: arg_list,
                        funcs: vec![],
                        labels: vec![],
                    }));
                    expr.push(Code::Instruction(Instruction::Value {
                        op: ValueOps::Construct,
                        dest: name.to_string(),
                        op_type: typ.clone(),
                        args: vec![tmp, index.to_string()],
                        funcs: vec![],
                        labels: vec![],
                    }));

                    expr
                }
                _ => panic!("arbitrary function expressions not supported"),
            },
            _ => panic!("unuspported function call syntax: {:?}", func),
        },
        syn::Expr::Macro(syn::ExprMacro {
            mac: syn::Macro { path, tokens, .. },
            ..
        }) => match &process_path(path)[..] {
            "println" => {
                let args: Vec<String> = tokens
                    .clone()
                    .into_iter()
                    .map(|token| token.to_string())
                    .filter(|s| s != ",")
                    .collect();
                vec![Code::Instruction(Instruction::Effect {
                    op: EffectOps::Print,
                    args: args,
                    funcs: vec![],
                    labels: vec![],
                })]
            }
            mac => panic!("unsupported macro: {:?}", mac),
        },
        syn::Expr::If(syn::ExprIf {
            cond,
            then_branch,
            else_branch,
            ..
        }) => {
            let tmp = unique_id();
            let b_tr = format!("{}_{}_true", unique_label(), tmp);
            let b_fa = format!("{}_{}_false", unique_label(), tmp);
            let end = format!("{}_{}_end", unique_label(), tmp);

            let mut expr = Vec::new();
            expr.append(&mut process_expr(
                &tmp,
                &Type::Bool,
                &cond,
                &func_map,
                &type_map,
                &return_type,
            ));
            expr.push(Code::Instruction(Instruction::Effect {
                op: EffectOps::Branch,
                args: vec![tmp],
                funcs: vec![],
                labels: vec![b_tr.clone(), b_fa.clone()],
            }));
            expr.push(Code::Label { label: b_tr });
            expr.append(&mut process_block(
                &then_branch,
                &func_map,
                &type_map,
                &return_type,
            ));
            expr.push(Code::Instruction(Instruction::Effect {
                op: EffectOps::Jump,
                args: vec![],
                funcs: vec![],
                labels: vec![end.clone()],
            }));
            expr.push(Code::Label { label: b_fa });
            match else_branch {
                Some((_, branch)) => expr.append(&mut process_expr(
                    &unique_id(),
                    &Type::unit(),
                    &branch,
                    &func_map,
                    &type_map,
                    &return_type,
                )),
                None => (),
            };
            expr.push(Code::Label { label: end });

            expr
        }
        syn::Expr::Block(syn::ExprBlock { block, .. }) => {
            process_block(&block, &func_map, &type_map, &return_type)
        }
        syn::Expr::While(syn::ExprWhile { cond, body, .. }) => {
            let tmp = unique_id();
            let header = format!("{}_{}_header", unique_label(), tmp);
            let start = format!("{}_{}_start", unique_label(), tmp);
            let end = format!("{}_{}_end", unique_label(), tmp);

            let mut expr = Vec::new();
            expr.push(Code::Label {
                label: header.clone(),
            });
            expr.append(&mut process_expr(
                &tmp,
                &Type::Bool,
                &cond,
                &func_map,
                &type_map,
                &return_type,
            ));
            expr.push(Code::Instruction(Instruction::Effect {
                op: EffectOps::Branch,
                args: vec![tmp],
                funcs: vec![],
                labels: vec![start.clone(), end.clone()],
            }));
            expr.push(Code::Label { label: start });
            expr.append(&mut process_block(
                &body,
                &func_map,
                &type_map,
                &return_type,
            ));
            expr.push(Code::Instruction(Instruction::Effect {
                op: EffectOps::Jump,
                args: vec![],
                funcs: vec![],
                labels: vec![header],
            }));
            expr.push(Code::Label { label: end });

            expr
        }
        syn::Expr::Tuple(syn::ExprTuple { elems, .. }) => {
            let typs = match typ {
                Type::Product(typs) => typs,
                _ => panic!("expected product typ, not {:?}", typ),
            };

            if typs.len() != elems.len() {
                panic!("tuple expression has mismatched arity")
            }

            let mut expr = Vec::new();
            let mut tmps = Vec::new();

            for (elem, typ) in elems.iter().zip(typs.iter()) {
                let tmp = unique_id();
                expr.append(&mut process_expr(
                    &tmp,
                    &typ,
                    &elem,
                    &func_map,
                    &type_map,
                    &return_type,
                ));
                tmps.push(tmp);
            }

            expr.push(Code::Instruction(Instruction::Value {
                op: ValueOps::Pack,
                dest: name.to_string(),
                op_type: typ.clone(),
                args: tmps,
                funcs: vec![],
                labels: vec![],
            }));

            expr
        }
        syn::Expr::Field(syn::ExprField {
            base,
            member: syn::Member::Unnamed(syn::Index { index, .. }),
            ..
        }) => match *base.clone() {
            syn::Expr::Path(syn::ExprPath { path, .. }) => {
                let var = process_path(&path);

                vec![Code::Instruction(Instruction::Value {
                    op: ValueOps::Unpack,
                    dest: name.to_string(),
                    op_type: typ.clone(),
                    args: vec![var, index.to_string()],
                    funcs: vec![],
                    labels: vec![],
                })]
            }
            _ => panic!("tuple-indexing arbitrary expressions is not supported"),
        },
        syn::Expr::Match(syn::ExprMatch { expr: e, arms, .. }) => {
            let mut variant_data = None;
            let mut branches = HashMap::new();
            for arm in arms {
                let (path, args) = match arm.pat.clone() {
                    syn::Pat::TupleStruct(syn::PatTupleStruct {
                        path,
                        pat: syn::PatTuple { elems, .. },
                        ..
                    }) => (path, elems.iter().map(|pat| process_pat(pat)).collect()),
                    syn::Pat::Path(syn::PatPath { path, .. }) => (path, Vec::new()),
                    _ => panic!("invalid match pattern: {:?}", arm.pat),
                };

                let (variant_name, constr) = match &process_path_parts(&path)[..] {
                    [variant_name, constr] => (variant_name.clone(), constr.clone()),
                    _ => panic!("unsupported pattern: {:?}", path),
                };

                let index_map = match &variant_data {
                    None => match type_map.get(&variant_name) {
                        Some((typ, index_map)) => {
                            variant_data = Some((variant_name, typ, index_map));
                            index_map
                        }
                        None => panic!("unrecognized type: {:?}", variant_name),
                    },
                    Some((existing_variant_name, _, index_map)) => {
                        if existing_variant_name != &variant_name {
                            panic!(
                                "multiple types in match: {:?}, {:?}",
                                existing_variant_name, variant_name
                            )
                        };
                        index_map
                    }
                };

                let index = match index_map.get(&constr) {
                    Some(index) => index,
                    None => panic!("should be impossible??"),
                };

                branches.insert(
                    index,
                    (
                        args,
                        *arm.body.clone(),
                        format!("{}_branch_{}", unique_label(), index),
                    ),
                );
            }

            let (_variant_name, variant_typ, constr_typs) = match variant_data {
                Some((variant_name, variant_typ, _)) => match &variant_typ {
                    Type::Sum(typs) => (variant_name, variant_typ, typs),
                    _ => panic!("should be impossible???"),
                },
                None => panic!("match statement cannot be empty"),
            };

            if constr_typs.len() != branches.len() {
                panic!("match missing constructors!");
            }

            let tmp = unique_id();
            let dst = unique_id();
            let end = format!("{}_match_end", unique_label());

            let mut expr = Vec::new();

            expr.append(&mut process_expr(
                &tmp,
                &variant_typ,
                &e,
                &func_map,
                &type_map,
                &return_type,
            ));

            let mut labels = Vec::new();
            for index in 0..branches.len() {
                match branches.get(&index) {
                    Some((_, _, label)) => labels.push(label.clone()),
                    None => panic!("should be impossible????"),
                };
            }

            expr.push(Code::Instruction(Instruction::Value {
                op: ValueOps::Destruct,
                dest: dst.clone(),
                op_type: variant_typ.clone(),
                args: vec![tmp],
                funcs: vec![],
                labels: labels,
            }));

            for index in 0..branches.len() {
                let (args, block, label) = match branches.get(&index) {
                    Some(data) => data,
                    None => panic!("should be impossible????"),
                };

                let arg_typs = match &constr_typs[index] {
                    Type::Product(arg_typs) => arg_typs,
                    _ => panic!("should be impossible?????"),
                };

                if arg_typs.len() != args.len() {
                    panic!(
                        "invalid arity for match pattern: expected {}, got {}",
                        arg_typs.len(),
                        args.len()
                    );
                }

                expr.push(Code::Label {
                    label: label.clone(),
                });

                for (i, arg) in args.iter().enumerate() {
                    expr.push(Code::Instruction(Instruction::Value {
                        op: ValueOps::Unpack,
                        dest: arg.clone(),
                        op_type: arg_typs[i].clone(),
                        args: vec![dst.clone(), i.to_string()],
                        funcs: vec![],
                        labels: vec![],
                    }));
                }

                expr.append(&mut process_expr(
                    &name,
                    &typ,
                    &block,
                    &func_map,
                    &type_map,
                    &return_type,
                ));

                expr.push(Code::Instruction(Instruction::Effect {
                    op: EffectOps::Jump,
                    args: vec![],
                    funcs: vec![],
                    labels: vec![end.clone()],
                }));
            }

            expr.push(Code::Label { label: end });

            expr
        }
        _ => panic!("unsupported expression: {:?}", expr),
    }
}

fn process_stmt(
    stmt: &syn::Stmt,
    return_type: &Option<Type>,
    func_map: &HashMap<String, Function>,
    type_map: &HashMap<String, (Type, HashMap<String, usize>)>,
) -> Vec<Code> {
    match stmt {
        syn::Stmt::Local(local) => match &local.init {
            Some((_, expr)) => process_expr(
                &process_pat(&local.pat),
                &process_pat_typ(&local.pat, &type_map),
                &expr,
                &func_map,
                &type_map,
                &return_type,
            ),
            _ => panic!("must specify type and value for let binding"),
        },
        syn::Stmt::Semi(syn::Expr::Return(syn::ExprReturn { expr, .. }), _) => {
            match (expr, return_type) {
                (Some(e), Some(ret_type)) => {
                    let tmp = unique_id();

                    let mut expr = Vec::new();
                    expr.append(&mut process_expr(
                        &tmp,
                        &ret_type,
                        &e,
                        &func_map,
                        &type_map,
                        &return_type,
                    ));
                    expr.push(Code::Instruction(Instruction::Effect {
                        op: EffectOps::Return,
                        args: vec![tmp],
                        funcs: vec![],
                        labels: vec![],
                    }));

                    expr
                }
                (None, None) => vec![Code::Instruction(Instruction::Effect {
                    op: EffectOps::Return,
                    args: vec![],
                    funcs: vec![],
                    labels: vec![],
                })],
                (Some(_), None) => {
                    panic!("attempt to return value when function has no return type")
                }
                (None, Some(_)) => panic!("function should specify return type"),
            }
        }
        syn::Stmt::Semi(expr, _) | syn::Stmt::Expr(expr) => {
            // make a dummy variable to store the output value and don't use it
            process_expr(
                &unique_id()[..],
                &Type::unit(),
                &expr,
                &func_map,
                &type_map,
                &return_type,
            )
        }
        _ => panic!("unsupported stmt: {:?}", stmt),
    }
}

fn process_block(
    block: &syn::Block,
    func_map: &HashMap<String, Function>,
    type_map: &HashMap<String, (Type, HashMap<String, usize>)>,
    return_type: &Option<Type>,
) -> Vec<Code> {
    let mut instrs = Vec::new();
    for stmt in &block.stmts[..] {
        instrs.append(&mut process_stmt(&stmt, &return_type, &func_map, &type_map));
    }
    instrs
}

fn process_func(
    func: &syn::ItemFn,
    func_map: &HashMap<String, Function>,
    type_map: &HashMap<String, (Type, HashMap<String, usize>)>,
    get_instrs: bool,
) -> Function {
    let name = func.sig.ident.to_string();
    let return_type = match &func.sig.output {
        syn::ReturnType::Default => {
            if get_instrs {
                None
            } else {
                Some(Type::unit())
            }
        }
        syn::ReturnType::Type(_, typ) => Some(process_type(&typ, &type_map)),
    };

    let mut args = Vec::new();
    for arg in func.sig.inputs.iter() {
        match arg {
            syn::FnArg::Receiver(_) => panic!("unsupported receiver"),
            syn::FnArg::Typed(pat_type) => {
                args.push(Argument {
                    name: process_pat(&pat_type.pat),
                    arg_type: process_type(&pat_type.ty, &type_map),
                });
            }
        }
    }

    let mut instrs = Vec::new();
    if get_instrs {
        instrs.append(&mut process_block(
            &func.block,
            &func_map,
            &type_map,
            &return_type,
        ));
    }

    Function {
        name,
        args,
        return_type,
        instrs,
    }
}

fn process_enum(
    enm: &syn::ItemEnum,
    type_map: &HashMap<String, (Type, HashMap<String, usize>)>,
) -> (String, HashMap<String, usize>, Type) {
    let name = enm.ident.to_string();

    let mut variants = Vec::new();
    let mut constructor_map = HashMap::new();

    for (i, variant) in enm.variants.iter().enumerate() {
        constructor_map.insert(variant.ident.to_string(), i);
        let typ = match &variant.fields {
            syn::Fields::Unit => Type::unit(),
            syn::Fields::Unnamed(syn::FieldsUnnamed { unnamed, .. }) => Type::Product(
                unnamed
                    .iter()
                    .map(|field| process_type(&field.ty, &type_map))
                    .collect(),
            ),
            syn::Fields::Named(_) => panic!("named types in constructors not supported"),
        };
        variants.push(typ.clone());
    }

    let typ = Type::Sum(variants.iter().map(|typ| typ.clone()).collect());

    (name, constructor_map, typ)
}

fn main() {
    let tree = parse_file();

    let mut functions = Vec::new();
    let mut type_map = HashMap::new();
    for item in tree.items {
        match item {
            syn::Item::Fn(function) => functions.push(function),
            syn::Item::Enum(enm) => {
                let (name, constructor_map, typ) = process_enum(&enm, &type_map);
                type_map.insert(name, (typ, constructor_map));
            }
            _ => (),
        }
    }

    let mut func_map = HashMap::new();
    for function in &functions[..] {
        let func = process_func(function, &func_map, &type_map, false);
        func_map.insert(func.name.clone(), func);
    }

    let mut program = Program {
        functions: Vec::new(),
    };

    for function in &functions[..] {
        program
            .functions
            .push(process_func(function, &func_map, &type_map, true))
    }

    output_program(&program);
}
