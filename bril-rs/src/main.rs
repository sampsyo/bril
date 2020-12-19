mod lib;

use lazy_static::lazy_static;
use std::collections::HashMap;
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

fn process_type(typ: &syn::Type) -> Type {
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
                    typs.push(process_type(typ));
                }
                Type::Sum(typs)
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

fn process_pat_typ(pat: &syn::Pat) -> Type {
    match pat {
        syn::Pat::Type(syn::PatType { ty, .. }) => process_type(ty),
        _ => panic!("missing type annotation for {:?}", pat),
    }
}

fn process_path(path: &syn::Path) -> String {
    return path.get_ident().unwrap().to_string();
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
    return_type: &Option<Type>,
) -> Vec<Code> {
    match expr {
        syn::Expr::Path(syn::ExprPath { path, .. }) => {
            vec![Code::Instruction(Instruction::Value {
                op: ValueOps::Id,
                dest: name.to_string(),
                op_type: typ.clone(),
                args: Some(vec![process_path(path)]),
                funcs: None,
                labels: None,
            })]
        }
        syn::Expr::Paren(syn::ExprParen { expr, .. }) => {
            process_expr(name, typ, expr, func_map, return_type)
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
                    process_expr(&var, &process_type(&ty), &right, &func_map, &return_type)
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
                e,
                &func_map,
                &return_type,
            ));
            expr.push(Code::Instruction(Instruction::Value {
                op: ValueOps::Not,
                dest: name.to_string(),
                op_type: Type::Bool,
                args: Some(vec![tmp]),
                funcs: None,
                labels: None,
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
                &return_type,
            ));
            expr.append(&mut process_expr(
                &tmp2,
                &arg_typ,
                right,
                &func_map,
                &return_type,
            ));
            expr.push(Code::Instruction(Instruction::Value {
                op: binop,
                dest: name.to_string(),
                op_type: ret_typ,
                args: Some(vec![tmp1, tmp2]),
                funcs: None,
                labels: None,
            }));

            expr
        }
        syn::Expr::Call(syn::ExprCall { func, args, .. }) => match *func.clone() {
            syn::Expr::Path(syn::ExprPath { path, .. }) => {
                let func_name = process_path(&path);
                let func_header = match func_map.get(&func_name) {
                    Some(header) => header,
                    None => panic!("undefined function: {}", func_name),
                };

                let mut arg_list = Vec::new();
                let mut expr = Vec::new();

                if args.len() > 0 {
                    let header_args = func_header.args.clone().unwrap();
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
                            &return_type,
                        ));
                        arg_list.push(tmp);
                    }
                }

                let ret_type = func_header.return_type.clone().unwrap();

                expr.push(if ret_type.is_unit() {
                    Code::Instruction(Instruction::Effect {
                        op: EffectOps::Call,
                        args: if arg_list.is_empty() {
                            None
                        } else {
                            Some(arg_list)
                        },
                        funcs: Some(vec![func_name]),
                        labels: None,
                    })
                } else {
                    Code::Instruction(Instruction::Value {
                        op: ValueOps::Call,
                        dest: name.to_string(),
                        op_type: ret_type,
                        args: if arg_list.is_empty() {
                            None
                        } else {
                            Some(arg_list)
                        },
                        funcs: Some(vec![func_name]),
                        labels: None,
                    })
                });

                expr
            }
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
                    args: Some(args),
                    funcs: None,
                    labels: None,
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
                cond,
                &func_map,
                &return_type,
            ));
            expr.push(Code::Instruction(Instruction::Effect {
                op: EffectOps::Branch,
                args: Some(vec![tmp]),
                funcs: None,
                labels: Some(vec![b_tr.clone(), b_fa.clone()]),
            }));
            expr.push(Code::Label { label: b_tr });
            expr.append(&mut process_block(&then_branch, &func_map, &return_type));
            expr.push(Code::Instruction(Instruction::Effect {
                op: EffectOps::Jump,
                args: None,
                funcs: None,
                labels: Some(vec![end.clone()]),
            }));
            expr.push(Code::Label { label: b_fa });
            match else_branch {
                Some((_, branch)) => expr.append(&mut process_expr(
                    &unique_id(),
                    &Type::unit(),
                    branch,
                    &func_map,
                    &return_type,
                )),
                None => (),
            };
            expr.push(Code::Label { label: end });

            expr
        }
        syn::Expr::Block(syn::ExprBlock { block, .. }) => {
            process_block(&block, &func_map, &return_type)
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
                cond,
                &func_map,
                &return_type,
            ));
            expr.push(Code::Instruction(Instruction::Effect {
                op: EffectOps::Branch,
                args: Some(vec![tmp]),
                funcs: None,
                labels: Some(vec![start.clone(), end.clone()]),
            }));
            expr.push(Code::Label { label: start });
            expr.append(&mut process_block(&body, &func_map, &return_type));
            expr.push(Code::Instruction(Instruction::Effect {
                op: EffectOps::Jump,
                args: None,
                funcs: None,
                labels: Some(vec![header]),
            }));
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
) -> Vec<Code> {
    match stmt {
        syn::Stmt::Local(local) => match &local.init {
            Some((_, expr)) => process_expr(
                &process_pat(&local.pat),
                &process_pat_typ(&local.pat),
                &expr,
                &func_map,
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
                        e,
                        &func_map,
                        &return_type,
                    ));
                    expr.push(Code::Instruction(Instruction::Effect {
                        op: EffectOps::Return,
                        args: Some(vec![tmp]),
                        funcs: None,
                        labels: None,
                    }));

                    expr
                }
                (None, None) => vec![Code::Instruction(Instruction::Effect {
                    op: EffectOps::Return,
                    args: None,
                    funcs: None,
                    labels: None,
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
                &return_type,
            )
        }
        _ => panic!("unsupported stmt: {:?}", stmt),
    }
}

fn process_block(
    block: &syn::Block,
    func_map: &HashMap<String, Function>,
    return_type: &Option<Type>,
) -> Vec<Code> {
    let mut instrs = Vec::new();
    for stmt in &block.stmts[..] {
        instrs.append(&mut process_stmt(&stmt, &return_type, &func_map));
    }
    instrs
}

fn process_func(
    func: &syn::ItemFn,
    func_map: &HashMap<String, Function>,
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
        syn::ReturnType::Type(_, typ) => Some(process_type(&typ)),
    };

    let mut args = Vec::new();
    for arg in func.sig.inputs.iter() {
        match arg {
            syn::FnArg::Receiver(_) => panic!("unsupported receiver"),
            syn::FnArg::Typed(pat_type) => {
                args.push(Argument {
                    name: process_pat(&pat_type.pat),
                    arg_type: process_type(&pat_type.ty),
                });
            }
        }
    }

    let mut instrs = Vec::new();
    if get_instrs {
        instrs.append(&mut process_block(&func.block, &func_map, &return_type));
    }

    Function {
        name,
        args: if args.is_empty() { None } else { Some(args) },
        return_type,
        instrs,
    }
}

fn main() {
    let tree = parse_file();

    let mut functions = Vec::new();
    for item in tree.items {
        match item {
            syn::Item::Fn(function) => functions.push(function),
            _ => (),
        }
    }

    let mut func_map = HashMap::new();
    for function in &functions[..] {
        let func = process_func(function, &func_map, false);
        func_map.insert(func.name.clone(), func);
    }

    let mut program = Program {
        functions: Vec::new(),
    };

    for function in &functions[..] {
        program
            .functions
            .push(process_func(function, &func_map, true))
    }

    output_program(&program);
}
