// Copyright (C) 2024 Ethan Uppal.
//
// SPDX-License-Identifier: MIT

use bril_frontend::{ast, loc::Loc};
use serde_json::{json, Value};

trait BrilFrontendAstMapExt<Item> {
    fn loc_map_to_vec<Out, F: FnMut(&Item) -> Out>(self, f: F) -> Vec<Out>;
}

impl<'a, Item: 'a, I: Iterator<Item = &'a Loc<Item>>> BrilFrontendAstMapExt<Item> for I {
    fn loc_map_to_vec<Out, F: FnMut(&Item) -> Out>(self, f: F) -> Vec<Out> {
        self.map(std::ops::Deref::deref).map(f).collect()
    }
}

fn extract_label_name(name: &str) -> String {
    name.chars().skip(1).collect()
}

fn extract_function_name(name: &str) -> String {
    name.chars().skip(1).collect()
}

pub fn imported_function_to_json(imported_function: &ast::ImportedFunction) -> Value {
    json!({
        "name": *imported_function.name,
        "alias": imported_function.alias.as_ref().map(|alias| *alias.name)
    })
}

pub fn import_to_json(import: &ast::Import) -> Value {
    let imported_functions = import
        .imported_functions
        .iter()
        .loc_map_to_vec(imported_function_to_json);

    json!({
        "path": *import.path,
        "functions": imported_functions
    })
}

pub fn type_to_json(ty: &ast::Type) -> Value {
    match ty {
        ast::Type::Int => json!("int"),
        ast::Type::Bool => json!("bool"),
        ast::Type::Float => json!("float"),
        ast::Type::Char => json!("char"),
        ast::Type::Ptr(inner) => json!({
            "ptr": type_to_json(inner)
        }),
    }
}

pub fn label_to_json(label: &ast::Label) -> Value {
    json!({
        "label": extract_label_name(&label.name)
    })
}

pub fn constant_value_to_json(constant_value: &ast::ConstantValue) -> Value {
    match constant_value {
        ast::ConstantValue::IntegerLiteral(integer) => json!(**integer),
        ast::ConstantValue::BooleanLiteral(boolean) => json!(**boolean),
        ast::ConstantValue::FloatLiteral(float) => json!(**float),
        ast::ConstantValue::CharacterLiteral(character) => json!(character.to_string()),
    }
}

pub fn constant_to_json(constant: &ast::Constant) -> Value {
    let ty = constant
        .type_annotation
        .as_ref()
        .map(|type_annotation| type_to_json(&type_annotation.ty));

    json!({
        "op": "const",
        "dest": *constant.name,
        "type": ty,
        "value": constant_value_to_json(&constant.value)
    })
}

pub fn decompose_value_operation(
    value_operation_op: &ast::ValueOperationOp,
) -> (&'static str, Vec<String>, Vec<String>, Vec<String>) {
    let temporary;
    let (op, arguments, functions, labels): (&'static str, &[&Loc<&str>], &[&str], &[&ast::Label]) =
        match value_operation_op {
            ast::ValueOperationOp::Add(lhs, rhs) => ("add", &[lhs, rhs], &[], &[]),
            ast::ValueOperationOp::Mul(lhs, rhs) => ("mul", &[lhs, rhs], &[], &[]),
            ast::ValueOperationOp::Sub(lhs, rhs) => ("sub", &[lhs, rhs], &[], &[]),
            ast::ValueOperationOp::Div(lhs, rhs) => ("div", &[lhs, rhs], &[], &[]),
            ast::ValueOperationOp::Eq(lhs, rhs) => ("eq", &[lhs, rhs], &[], &[]),
            ast::ValueOperationOp::Lt(lhs, rhs) => ("lt", &[lhs, rhs], &[], &[]),
            ast::ValueOperationOp::Gt(lhs, rhs) => ("gt", &[lhs, rhs], &[], &[]),
            ast::ValueOperationOp::Le(lhs, rhs) => ("le", &[lhs, rhs], &[], &[]),
            ast::ValueOperationOp::Ge(lhs, rhs) => ("ge", &[lhs, rhs], &[], &[]),
            ast::ValueOperationOp::Not(value) => ("not", &[value], &[], &[]),
            ast::ValueOperationOp::And(lhs, rhs) => ("and", &[lhs, rhs], &[], &[]),
            ast::ValueOperationOp::Or(lhs, rhs) => ("or", &[lhs, rhs], &[], &[]),
            ast::ValueOperationOp::Call(function_name, arguments) => {
                temporary = arguments.iter().collect::<Vec<_>>();
                ("call", temporary.as_slice(), &[function_name], &[])
            }
            ast::ValueOperationOp::Id(value) => ("id", &[value], &[], &[]),
            ast::ValueOperationOp::Fadd(lhs, rhs) => ("fadd", &[lhs, rhs], &[], &[]),
            ast::ValueOperationOp::Fmul(lhs, rhs) => ("fmul", &[lhs, rhs], &[], &[]),
            ast::ValueOperationOp::Fsub(lhs, rhs) => ("fsub", &[lhs, rhs], &[], &[]),
            ast::ValueOperationOp::Fdiv(lhs, rhs) => ("fdiv", &[lhs, rhs], &[], &[]),
            ast::ValueOperationOp::Feq(lhs, rhs) => ("feq", &[lhs, rhs], &[], &[]),
            ast::ValueOperationOp::Flt(lhs, rhs) => ("flt", &[lhs, rhs], &[], &[]),
            ast::ValueOperationOp::Fle(lhs, rhs) => ("fle", &[lhs, rhs], &[], &[]),
            ast::ValueOperationOp::Fgt(lhs, rhs) => ("fgt", &[lhs, rhs], &[], &[]),
            ast::ValueOperationOp::Fge(lhs, rhs) => ("fge", &[lhs, rhs], &[], &[]),
            ast::ValueOperationOp::Alloc(size) => ("alloc", &[size], &[], &[]),
            ast::ValueOperationOp::Load(pointer) => ("load", &[pointer], &[], &[]),
            ast::ValueOperationOp::PtrAdd(pointer, offset) => {
                ("ptradd", &[pointer, offset], &[], &[])
            }
            ast::ValueOperationOp::Ceq(lhs, rhs) => ("ceq", &[lhs, rhs], &[], &[]),
            ast::ValueOperationOp::Clt(lhs, rhs) => ("clt", &[lhs, rhs], &[], &[]),
            ast::ValueOperationOp::Cle(lhs, rhs) => ("cle", &[lhs, rhs], &[], &[]),
            ast::ValueOperationOp::Cgt(lhs, rhs) => ("cgt", &[lhs, rhs], &[], &[]),
            ast::ValueOperationOp::Cge(lhs, rhs) => ("cge", &[lhs, rhs], &[], &[]),
            ast::ValueOperationOp::Char2Int(value) => ("char2int", &[value], &[], &[]),
            ast::ValueOperationOp::Int2Char(value) => ("int2char", &[value], &[], &[]),
        };

    (
        op,
        arguments
            .iter()
            .map(|argument| argument.to_string())
            .collect(),
        functions
            .iter()
            .map(|function| extract_function_name(function))
            .collect(),
        labels
            .iter()
            .map(|label| extract_label_name(&label.name))
            .collect(),
    )
}

pub fn value_operation_to_json(value_operation: &ast::ValueOperation) -> Value {
    let ty = value_operation
        .type_annotation
        .as_ref()
        .map(|type_annotation| type_to_json(&type_annotation.ty));

    let (op, arguments, functions, labels) = decompose_value_operation(&value_operation.op);
    json!({
        "op": op,
        "dest": *value_operation.name,
        "type": ty,
        "args": arguments,
        "funcs": functions,
        "labels": labels
    })
}

pub fn effect_operation_to_json(effect_operation: &ast::EffectOperation) -> Value {
    let temporary;
    let (op, arguments, functions, labels): (&'static str, &[&Loc<&str>], &[&str], &[&ast::Label]) =
        match &*effect_operation.op {
            ast::EffectOperationOp::Jmp(destination) => ("jmp", &[], &[], &[destination]),
            ast::EffectOperationOp::Br(condition, if_true, if_false) => {
                ("br", &[condition], &[], &[if_true, if_false])
            }
            ast::EffectOperationOp::Call(function_name, arguments) => {
                temporary = arguments.iter().collect::<Vec<_>>();
                ("call", temporary.as_slice(), &[function_name], &[])
            }
            ast::EffectOperationOp::Ret(value) => (
                "ret",
                if let Some(value) = value {
                    &[value]
                } else {
                    &[]
                },
                &[],
                &[],
            ),
            ast::EffectOperationOp::Print(arguments) => {
                temporary = arguments.iter().collect::<Vec<_>>();
                ("print", temporary.as_slice(), &[], &[])
            }
            ast::EffectOperationOp::Nop => ("nop", &[], &[], &[]),
            ast::EffectOperationOp::Store(pointer, value) => ("store", &[pointer, value], &[], &[]),
            ast::EffectOperationOp::Free(pointer) => ("free", &[pointer], &[], &[]),
        };

    let arguments = arguments
        .iter()
        .map(|argument| argument.to_string())
        .collect::<Vec<_>>();
    let functions = functions
        .iter()
        .map(|function| extract_function_name(function))
        .collect::<Vec<_>>();
    let labels = labels
        .iter()
        .map(|label| extract_label_name(&label.name))
        .collect::<Vec<_>>();

    json!({
        "op": op,
        "args": arguments,
        "funcs": functions,
        "labels": labels
    })
}

pub fn instruction_to_json(instruction: &ast::Instruction) -> Value {
    match instruction {
        ast::Instruction::Constant(constant) => constant_to_json(constant),
        ast::Instruction::ValueOperation(value_operation) => {
            value_operation_to_json(value_operation)
        }
        ast::Instruction::EffectOperation(effect_operation) => {
            effect_operation_to_json(effect_operation)
        }
    }
}

pub fn function_code_to_json(code: &ast::FunctionCode) -> Value {
    match code {
        ast::FunctionCode::Label { label, .. } => label_to_json(label),
        ast::FunctionCode::Instruction(instruction) => instruction_to_json(instruction),
    }
}

pub fn function_to_json(function: &ast::Function) -> Value {
    let parameters = function
        .parameters
        .iter()
        .map(|(name, type_annotation)| {
            json!({
                "name": **name,
                "type":type_to_json(&type_annotation.ty)
            })
        })
        .collect::<Vec<_>>();

    let return_type = function
        .return_type
        .as_ref()
        .map(|type_annotation| type_to_json(&type_annotation.ty));

    let body = function.body.iter().loc_map_to_vec(function_code_to_json);

    json!({
        "name": extract_function_name(&function.name),
        "args": parameters,
        "type": return_type,
        "instrs": body
    })
}

pub fn program_to_json(program: &ast::Program) -> Value {
    let imports = program.imports.iter().loc_map_to_vec(import_to_json);
    let functions = program.functions.iter().loc_map_to_vec(function_to_json);
    json!({
        "imports": imports,
        "functions": functions
    })
}
