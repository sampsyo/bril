import * as ts from "https://esm.sh/typescript@5.7.3";
import * as bril from "./bril-ts/bril.ts";
import { Builder } from "./bril-ts/builder.ts";

// deno-fmt-ignore
const opTokens = new Map<ts.SyntaxKind, [bril.ValueOpCode, bril.Type]>([
  [ts.SyntaxKind.PlusToken,               ["add", "int"]],
  [ts.SyntaxKind.AsteriskToken,           ["mul", "int"]],
  [ts.SyntaxKind.MinusToken,              ["sub", "int"]],
  [ts.SyntaxKind.SlashToken,              ["div", "int"]],
  [ts.SyntaxKind.LessThanToken,           ["lt", "bool"]],
  [ts.SyntaxKind.LessThanEqualsToken,     ["le", "bool"]],
  [ts.SyntaxKind.GreaterThanToken,        ["gt", "bool"]],
  [ts.SyntaxKind.GreaterThanEqualsToken,  ["ge", "bool"]],
  [ts.SyntaxKind.EqualsEqualsToken,       ["eq", "bool"]],
  [ts.SyntaxKind.EqualsEqualsEqualsToken, ["eq", "bool"]],
]);

// deno-fmt-ignore
const opTokensFloat = new Map<ts.SyntaxKind, [bril.ValueOpCode, bril.Type]>([
  [ts.SyntaxKind.PlusToken,               ["fadd", "float"]],
  [ts.SyntaxKind.AsteriskToken,           ["fmul", "float"]],
  [ts.SyntaxKind.MinusToken,              ["fsub", "float"]],
  [ts.SyntaxKind.SlashToken,              ["fdiv", "float"]],
  [ts.SyntaxKind.LessThanToken,           ["flt", "bool"]],
  [ts.SyntaxKind.LessThanEqualsToken,     ["fle", "bool"]],
  [ts.SyntaxKind.GreaterThanToken,        ["fgt", "bool"]],
  [ts.SyntaxKind.GreaterThanEqualsToken,  ["fge", "bool"]],
  [ts.SyntaxKind.EqualsEqualsToken,       ["feq", "bool"]],
  [ts.SyntaxKind.EqualsEqualsEqualsToken, ["feq", "bool"]],
]);

function isTypeReference(ty: ts.Type): ty is ts.TypeReference {
  return "typeArguments" in ty;
}

function tsTypeToBril(tsType: ts.Type, checker: ts.TypeChecker): bril.Type {
  if (tsType.flags & (ts.TypeFlags.Number | ts.TypeFlags.NumberLiteral)) {
    return "float";
  } else if (
    tsType.flags &
    (ts.TypeFlags.Boolean | ts.TypeFlags.BooleanLiteral)
  ) {
    return "bool";
  } else if (
    tsType.flags &
    (ts.TypeFlags.BigInt | ts.TypeFlags.BigIntLiteral)
  ) {
    return "int";
  } else if (
    isTypeReference(tsType) && tsType.symbol && tsType.symbol.name === "Pointer"
  ) {
    const params = checker.getTypeArguments(tsType);
    return { ptr: tsTypeToBril(params[0], checker) };
  } else {
    throw "unimplemented type " + checker.typeToString(tsType);
  }
}

function brilType(node: ts.Node, checker: ts.TypeChecker): bril.Type {
  const tsType = checker.getTypeAtLocation(node);
  return tsTypeToBril(tsType, checker);
}

/**
 * Compile a complete TypeScript AST to a Bril program.
 */
function emitBril(prog: ts.Node, checker: ts.TypeChecker): bril.Program {
  const builder = new Builder();
  const mainFn = builder.buildFunction("main", []); // Main has no return type.

  function emitExpr(expr: ts.Expression): bril.ValueInstruction {
    switch (expr.kind) {
      case ts.SyntaxKind.NumericLiteral: {
        const lit = expr as ts.NumericLiteral;
        const val = parseFloat(lit.text);
        return builder.buildFloat(val);
      }

      case ts.SyntaxKind.BigIntLiteral: {
        const lit = expr as ts.BigIntLiteral;
        const val = parseInt(lit.text);
        return builder.buildInt(val);
      }

      case ts.SyntaxKind.TrueKeyword: {
        return builder.buildBool(true);
      }

      case ts.SyntaxKind.FalseKeyword: {
        return builder.buildBool(false);
      }

      case ts.SyntaxKind.Identifier: {
        const ident = expr as ts.Identifier;
        const type = brilType(ident, checker);
        return builder.buildValue("id", type, [ident.text]);
      }

      case ts.SyntaxKind.BinaryExpression: {
        const bin = expr as ts.BinaryExpression;
        const kind = bin.operatorToken.kind;

        // Handle assignments.
        switch (kind) {
          case ts.SyntaxKind.EqualsToken: {
            if (!ts.isIdentifier(bin.left)) {
              throw "assignment to non-variables unsupported";
            }
            const dest = bin.left as ts.Identifier;
            const rhs = emitExpr(bin.right);
            const type = brilType(dest, checker);
            return builder.buildValue(
              "id",
              type,
              [rhs.dest],
              undefined,
              undefined,
              dest.text,
            );
          }
        }

        // Handle "normal" value operators.
        let op: bril.ValueOpCode;
        let type: bril.Type;
        if (
          brilType(bin.left, checker) === "float" ||
          brilType(bin.right, checker) === "float"
        ) {
          // Floating point operators.
          const p = opTokensFloat.get(kind);
          if (!p) {
            throw `unhandled FP binary operator kind ${kind}`;
          }
          [op, type] = p;
        } else {
          // Non-float.
          const p = opTokens.get(kind);
          if (!p) {
            throw `unhandled binary operator kind ${kind}`;
          }
          [op, type] = p;
        }

        const lhs = emitExpr(bin.left);
        const rhs = emitExpr(bin.right);
        return builder.buildValue(op, type, [lhs.dest, rhs.dest]);
      }

      // Support call instructions---but only for printing, for now.
      case ts.SyntaxKind.CallExpression: {
        const call = expr as ts.CallExpression;
        const callText = call.expression.getText();
        if (callText === "console.log") {
          const values: bril.ValueInstruction[] = call.arguments.map(emitExpr);
          builder.buildEffect("print", values.map((v) => v.dest));
          return builder.buildInt(0); // Expressions must produce values.
        } else if (memoryBuiltins[callText]) {
          return memoryBuiltins[callText](call);
        } else {
          // Recursively translate arguments.
          const values: bril.ValueInstruction[] = call.arguments.map(emitExpr);

          // Check if effect statement, i.e., a call that is not a subexpression
          if (call.parent.kind === ts.SyntaxKind.ExpressionStatement) {
            builder.buildCall(callText, values.map((v) => v.dest));
            return builder.buildInt(0); // Expressions must produce values
          } else {
            const decl = call.parent as ts.VariableDeclaration;
            const type = brilType(decl, checker);
            const name = (decl.name != undefined)
              ? decl.name.getText()
              : undefined;
            return builder.buildCall(
              callText,
              values.map((v) => v.dest),
              type,
              name,
            );
          }
        }
      }
      default:
        throw `unsupported expression kind: ${expr.getText()}`;
    }
  }

  function emit(node: ts.Node) {
    switch (node.kind) {
      // Descend through containers.
      case ts.SyntaxKind.SourceFile:
      case ts.SyntaxKind.Block:
      case ts.SyntaxKind.VariableStatement:
      case ts.SyntaxKind.VariableDeclarationList:
        ts.forEachChild(node, emit);
        break;

      // No-op.
      case ts.SyntaxKind.EndOfFileToken:
        break;

      // Emit declarations.
      case ts.SyntaxKind.VariableDeclaration: {
        const decl = node as ts.VariableDeclaration;
        // Declarations without initializers are no-ops.
        if (decl.initializer) {
          const init = emitExpr(decl.initializer);
          const type = brilType(decl, checker);
          builder.buildValue(
            "id",
            type,
            [init.dest],
            undefined,
            undefined,
            decl.name.getText(),
          );
        }
        break;
      }

      // Expressions by themselves.
      case ts.SyntaxKind.ExpressionStatement: {
        const exstmt = node as ts.ExpressionStatement;
        emitExpr(exstmt.expression); // Ignore the result.
        break;
      }

      // Conditionals.
      case ts.SyntaxKind.IfStatement: {
        const if_ = node as ts.IfStatement;

        // Label names.
        const sfx = builder.freshSuffix();
        const thenLab = "then" + sfx;
        const elseLab = "else" + sfx;
        const endLab = "endif" + sfx;

        // Branch.
        const cond = emitExpr(if_.expression);
        builder.buildEffect("br", [cond.dest], undefined, [thenLab, elseLab]);

        // Statement chunks.
        builder.buildLabel(thenLab);
        emit(if_.thenStatement);
        const then_branch_terminated = builder.getLastInstr()?.op === "ret";
        if (!then_branch_terminated) {
          builder.buildEffect("jmp", [], undefined, [endLab]);
        }
        builder.buildLabel(elseLab);
        if (if_.elseStatement) {
          emit(if_.elseStatement);
        }
        // The else branch otherwise just falls through without needing a target label
        if (!then_branch_terminated) {
          builder.buildLabel(endLab);
        }

        break;
      }

      // Plain "for" loops.
      case ts.SyntaxKind.ForStatement: {
        const for_ = node as ts.ForStatement;

        // Label names.
        const sfx = builder.freshSuffix();
        const condLab = "for.cond" + sfx;
        const bodyLab = "for.body" + sfx;
        const endLab = "for.end" + sfx;

        // Initialization.
        if (for_.initializer) {
          emit(for_.initializer);
        }

        // Condition check.
        builder.buildLabel(condLab);
        if (for_.condition) {
          const cond = emitExpr(for_.condition);
          builder.buildEffect("br", [cond.dest], undefined, [bodyLab, endLab]);
        }

        builder.buildLabel(bodyLab);
        emit(for_.statement);
        if (for_.incrementor) {
          emitExpr(for_.incrementor);
        }
        builder.buildEffect("jmp", [], undefined, [condLab]);
        builder.buildLabel(endLab);

        break;
      }

      case ts.SyntaxKind.FunctionDeclaration: {
        const funcDef = node as ts.FunctionDeclaration;
        if (funcDef.name === undefined) {
          throw `no anonymous functions!`;
        }
        const name: string = funcDef.name.getText();
        const args: bril.Argument[] = [];

        for (const p of funcDef.parameters) {
          const argName = p.name.getText();
          const argType = brilType(p, checker);
          args.push({ name: argName, type: argType } as bril.Argument);
        }

        // The type checker gives a full function type;
        // we want only the return type.
        if (funcDef.type && funcDef.type.getText() !== "void") {
          builder.buildFunction(name, args, brilType(funcDef.type, checker));
        } else {
          builder.buildFunction(name, args);
        }
        if (funcDef.body) {
          emit(funcDef.body);
        }
        builder.setCurrentFunction(mainFn);
        break;
      }

      case ts.SyntaxKind.ReturnStatement: {
        const retstmt = node as ts.ReturnStatement;
        if (retstmt.expression) {
          const val = emitExpr(retstmt.expression);
          builder.buildEffect("ret", [val.dest]);
        } else {
          builder.buildEffect("ret", []);
        }
        break;
      }

      case ts.SyntaxKind.ImportDeclaration:
        break;

      default:
        throw `unhandled TypeScript AST node kind ${ts.SyntaxKind[node.kind]}`;
    }
  }

  const memoryBuiltins: Record<
    string,
    (call: ts.CallExpression) => bril.ValueInstruction
  > = {
    "mem.alloc": (call) => {
      const type = brilType(call, checker);
      const values: bril.ValueInstruction[] = call.arguments.map(emitExpr);
      return builder.buildValue("alloc", type, values.map((v) => v.dest));
    },

    "mem.store": (call) => {
      const values: bril.ValueInstruction[] = call.arguments.map(emitExpr);
      builder.buildEffect("store", values.map((v) => v.dest));
      return builder.buildInt(0); // Expressions must produce values.
    },

    "mem.load": (call) => {
      const type = brilType(call, checker);
      const values: bril.ValueInstruction[] = call.arguments.map(emitExpr);
      return builder.buildValue("load", type, values.map((v) => v.dest));
    },

    "mem.free": (call) => {
      const values: bril.ValueInstruction[] = call.arguments.map(emitExpr);
      builder.buildEffect("free", values.map((v) => v.dest));
      return builder.buildInt(0);
    },

    "mem.ptradd": (call) => {
      const type = brilType(call, checker);
      const values: bril.ValueInstruction[] = call.arguments.map(emitExpr);
      return builder.buildValue("ptradd", type, values.map((v) => v.dest));
    },
  };

  emit(prog);
  return builder.program;
}

async function main() {
  // Get the TypeScript filename.
  const filename = Deno.args[0];
  if (!filename) {
    console.error(`usage: ts2bril src.ts`);
    Deno.exit(1);
  }

  // Load up the TypeScript context.
  const program = ts.createProgram([filename], {
    target: ts.ScriptTarget.ES5,
  });
  const checker = program.getTypeChecker();

  // Do a weird dance to look up our source file.
  let sf: ts.SourceFile | undefined;
  for (const file of program.getSourceFiles()) {
    if (file.fileName === filename) {
      sf = file;
      break;
    }
  }
  if (!sf) {
    throw "source file not found";
  }

  // Generate Bril code.
  const brilProg = emitBril(sf, checker);
  const json = JSON.stringify(brilProg, undefined, 2);
  await Deno.stdout.write(
    new TextEncoder().encode(json),
  );
}

main();
