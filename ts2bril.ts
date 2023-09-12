import * as ts from 'https://esm.sh/typescript@5.0.4';
import * as bril from './bril-ts/bril.ts';
import {Builder} from './bril-ts/builder.ts';

const opTokens = new Map<ts.SyntaxKind, [bril.ValueOpCode, bril.Type]>([
  [ts.SyntaxKind.PlusToken,               ["add", "int"]],
  [ts.SyntaxKind.AsteriskToken,           ["mul", "int"]],
  [ts.SyntaxKind.MinusToken,              ["sub", "int"]],
  [ts.SyntaxKind.SlashToken,              ["div", "int"]],
  [ts.SyntaxKind.LessThanToken,           ["lt",  "bool"]],
  [ts.SyntaxKind.LessThanEqualsToken,     ["le",  "bool"]],
  [ts.SyntaxKind.GreaterThanToken,        ["gt",  "bool"]],
  [ts.SyntaxKind.GreaterThanEqualsToken,  ["ge",  "bool"]],
  [ts.SyntaxKind.EqualsEqualsToken,       ["eq",  "bool"]],
  [ts.SyntaxKind.EqualsEqualsEqualsToken, ["eq",  "bool"]],
]);

const opTokensFloat = new Map<ts.SyntaxKind, [bril.ValueOpCode, bril.Type]>([
  [ts.SyntaxKind.PlusToken,               ["fadd", "float"]],
  [ts.SyntaxKind.AsteriskToken,           ["fmul", "float"]],
  [ts.SyntaxKind.MinusToken,              ["fsub", "float"]],
  [ts.SyntaxKind.SlashToken,              ["fdiv", "float"]],
  [ts.SyntaxKind.LessThanToken,           ["flt",  "bool"]],
  [ts.SyntaxKind.LessThanEqualsToken,     ["fle",  "bool"]],
  [ts.SyntaxKind.GreaterThanToken,        ["fgt",  "bool"]],
  [ts.SyntaxKind.GreaterThanEqualsToken,  ["fge",  "bool"]],
  [ts.SyntaxKind.EqualsEqualsToken,       ["feq",  "bool"]],
  [ts.SyntaxKind.EqualsEqualsEqualsToken, ["feq",  "bool"]],
]);

function isTypeReference(ty: ts.Type): ty is ts.TypeReference {
  return 'typeArguments' in ty;
}

function tsTypeToBril(tsType: ts.Type, checker: ts.TypeChecker): bril.Type {
  if (tsType.flags & (ts.TypeFlags.Number | ts.TypeFlags.NumberLiteral)) {
    return "float";
  } else if (tsType.flags &
             (ts.TypeFlags.Boolean | ts.TypeFlags.BooleanLiteral)) {
    return "bool";
  } else if (tsType.flags &
             (ts.TypeFlags.BigInt | ts.TypeFlags.BigIntLiteral)) {
    return "int";
  } else if (isTypeReference(tsType) && tsType.symbol && tsType.symbol.name === "Pointer") {
    const params = checker.getTypeArguments(tsType);
    return { ptr: tsTypeToBril(params[0], checker) };
  } else {
    throw "unimplemented type " + checker.typeToString(tsType);
  }
}

function brilType(node: ts.Node, checker: ts.TypeChecker): bril.Type {
  let tsType = checker.getTypeAtLocation(node);
  return tsTypeToBril(tsType, checker);
}

/**
 * Compile a complete TypeScript AST to a Bril program.
 */
function emitBril(prog: ts.Node, checker: ts.TypeChecker): bril.Program {
  let builder = new Builder();
  let mainFn = builder.buildFunction("main", []);  // Main has no return type.

  function emitExpr(expr: ts.Expression): bril.ValueInstruction {
    switch (expr.kind) {
    case ts.SyntaxKind.NumericLiteral: {
      let lit = expr as ts.NumericLiteral;
      let val = parseFloat(lit.text);
      return builder.buildFloat(val);
    }

    case ts.SyntaxKind.BigIntLiteral: {
      let lit = expr as ts.BigIntLiteral;
      let val = parseInt(lit.text);
      return builder.buildInt(val);
    }

    case ts.SyntaxKind.TrueKeyword: {
      return builder.buildBool(true);
    }

    case ts.SyntaxKind.FalseKeyword: {
      return builder.buildBool(false);
    }

    case ts.SyntaxKind.Identifier: {
      let ident = expr as ts.Identifier;
      let type = brilType(ident, checker);
      return builder.buildValue("id", type, [ident.text]);
    }

    case ts.SyntaxKind.BinaryExpression:
      let bin = expr as ts.BinaryExpression;
      let kind = bin.operatorToken.kind;

      // Handle assignments.
      switch (kind) {
      case ts.SyntaxKind.EqualsToken:
        if (!ts.isIdentifier(bin.left)) {
          throw "assignment to non-variables unsupported";
        }
        let dest = bin.left as ts.Identifier;
        let rhs = emitExpr(bin.right);
        let type = brilType(dest, checker);
        return builder.buildValue("id", type, [rhs.dest], undefined, undefined, dest.text);
      }

      // Handle "normal" value operators.
      let op: bril.ValueOpCode;
      let type: bril.Type;
      if (brilType(bin.left, checker) === "float" ||
          brilType(bin.right, checker) === "float") {
        // Floating point operators.
        let p = opTokensFloat.get(kind);
        if (!p) {
          throw `unhandled FP binary operator kind ${kind}`;
        }
        [op, type] = p;
      } else {
        // Non-float.
        let p = opTokens.get(kind);
        if (!p) {
          throw `unhandled binary operator kind ${kind}`;
        }
        [op, type] = p;
      }

      let lhs = emitExpr(bin.left);
      let rhs = emitExpr(bin.right);
      return builder.buildValue(op, type, [lhs.dest, rhs.dest]);

    // Support call instructions---but only for printing, for now.
    case ts.SyntaxKind.CallExpression:
      let call = expr as ts.CallExpression;
      let callText = call.expression.getText();
      if (callText === "console.log") {
        let values = call.arguments.map(emitExpr);
        builder.buildEffect("print", values.map(v => v.dest));
        return builder.buildInt(0);  // Expressions must produce values.
      } else if (memoryBuiltins[callText]) {
        return memoryBuiltins[callText](call);
      } else {
        // Recursively translate arguments.
        let values = call.arguments.map(emitExpr);

        // Check if effect statement, i.e., a call that is not a subexpression
        if (call.parent.kind === ts.SyntaxKind.ExpressionStatement) {
          builder.buildCall(callText,
            values.map(v => v.dest));
          return builder.buildInt(0);  // Expressions must produce values
        } else {
          let decl = call.parent as ts.VariableDeclaration;
          let type = brilType(decl, checker);
          let name = (decl.name != undefined) ? decl.name.getText() : undefined;
          return builder.buildCall(
            callText,
            values.map(v => v.dest),
            type,
            name,
          );
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
        let decl = node as ts.VariableDeclaration;
        // Declarations without initializers are no-ops.
        if (decl.initializer) {
          let init = emitExpr(decl.initializer);
          let type = brilType(decl, checker);
          builder.buildValue("id", type, [init.dest], undefined, undefined, decl.name.getText());
        }
        break;
      }

      // Expressions by themselves.
      case ts.SyntaxKind.ExpressionStatement: {
        let exstmt = node as ts.ExpressionStatement;
        emitExpr(exstmt.expression);  // Ignore the result.
        break;
      }

      // Conditionals.
      case ts.SyntaxKind.IfStatement: {
        let if_ = node as ts.IfStatement;

        // Label names.
        let sfx = builder.freshSuffix();
        let thenLab = "then" + sfx;
        let elseLab = "else" + sfx;
        let endLab = "endif" + sfx;

        // Branch.
        let cond = emitExpr(if_.expression);
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
        let for_ = node as ts.ForStatement;

        // Label names.
        let sfx = builder.freshSuffix();
        let condLab = "for.cond" + sfx;
        let bodyLab = "for.body" + sfx;
        let endLab = "for.end" + sfx;

        // Initialization.
        if (for_.initializer) {
          emit(for_.initializer);
        }

        // Condition check.
        builder.buildLabel(condLab);
        if (for_.condition) {
          let cond = emitExpr(for_.condition);
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

      case ts.SyntaxKind.FunctionDeclaration:
        let funcDef = node as ts.FunctionDeclaration;
        if (funcDef.name === undefined) {
          throw `no anonymous functions!`;
        }
        let name: string = funcDef.name.getText();
        let args: bril.Argument[] = [];

        for (let p of funcDef.parameters) {
          let argName = p.name.getText();
          let argType = brilType(p, checker);
          args.push({name: argName, type: argType} as bril.Argument);
        }

        // The type checker gives a full function type;
        // we want only the return type.
        if (funcDef.type && funcDef.type.getText() !== 'void') {
          builder.buildFunction(name, args, brilType(funcDef.type, checker));
        } else {
          builder.buildFunction(name, args);
        }
        if (funcDef.body) {
          emit(funcDef.body);
        }
        builder.setCurrentFunction(mainFn);
        break;

      case ts.SyntaxKind.ReturnStatement: {
        let retstmt = node as ts.ReturnStatement;
        if (retstmt.expression) {
          let val = emitExpr(retstmt.expression);
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

  const memoryBuiltins: Record<string, (call: ts.CallExpression) => bril.ValueInstruction> = {
    "mem.alloc": (call) => {
      let type = brilType(call, checker);
      let values = call.arguments.map(emitExpr);
      return builder.buildValue("alloc", type, values.map(v => v.dest));
    },

    "mem.store": (call) => {
      let values = call.arguments.map(emitExpr);
      builder.buildEffect("store", values.map(v => v.dest));
      return builder.buildInt(0);  // Expressions must produce values.
    },

    "mem.load": (call) => {
      let type = brilType(call, checker);
      let values = call.arguments.map(emitExpr);
      return builder.buildValue("load", type, values.map(v => v.dest));
    },

    "mem.free": (call) => {
      let values = call.arguments.map(emitExpr);
      builder.buildEffect("free", values.map(v => v.dest));
      return builder.buildInt(0);
    },

    "mem.ptradd": (call) => {
      let type = brilType(call, checker);
      let values = call.arguments.map(emitExpr);
      return builder.buildValue("ptradd", type, values.map(v => v.dest));
    }
  }

  emit(prog);
  return builder.program;
}

async function main() {
  // Get the TypeScript filename.
  let filename = Deno.args[0];
  if (!filename) {
    console.error(`usage: ts2bril src.ts`)
    Deno.exit(1);
  }

  // Load up the TypeScript context.
  let program = ts.createProgram([filename], {
    target: ts.ScriptTarget.ES5,
  });
  let checker = program.getTypeChecker();

  // Do a weird dance to look up our source file.
  let sf: ts.SourceFile | undefined;
  for (let file of program.getSourceFiles()) {
    if (file.fileName === filename) {
      sf = file;
      break;
    }
  }
  if (!sf) {
    throw "source file not found";
  }

  // Generate Bril code.
  let brilProg = emitBril(sf, checker);
  let json = JSON.stringify(brilProg, undefined, 2);
  await Deno.stdout.write(
    new TextEncoder().encode(json)
  );
}

main();
