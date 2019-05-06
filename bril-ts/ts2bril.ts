import * as ts from 'typescript';
import * as bril from './bril';
import {Builder} from './builder';
import {readStdin} from './util';

const tokenToOp = new Map<ts.SyntaxKind, bril.OpCode>([
  [ts.SyntaxKind.PlusToken,               bril.OpCode.add],
  [ts.SyntaxKind.LessThanToken,           bril.OpCode.lt],
  [ts.SyntaxKind.LessThanEqualsToken,     bril.OpCode.le],
  [ts.SyntaxKind.GreaterThanToken,        bril.OpCode.lt],
  [ts.SyntaxKind.GreaterThanEqualsToken,  bril.OpCode.le],
  [ts.SyntaxKind.EqualsEqualsToken,       bril.OpCode.eq],
  [ts.SyntaxKind.EqualsEqualsEqualsToken, bril.OpCode.eq],
]);

/**
 * Compile a complete TypeScript AST to a Bril program.
 */
function emitBril(prog: ts.Node): bril.Program {
  let builder = new Builder();
  builder.buildFunction("main");

  function emitExpr(expr: ts.Expression): bril.Instruction {
    switch (expr.kind) {
    case ts.SyntaxKind.NumericLiteral: {
      let lit = expr as ts.NumericLiteral;
      let val = parseInt(lit.getText());
      return builder.buildConst(val);
    }

    case ts.SyntaxKind.TrueKeyword: {
      return builder.buildConst(true);
    }

    case ts.SyntaxKind.FalseKeyword: {
      return builder.buildConst(false);
    }

    case ts.SyntaxKind.Identifier:
      let ident = expr as ts.Identifier;
      return builder.buildOp(bril.OpCode.id, [ident.getText()]);

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
        return builder.buildOp(bril.OpCode.id, [rhs.dest], dest.getText());
      }

      // Handle "normal" value operators.
      let lhs = emitExpr(bin.left);
      let rhs = emitExpr(bin.right);
      let op = tokenToOp.get(kind);
      if (!op) {
        throw `unhandled binary operator kind ${kind}`;
      }
      return builder.buildOp(op, [lhs.dest, rhs.dest]);

    // Support call instructions---but only for printing, for now.
    case ts.SyntaxKind.CallExpression:
      let call = expr as ts.CallExpression;
      if (call.expression.getText() === "console.log") {
        let values = call.arguments.map(emitExpr);
        return builder.buildOp(bril.OpCode.print, values.map(v => v.dest));
      } else {
        throw "function calls unsupported";
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
          builder.buildOp(bril.OpCode.id, [init.dest], decl.name.getText());
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

        // Branch.
        let cond = emitExpr(if_.expression);
        builder.buildOp(bril.OpCode.br, [cond.dest, "then", "else"]);

        // Statement chunks.
        builder.buildLabel("then");  // TODO unique name
        emit(if_.thenStatement);
        builder.buildOp(bril.OpCode.jmp, ["endif"]);
        builder.buildLabel("else");
        if (if_.elseStatement) {
          emit(if_.elseStatement);
        }
        builder.buildLabel("endif");

        break;
      }

      default:
        throw `unhandled TypeScript AST node kind ${node.kind}`;
    }
  }

  emit(prog);
  return builder.program;
}

async function main() {
  let sf = ts.createSourceFile(
    '-',
    await readStdin(),
    ts.ScriptTarget.ES2015,
    true,
  );
  let prog = emitBril(sf);
  process.stdout.write(
    JSON.stringify(prog, undefined, 2)
  );
}

main();
