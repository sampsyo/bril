import * as ts from 'typescript';
import * as bril from './bril';

class Builder {
  public program: bril.Program = { functions: [] };

  private curFunction: bril.Function | null = null;

  buildFunction(name: string) {
    let func: bril.Function = { name, instrs: [] };
    this.program.functions.push(func);
    this.curFunction = func;
    return func;
  }

  buildInstr(op: bril.Operation, args: string[], dest: string) {
    let instr: bril.Instruction = { op, args, dest };
    if (!this.curFunction) {
      throw "cannot build instruction without a function";
    }
    this.curFunction.instrs.push(instr);
    return instr;
  }
}

/**
 * Compile a complete TypeScript AST to a Bril program.
 */
function emitBril(node: ts.Node): bril.Program {
  let builder = new Builder();
  builder.buildFunction("main");

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
      case ts.SyntaxKind.VariableDeclaration:
        let decl = node as ts.VariableDeclaration;
        builder.buildInstr(bril.Operation.id, ["foo"], decl.name.getText());
        break;
      
      // Operations.
      case ts.SyntaxKind.BinaryExpression:
        console.log(node);
        break;
      
      default:
        console.error('unhandled TypeScript AST node', node.kind);
        break;
    }
  }

  emit(node);
  return builder.program;
}

/**
 * Read all the data from stdin as a string.
 */
function readStdin(): Promise<string> {
  return new Promise((resolve, reject) => {
    let chunks: string[] = [];
    process.stdin.on("data", function (chunk: string) {
      chunks.push(chunk);
    }).on("end", function () {
      resolve(chunks.join(""))
    }).setEncoding("utf8");
  });
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
