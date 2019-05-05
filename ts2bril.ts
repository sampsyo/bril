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

function emitBril(node: ts.Node): bril.Program {
  let builder = new Builder();
  builder.buildFunction("main");

  function emit(node: ts.Node) {
    switch (node.kind) {
      case ts.SyntaxKind.IfStatement:
        //console.log(node);
        break;
      case ts.SyntaxKind.VariableDeclaration:
        let decl = node as ts.VariableDeclaration;
        builder.buildInstr(bril.Operation.id, ["foo"], decl.name.getText());
        break;
      case ts.SyntaxKind.BinaryExpression:
        //console.log(node);
        break;
    }
    ts.forEachChild(node, emit);
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
