import * as ts from "typescript";
import * as fs from "fs";
import * as util from "util";

const readFile = util.promisify(fs.readFile);

function bril(node: ts.Node) {
  switch (node.kind) {
    case ts.SyntaxKind.IfStatement:
      console.log(node);
      break;
  }
  ts.forEachChild(node, bril);
}

async function main() {
  const filenames = process.argv.slice(2);
  for (let filename of filenames) {
    let sf = ts.createSourceFile(
      filename,
      (await readFile(filename)).toString(),
      ts.ScriptTarget.ES2015,
      true,
    );
    bril(sf);
  }
}

main();
