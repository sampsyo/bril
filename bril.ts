import * as ts from "typescript";
import * as fs from "fs";

function bril(node: ts.Node) {
  switch (node.kind) {
    case ts.SyntaxKind.IfStatement:
      console.log(node);
      break;
  }
  ts.forEachChild(node, bril);
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
  bril(sf);
}

main();
