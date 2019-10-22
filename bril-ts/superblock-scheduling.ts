#!/usr/bin/env node
import * as bril from './bril';
import * as df from './dataflow';
import { readStdin } from './util';

function simple(prog: bril.Program): Array<bril.Instruction> {
  for (let func of prog.functions) {
    if (func.name === "main") {
      let res: bril.Instruction[] = [];
      for (let ins of func.instrs) {
        if ("instrs" in ins || "op" in ins) {
          res.push(ins);
        }
      }
      return res;
    }
  }
  return [];
}

async function main() {
  let prog = JSON.parse(await readStdin()) as bril.Program;
  let trace = simple(prog);
  console.dir(df.dataflow(trace), { depth: 3 });
}

// Make unhandled promise rejections terminate.
process.on('unhandledRejection', e => { throw e });

main();
