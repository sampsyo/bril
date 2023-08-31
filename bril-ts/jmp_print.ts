import * as bril from "./bril.ts";
import { readStdin } from "./util.ts";

const program = JSON.parse(await readStdin()) as bril.Program;
for (const func of program.functions) {
  for (let i = 0; i < func.instrs.length; i++) {
    const instr = func.instrs[i];
    if ("op" in instr) {
      if (instr.op == "jmp") {
        func.instrs.splice(i++, 0, {
          op: "const",
          dest: "tmp",
          value: 69,
          type: "int",
        });
        func.instrs.splice(i++, 0, { op: "print", args: ["tmp"] });
      }
    }
  }
}

const json = JSON.stringify(program, undefined, 2);
await Deno.stdout.write(new TextEncoder().encode(json));
