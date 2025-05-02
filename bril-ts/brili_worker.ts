// Lightweight runtime that a web worker runs to execute a spawned thread
// It imports the og interpreter as a module

import {
  runBrilFunction,
} from "../brili.ts";
import * as bril from "./bril.ts";
import { Heap } from "../brili.ts";

// message protocol -----------------------------------------------------------------------
// Parent => worker: {kind: "start", prog: Program, func: string, args: any[], tid: number}
// Worked => parent: {kind: "done", tid: number}

self.onmessage = async (ev: MessageEvent) => {
  const msg = ev.data;
  if (msg?.kind !== "start") return;

  const { prog, func, args, tid } = msg as {
    prog: bril.Program;
    func: string;
    args: any[];
    tid: number;
  };

  // build an Env from formal parameters & args
  const callee = prog.functions.find(f => f.name === func)!;
  const env = new Map<bril.Ident, any>();
  (callee.args ?? []).forEach((p, i) => env.set(p.name, args[i]));

  // NOTE to self --> every worker gets its own private heap rn!!
  const heap = new Heap<any>();

  try {
    await runBrilFunction(prog, func, env, heap);
    // sanity check: all mem freed??
    if (!heap.isEmpty()) {
      const msg = `worker tid=${tid} exited with unfreed heap blocks`;
      self.postMessage({ kind: "error", tid, message: msg });
      // still throw error so worker's own console gets stack
      throw new Error(msg);
    }
    // success --> notify parent
    self.postMessage({ kind: "done", tid });
  } catch (e) {
    // bubble the error up to parent
    self.postMessage({ kind: "error", tid, message: String(e) });
    throw e;
  }
};