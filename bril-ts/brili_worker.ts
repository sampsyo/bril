// bril-ts/brili_worker.ts
import { runBrilFunction, Key } from "../brili.ts";
import * as bril from "./bril.ts";

// The main thread fills this when it sends the start message
let tid = -1;

// Proxy for shared heap RPCs
class HeapProxy<X> {
  private next = 1;
  private wait = new Map<number, (v: any) => void>();

  private req(op: string, pay: Record<string, unknown>): Promise<any> {
    const id = this.next++;
    return new Promise((resolve) => {
      this.wait.set(id, resolve);
      // send request back to main thread
      (self as DedicatedWorkerGlobalScope).postMessage({
        kind: "heap_req",
        id,
        op,
        workerTid: tid,
        ...pay,
      });
    });
  }

  alloc(n: number) { return this.req("alloc", { amt: n }); }
  read(k: Key) { return this.req("load", { base: k.base, offset: k.offset }); }
  write(k: Key, v: X) { return this.req("store", { base: k.base, offset: k.offset, value: v }); }
  free(k: Key) { return this.req("free", { base: k.base, offset: k.offset }); }

  handle(msg: any) {
    if (msg.kind === "heap_res" || msg.kind === "heap_err") {
      const fn = this.wait.get(msg.id);
      if (fn) {
        fn(msg.value);
        this.wait.delete(msg.id);
      }
    }
  }
}
const sharedHeap = new HeapProxy<any>();

// Message handler
self.onmessage = async (ev: MessageEvent) => {
  const msg = ev.data as any;
  // First, let the heap proxy munch any heap_res/heap_err
  sharedHeap.handle(msg);

  if (msg.kind !== "start") return;

  tid = msg.tid;
  const prog = msg.prog as bril.Program;
  const func = msg.func as string;
  const rawArgs = msg.args as any[];

  // Build callee env, rehydrating Pointer arguments into Key-based Pointers
  const callee = prog.functions.find((f) => f.name === func)!;
  const env = new Map<bril.Ident, any>();
  (callee.args || []).forEach((param, i) => {
    let v = rawArgs[i];
    // If the static type is a ptr<...>, re-wrap it
    if (typeof param.type === "object" && Object.hasOwn(param.type, "ptr")) {
      // v is { loc: { base: number, offset: number }, type: <Type> }
      const raw = v as { loc: { base: number; offset: number } };
      v = {
        loc: new Key(raw.loc.base, raw.loc.offset),
        type: param.type.ptr,
      };
    }
    env.set(param.name, v);
  });

  try {
    await runBrilFunction(prog, func, env, sharedHeap as any);
    (self as DedicatedWorkerGlobalScope).postMessage({ kind: "done", tid });
  } catch (e) {
    (self as DedicatedWorkerGlobalScope).postMessage({
      kind: "error",
      tid,
      message: String(e),
    });
    throw e;
  }
};