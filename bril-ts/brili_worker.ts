/* Lightweight runtime executed inside each web-worker */

/// <reference lib="webworker" />

import { runBrilFunction, Key } from "../brili.ts";
import * as bril from "./bril.ts";

/* The main thread fills this when it sends the start message */
let tid = -1;

/* ───────── Heap proxy class ───────── */

class HeapProxy<X> {
  private next = 1;
  private wait = new Map<number, (v: any) => void>();

  private req(op: string, pay: Record<string, unknown>): Promise<any> {
    const id = this.next++;
    return new Promise((resolve) => {
      this.wait.set(id, resolve);
      (self as DedicatedWorkerGlobalScope).postMessage({ kind: "heap_req", id, op, workerTid: tid, ...pay });
    });
  }
  alloc(n: number) { return this.req("alloc", { amt: n }); }
  read(k: Key) { return this.req("load", { base: k.base, offset: k.offset }); }
  write(k: Key, v: X) { return this.req("store", { base: k.base, offset: k.offset, value: v }); }
  free(k: Key) { return this.req("free", { base: k.base, offset: k.offset }); }

  handle(msg: any) {
    if (msg?.kind === "heap_res" || msg?.kind === "heap_err") {
      const fn = this.wait.get(msg.id); if (fn) { fn(msg.value); this.wait.delete(msg.id); }
    }
  }
}
const sharedHeap = new HeapProxy<any>();

/* ───────── message handler ───────── */

self.onmessage = async (ev: MessageEvent) => {
  const msg = ev.data as any;
  /* heap responses are handled by the proxy */
  sharedHeap.handle(msg);

  if (msg?.kind !== "start") return;

  tid = msg.tid;
  const prog = msg.prog as bril.Program;
  const func = msg.func as string;
  const args = msg.args as any[];

  /* build callee env */
  const callee = prog.functions.find(f => f.name === func)!;
  const env = new Map<bril.Ident, any>();
  (callee.args ?? []).forEach((p, i) => env.set(p.name, args[i]));

  try {
    await runBrilFunction(prog, func, env, sharedHeap as any);
    (self as DedicatedWorkerGlobalScope).postMessage({ kind: "done", tid });
  } catch (e) {
    (self as DedicatedWorkerGlobalScope).postMessage({ kind: "error", tid, message: String(e) });
  }
};