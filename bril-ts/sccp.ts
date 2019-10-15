#!/usr/bin/env node

import * as bril from "./bril";
import {readStdin} from "./util";

class BasicBlock {
    constructor(label: string | null, insts: bril.Instruction[]) {
        this.label = label;
        this.instructions = insts;
    }
    label: string | null;
    instructions: bril.Instruction[];
    successors: BasicBlock[] = [];
    predecessors: BasicBlock[] = [];
    parent: BasicBlock = this;
    children: BasicBlock[] = [];
    id: number = 0;
}

function isLabel(inst: bril.Label | bril.Instruction): inst is bril.Label {
    return "label" in inst;
}

function isTerminator(inst: bril.Instruction): boolean {
    return ["br", "jmp", "ret"].includes(inst.op);
}

function basicBlocks(func: bril.Function): BasicBlock[] {
    let out: BasicBlock[] = [];
    let block: BasicBlock = new BasicBlock(null, []);
    for (let inst of func.instrs) {
        if (isLabel(inst)) {
            if (block.label != null || block.instructions.length)
                out.push(block);
            block = new BasicBlock(inst.label, []);
        } else {
            block.instructions.push(inst);
            if (isTerminator(inst)) {
                out.push(block);
                block = new BasicBlock(null, []);
            }
        }
    }
    out.push(block);
    return out;
}

function cfg(blocks: BasicBlock[]): BasicBlock[] {
    let labelDict: { [label: string]: number } = {};
    let inCFG = new Set<number>();
    let queue = [0];
    let out: BasicBlock[] = [];
    let cfgSize = 0;

    function addEdge(b: BasicBlock, i: number) {
        let s = blocks[i]
        b.successors.push(s);
        s.predecessors.push(b);
        if (!inCFG.has(i))
            queue.push(i);
    }

    blocks.forEach( (b: BasicBlock, i: number) => {
        if (b.label != null)
            labelDict[b.label] = i;
    });

    while (queue.length) {
        let i = queue.shift() as number;
        if (inCFG.has(i))
            continue;
        inCFG.add(i);
        let block = blocks[i];
        block.id = ++cfgSize;
        out.push(block);
        let last = block.instructions[block.instructions.length - 1];
        if (last == undefined || !isTerminator(last)) {
            if (i + 1 < blocks.length)
                addEdge(block, i + 1);
        } else switch (last.op) {
            case "br":
                addEdge(block, labelDict[last.args[1]]);
                addEdge(block, labelDict[last.args[2]]);
                break;
            case "jmp":
                addEdge(block, labelDict[last.args[0]]);
                break;
        }
    }

    return out;
}

function dominatorTree(blocks: BasicBlock[]) {
    let n: number = blocks.length;
    const r = 1;
    let dom = new Array<number>(n + 1);
    let parent = new Array<number>(n + 1);
    let ancestor = new Array<number>(n + 1);
    let child = new Array<number>(n + 1);
    let vertex = new Array<number>(n + 1);
    let label = new Array<number>(n + 1);
    let semi = new Array<number>(n + 1);
    let size = new Array<number>(n + 1);
    let pred = new Array<Set<number>>(n + 1);
    let bucket = new Array<Set<number>>(n + 1);
    let u: number, v: number, x: number;
    
    function dfs(v: number) {
        semi[v] = ++n;
        vertex[n] = label[v] = v;
        ancestor[v] = child[v] = 0;
        size[v] = 1;
        for (let s of blocks[v - 1].successors) {
            let w = s.id;
            if (semi[w] == 0) {
                parent[w] = v;
                dfs(w);
            }
            pred[w].add(v);
        }
    }

    function compress(v: number) {
        if (ancestor[ancestor[v]] != 0) {
            compress(ancestor[v]);
            if (semi[label[ancestor[v]]] < semi[label[v]])
                label[v] = label[ancestor[v]];
            ancestor[v] = ancestor[ancestor[v]];
        }
    }

    function evalFunc(v: number): number {
        if (ancestor[v] == 0)
            return label[v];
        else {
            compress(v);
            if (semi[label[ancestor[v]]] >= semi[label[v]])
                return label[v]
            else
                return label[ancestor[v]];
        }
    }

    function link(v: number, w: number) {
        let s = w;
        while (semi[label[w]] < semi[label[child[s]]]) {
            if (size[s] + size[child[child[s]]] >= 2 * size[child[s]]) {
                ancestor[child[s]] = s;
                child[s] = child[child[s]];
            } else {
                size[child[s]] = size[s];
                s = ancestor[s] = child[s];
            }
        }
        label[s] = label[w];
        size[v] = size[v] + size[w];
        if (size[v] < 2 * size[w]) {
            let temp = s;
            s = child[v];
            child[v] = temp;
        }
        while (s != 0) {
            ancestor[s] = v;
            s = child[s];
        }
    }

    for (let v = 1; v <= n; v++) {
        pred[v] = new Set();
        bucket[v] = new Set();
        semi[v] = 0;
    }
    n = 0;
    dfs(r);
    size[0] = label[0] = semi[0] = 0;
    for (let i = n; i >= 2; i--) {
        let w = vertex[i];
        for (v of pred[w]) {
            u = evalFunc(v);
            if (semi[u] < semi[w])
                semi[w] = semi[u];
        }
        bucket[vertex[semi[w]]].add(w);
        link(parent[w], w);
        for (v of bucket[parent[w]]) {
            bucket[parent[w]].delete(v);
            u = evalFunc(v);
            dom[v] = semi[u] < semi[v] ? u : parent[w];
        }
    }
    for (let i = 2; i <= n; i++) {
        let w = vertex[i];
        if (dom[w] != vertex[semi[w]])
            dom[w] = dom[dom[w]];
    }
    dom[r] = 0;

    dom.forEach( (p: number, c: number) => {
        if (2 <= c) {
            blocks[c - 1].parent = blocks[p - 1];
            blocks[p - 1].children.push(blocks[c - 1]);
        }
    });
}

async function main() {
    let prog: bril.Program = JSON.parse(await readStdin());
    let blocks = cfg(basicBlocks(prog.functions[0]));
    dominatorTree(blocks);
    blocks.forEach( b => console.log(b.label + " : " + b.parent.label) );
}

process.on('unhandledRejection', e => { throw e });

main();

