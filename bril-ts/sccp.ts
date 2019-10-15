#!/usr/bin/env node

import * as bril from "./bril";
import {readStdin} from "./util";

interface PhiOperation {
    op: "phi";
    args: bril.Ident[];
    dest: bril.Ident;
    type: bril.Type;
}

type SSAInstruction = bril.Instruction | PhiOperation;

class BasicBlock {
    constructor(label: string | null, insts: bril.Instruction[]) {
        this.label = label;
        this.instructions = insts;
    }
    label: string | null;
    instructions: SSAInstruction[];
    successors: BasicBlock[] = [];
    predecessors: BasicBlock[] = [];
    parent: BasicBlock | null = null;
    children: BasicBlock[] = [];
    frontier: Set<BasicBlock> | null= null
    id: number = 0;
}

function isLabel(inst: bril.Label | bril.Instruction): inst is bril.Label {
    return "label" in inst;
}

function isTerminator(inst: SSAInstruction): boolean {
    return ["br", "jmp", "ret"].includes(inst.op);
}

// Separates instructions into an array of basic blocks.
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

// Sets the .predecessors and .successors properties of blocks to
// represent the control-flow graph.
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

// Sets .parent and .children properties of blocks to the immediate
// dominator and the immediately dominated blocks respectively.
// Lengauer-Tarjan algorithm. 
function dominatorTree(blocks: BasicBlock[]) {
    const r = 1;
    let n = blocks.length + 1;
    let dom = new Array<number>(n);
    let parent = new Array<number>(n);
    let ancestor = new Array<number>(n);
    let child = new Array<number>(n);
    let vertex = new Array<number>(n);
    let label = new Array<number>(n);
    let semi = new Array<number>(n);
    let size = new Array<number>(n);
    let pred = new Array<Set<number>>(n);
    let bucket = new Array<Set<number>>(n);
    
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
        if (size[v] < 2 * size[w])
            [s, child[v]] = [child[v], s];
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
        for (let v of pred[w]) {
            let u = evalFunc(v);
            if (semi[u] < semi[w])
                semi[w] = semi[u];
        }
        bucket[vertex[semi[w]]].add(w);
        link(parent[w], w);
        for (let v of bucket[parent[w]]) {
            bucket[parent[w]].delete(v);
            let u = evalFunc(v);
            dom[v] = semi[u] < semi[v] ? u : parent[w];
        }
    }
    for (let i = 2; i <= n; i++) {
        let w = vertex[i];
        if (dom[w] != vertex[semi[w]])
            dom[w] = dom[dom[w]];
    }

    dom.forEach( (p: number, c: number) => {
        if (2 <= c) {
            blocks[c - 1].parent = blocks[p - 1];
            blocks[p - 1].children.push(blocks[c - 1]);
        }
    });
}

// m dom n
function dom(m: BasicBlock, n: BasicBlock): boolean {
    return m == n || n.parent != null && dom(m, n.parent);
}

function strictDom(m: BasicBlock, n: BasicBlock): boolean {
    return m != n && dom(m, n);
}

// The dominance frontier of a basic block. Memoized.
function dominanceFrontier(block: BasicBlock): Set<BasicBlock> {
    if (block.frontier != null)
        return block.frontier;
    let frontier: Set<BasicBlock> = new Set()
    for (let c of block.children)
        for (let n of dominanceFrontier(c))
            if (!strictDom(block, n))
                frontier.add(n);
    for (let s of block.successors)
        if (!strictDom(block, s))
            frontier.add(s);
    block.frontier = frontier;
    return frontier;
}

// Inserts phi functions where needed for SSA.
function insertPhis(blocks: BasicBlock[]) {
    let defs = new Map<string, Set<BasicBlock>>();
    for (let block of blocks)
        for (let inst of block.instructions)
            if ("dest" in inst) {
                let set = defs.get(inst.dest)
                if (set == undefined)
                    defs.set(inst.dest, new Set([block]));
                else
                    set.add(block);
            }
    for (let v of defs.keys()) {
        let visited = new Set<BasicBlock>();
        let vDefs = [...defs.get(v) as Set<BasicBlock>];
        while (vDefs.length != 0) {
            let block = vDefs.shift() as BasicBlock;
            if (visited.has(block))
                continue;
            visited.add(block);
            for (let fBlock of dominanceFrontier(block)) {
                let insts = fBlock.instructions;
                if (insts.length == 0 || insts[0].op != "phi" || insts[0].dest != v) {
                    insts.unshift({op: "phi", args: [], dest: v, type: "int"});
                    vDefs.push(fBlock);
                }
            }
        }
    }
}

async function main() {
    let prog: bril.Program = JSON.parse(await readStdin());
    let blocks = cfg(basicBlocks(prog.functions[0]));
    dominatorTree(blocks);
    insertPhis(blocks);
    console.log(blocks.map(b => b.instructions));
}

process.on('unhandledRejection', e => { throw e });

main();

