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

type LatticeElement = "top" | "bottom" | bril.Value;

type FlowWorkListItem = [BasicBlock, BasicBlock]
type SSAWorkListItem = [SSAInstruction, BasicBlock]
type WorkListItem = FlowWorkListItem | SSAWorkListItem;

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
    frontier: Set<BasicBlock> | null = null;
    id: number = 0;
    inEdgeExecutable: boolean[] = [];
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
    let labelMap = new Map<string, number>();
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

    blocks.forEach((b: BasicBlock, i: number) => {
        if (b.label != null)
            labelMap.set(b.label, i);
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
                addEdge(block, labelMap.get(last.args[1]) as number);
                addEdge(block, labelMap.get(last.args[2]) as number);
                break;
            case "jmp":
                addEdge(block, labelMap.get(last.args[0]) as number);
            case "ret":
                block.instructions.pop();
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

    dom.forEach((p: number, c: number) => {
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
    let types = new Map<string, bril.Type>();
    for (let block of blocks)
        for (let inst of block.instructions)
            if ("dest" in inst) {
                let set = defs.get(inst.dest)
                if (set == undefined) {
                    defs.set(inst.dest, new Set([block]));
                    types.set(inst.dest, inst.type);
                } else
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
                    insts.unshift({
                        op: "phi",
                        args: new Array(fBlock.predecessors.length).fill(v),
                        dest: v,
                        type: types.get(v) as bril.Type
                    });
                    vDefs.push(fBlock);
                }
            }
        }
    }
}

// Renames all variables such that each definition is to a unique variable.
// Returns a map from variables to all their uses, and a map to their def.
function renameVariables(blocks: BasicBlock[]):
        [Map<string, SSAWorkListItem[]>, Map<string, SSAWorkListItem>] {
    let stacks = new Map<string, string[]>();
    let defCounts = new Map<string, number>();
    let def = new Map<string, SSAWorkListItem>();
    let out = new Map<string, SSAWorkListItem[]>();

    function getStack(x: string): string[] {
        if (stacks.has(x))
            return stacks.get(x) as string[];
        let s: string[] = [];
        stacks.set(x, s);
        return s;
    }

    function rename(block: BasicBlock) {
        let pushCounts = new Map<string, number>();

        for (let inst of block.instructions) {
            if ("args" in inst)
                switch (inst.op) {
                    case "phi":
                    case "jmp":
                        break;
                    case "br":
                        let s = getStack(inst.args[0]);
                        let newArg = s[s.length - 1];
                        let uses = out.get(newArg) as [SSAInstruction, BasicBlock][];
                        inst.args[0] = newArg;
                        uses.push([inst, block]);
                        break;
                    default:
                        inst.args.forEach((arg, i, args) => {
                            let s = getStack(arg);
                            let newArg = s[s.length - 1];
                            let uses = out.get(newArg) as [SSAInstruction, BasicBlock][];
                            args[i] = newArg;
                            uses.push([inst, block]);
                        });
                }
            if ("dest" in inst) {
                let dest = inst.dest;
                let defs = (defCounts.has(dest) ? defCounts.get(dest) : 0) as number;
                let pushes = (pushCounts.has(dest) ? pushCounts.get(dest) : 0) as number;
                let newDest = dest + "_" + defs;
                defCounts.set(dest, defs + 1);
                pushCounts.set(dest, pushes + 1);
                getStack(dest).push(newDest);
                def.set(newDest, [inst, block]);
                out.set(newDest, []);
                inst.dest = newDest;
            }
        }
        for (let succ of block.successors) {
            let pred = -1;
            while (block != succ.predecessors[++pred]);
            for (let inst of succ.instructions) {
                if (inst.op != "phi")
                    break;
                let s = getStack(inst.args[pred]);
                let newArg = s[s.length - 1];
                let uses = out.get(newArg);
                inst.args[pred] = newArg;
                if (uses != undefined)
                    uses.push([inst, succ]);
            }
        }
        for (let child of block.children)
            rename(child)
        pushCounts.forEach((c, x) => getStack(x).splice(-c, c));
    }

    rename(blocks[0]);

    // remove variables that have no uses
    let toRemove: string[] = [];
    out.forEach((v, k) => {
        if (v.length == 0)
            toRemove.push(k);
    });
    while (toRemove.length != 0) {
        let rem = toRemove.pop() as string;
        let [i, b] = def.get(rem) as SSAWorkListItem;
        out.delete(rem);
        b.instructions.splice(b.instructions.indexOf(i), 1);
        if ("args" in i) {
            for (let arg of i.args) {
                if (arg != undefined) {
                    let uses = out.get(arg) as SSAWorkListItem[];
                    uses.splice(uses.findIndex(u => u[0] == i), 1);
                    if (uses.length == 0)
                        toRemove.push(arg);
                }
            }
        }
    }
    return [out, def];
}

// greatest lower bound of x and y in the lattice
function meet(x: LatticeElement, y: LatticeElement): LatticeElement {
    if (x == "top")
        return y;
    if (y == "top" || x == y)
        return x;
    return "bottom";
}

function toInt(x: LatticeElement): number {
    if (typeof x == 'number')
        return x;
    throw "expected int"
}

function toBool(x: LatticeElement): boolean {
    if (typeof x == 'boolean')
        return x;
    throw "expected boolean"
}


// Performs the sparse conditional constant propagation analysis.
// Returns a mapping from variables to their lattice value.
// Does not modify the program.
function sccp(blocks: BasicBlock[], uses: Map<string, SSAWorkListItem[]>):
        Map<string, LatticeElement> {
    
    let out = new Map<string, LatticeElement>();
    let start = new BasicBlock(null, []);
    let worklist: WorkListItem[] = [[start, blocks[0]]];

    function roundTowardZero(x: number): number {
        return x < 0 ? Math.ceil(x) : Math.floor(x);
    }

    function evaluateExpression(inst: bril.ValueOperation): LatticeElement {
        let argLatElems = inst.args.map(v => out.get(v)) as LatticeElement[];
        if (!["and", "or"].includes(inst.op) && argLatElems.includes("bottom"))
            return "bottom";
        switch (inst.op) {
        case "add": return toInt(argLatElems[0]) + toInt(argLatElems[1]);
        case "mul": return toInt(argLatElems[0]) * toInt(argLatElems[1]);
        case "sub": return toInt(argLatElems[0]) - toInt(argLatElems[1]);
        case "div":
            if (argLatElems[1] == 0)
                return "bottom"
            else
                return roundTowardZero(toInt(argLatElems[0]) / toInt(argLatElems[1]));
        case "id":  return argLatElems[0];
        case "eq":  return argLatElems[0] == argLatElems[1];
        case "lt":  return toInt(argLatElems[0]) < toInt(argLatElems[1]);
        case "gt":  return toInt(argLatElems[0]) > toInt(argLatElems[1]);
        case "ge":  return toInt(argLatElems[0]) >= toInt(argLatElems[1]);
        case "le":  return toInt(argLatElems[0]) <= toInt(argLatElems[1]);
        case "not": return !toBool(argLatElems[0]);
        case "and": return !argLatElems.includes(false);
        case "or":  return argLatElems.includes(true);
        }
        throw "constant propagation error";
    }

    function visitPhi(inst: PhiOperation, executable: boolean[]) {
        let executables = inst.args.filter((a, i) => executable[i]);
        let elems = executables.map(v => out.get(v)) as LatticeElement[];
        let glb = elems.reduce(meet, "top");
        if (glb != out.get(inst.dest)) {
            out.set(inst.dest, glb);
            for (let use of uses.get(inst.dest) as SSAWorkListItem[])
                worklist.push(use);
        }
    }

    function visitExpression(inst: SSAInstruction, block: BasicBlock) {
        switch (inst.op) {
        case "br": // add executable out flow edges to worklist
            let latElem = out.get(inst.args[0]) as LatticeElement;
            if (meet(latElem, false) == "bottom")
                worklist.push([block, block.successors[0]]);
            if (meet(latElem, true) == "bottom")
                worklist.push([block, block.successors[1]]);
            break;
        case "const": // set lattice element to const value
            if (out.get(inst.dest) != inst.value) {
                out.set(inst.dest, inst.value);
                for (let use of uses.get(inst.dest) as SSAWorkListItem[])
                    worklist.push(use);
            }
            break;
        case "print": // do nothing
        case "nop":
            break;
        default: // value operation
            let valOp = inst as bril.ValueOperation
            let updated = evaluateExpression(valOp);
            if (out.get(valOp.dest) != updated) {
                out.set(valOp.dest, updated);
                for (let use of uses.get(valOp.dest) as SSAWorkListItem[])
                    worklist.push(use);
            }
        }
    }

    blocks[0].predecessors.push(start);
    for (let v of uses.keys())
        out.set(v, "top");
    for (let block of blocks)
        block.inEdgeExecutable = new Array(block.predecessors.length).fill(false);
    while (worklist.length) {
        let [predOrInst, block] = worklist.pop() as WorkListItem;
        if (predOrInst instanceof BasicBlock) {
            let predIdx = block.predecessors.indexOf(predOrInst);
            if (block.inEdgeExecutable[predIdx])
                continue;
            block.inEdgeExecutable[predIdx] = true;
            let i = 0;
            while (i < block.instructions.length && block.instructions[i].op == "phi")
                visitPhi(block.instructions[i++] as PhiOperation, block.inEdgeExecutable);
            if (block.inEdgeExecutable.every((b, i) => !b || i == predIdx))
                while (i < block.instructions.length)
                    visitExpression(block.instructions[i++], block);
            if (block.successors.length == 1)
                worklist.push([block, block.successors[0]]);
        } else if (predOrInst.op == "phi")
            visitPhi(predOrInst, block.inEdgeExecutable);
        else if (block.inEdgeExecutable.some(b => b))
            visitExpression(predOrInst, block);
    }
    return out;
}

// Performs optimizations based on SCC analysis results: removes branches determined
// to have a constant condition, removes unreachable blocks, replaces operations with
// consts where possible, removes unused variables.
function replaceConstants(blocks: BasicBlock[], uses: Map<string, SSAWorkListItem[]>,
        defs: Map<string, SSAWorkListItem>, values: Map<string, LatticeElement>) {
    for (let block of blocks) {
        if (block.inEdgeExecutable.some(b => b)) {
            if (block.successors.length != 2)
                continue;
            let insts = block.instructions;
            let branch = insts[insts.length - 1] as bril.EffectOperation;
            let val = values.get(branch.args[0]);
            if (typeof val != "boolean")
                continue;
            insts.pop();
            let u = uses.get(branch.args[0]) as SSAWorkListItem[];
            u.splice(u.findIndex(([i, b]) => i == branch), 1);
            let removedSucc = block.successors[val ? 1 : 0];
            let remainingSucc = block.successors[val ? 0 : 1];
            block.successors = [remainingSucc];
            removedSucc.predecessors.splice(removedSucc.predecessors.indexOf(block), 1);
        } else {
            for (let [v, u] of uses)
                uses.set(v, u.filter(([i, b]) => b != block));
            for (let [v, [i, b]] of defs)
                if (b == block)
                    defs.delete(v);
            for (let succ of block.successors)
                succ.predecessors.splice(succ.predecessors.indexOf(block), 1);
        }
    }
    for (let [v, val] of values) {
        if (val == "bottom" || val == "top")
            continue;
        let [def, defBlock] = defs.get(v) as SSAWorkListItem;
        if (def.op == "const")
            continue;
        for (let arg of def.args) {
            let u = uses.get(arg) as SSAWorkListItem[];
            u.splice(u.findIndex(([i, b]) => i == def, 1));
        }
        let insts = defBlock.instructions;
        let type = (def as bril.ValueOperation | PhiOperation).type;
        let newDef: bril.Constant = {op: "const", value: val, dest: v, type: type};
        insts[insts.indexOf(def)] = newDef;
        defs.set(v, [newDef, defBlock]);
    }

    let toRemove: string[] = [];
    for (let [v, u] of uses)
        if (u.length == 0)
            toRemove.push(v);
    while (toRemove.length != 0) {
        let v = toRemove.pop() as string;
        let [def, defBlock] = defs.get(v) as SSAWorkListItem;
        let insts = defBlock.instructions;
        insts.splice(insts.indexOf(def), 1);
        if ("args" in def)
            for (let arg of def.args) {
                let u = uses.get(arg) as SSAWorkListItem[];
                u.splice(u.findIndex(([i, b]) => i == def), 1);
                if (u.length == 0)
                    toRemove.push(arg);
            }
    }
}

// Converts out of SSA form. This is a hacky version that relies on the
// above implementation of to-SSA conversion, and properties of SCCP,
// but produces a more efficient (pre-move coalescing) program than
// an actual from-SSA.
function removePhis(blocks: BasicBlock[]) {
    function removeSubscript(x: string): string {
        return x.substring(0, x.lastIndexOf("_"));
    }

    for (let block of blocks) {
        let insts = block.instructions;
        for (let i = 0; i < insts.length; i++) {
            let inst = insts[i];
            switch (inst.op) {
            case "phi":
                insts.splice(i--, 1);
                continue;
            case "jmp":
            case "const":
                break;
            case "br":
                inst.args[0] = removeSubscript(inst.args[0]);
                break;
            default:
                inst.args = inst.args.map(removeSubscript);
            }
            if ("dest" in inst)
                inst.dest = removeSubscript(inst.dest);
        }
    }
}

// Convert CFG back into a list of instructions, inserting
// branches, jumps, and labels where necessary.
function flattenCFG(blocks: BasicBlock[]): (bril.Instruction | bril.Label)[] {
    let placed = new Set<BasicBlock>();
    let stack = [blocks[0]];
    let out: (bril.Instruction | bril.Label)[] = [];

    function label(): string {
        //TODO
        return "totally_unique_label";
    }

    function placeBlock(block: BasicBlock) {
        if (block.label != null)
            out.push({label: block.label});
        out = out.concat(block.instructions as bril.Instruction[]);
        placed.add(block);
    }

    while (stack.length != 0) {
        let block = stack.pop() as BasicBlock;
        if (placed.has(block))
            continue;
        placeBlock(block);
        while (block.successors.length == 1) {
            let succ = block.successors[0];
            if (placed.has(succ)) {
                if (succ.label == null)
                    succ.label = label();
                block.instructions.push({op: "jmp", args: [succ.label]});
                break;
            }
            block = succ;
            placeBlock(block);
        }
        if (block.successors.length == 0 && stack.length != 0)
            out.push({op: "ret", args: []});
        for (let succ of block.successors)
            if (!placed.has(succ) && !stack.includes(succ))
                stack.push(succ);
    }

    let referencedLabels = new Set<string>();
    for (let inst of out) {
        if (!("op" in inst))
            continue;
        switch (inst.op) {
        case "br":
            referencedLabels.add(inst.args[1]);
            referencedLabels.add(inst.args[2]);
            break;
        case "jmp":
            referencedLabels.add(inst.args[0]);
        }
    }
    for (let i = 0; i < out.length; i++) {
        let inst = out[i];
        if ("label" in inst && !referencedLabels.has(inst.label))
            out.splice(i--, 1)
    }

    return out;
}

async function main() {
    let prog: bril.Program = JSON.parse(await readStdin());
    for (let func of prog.functions) {
        let blocks = cfg(basicBlocks(func));
        dominatorTree(blocks);
        insertPhis(blocks);
        let [uses, defs] = renameVariables(blocks);
        let values = sccp(blocks, uses);
        replaceConstants(blocks, uses, defs, values);
        removePhis(blocks);
        func.instrs = flattenCFG(blocks);
    }
    console.log(JSON.stringify(prog, undefined, 2));
}

process.on('unhandledRejection', e => { throw e });

main();

