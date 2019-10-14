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

function cfg(blocks: BasicBlock[]): BasicBlock {
    let labelDict: { [label: string]: number } = {};
    let inCFG = new Set<number>();
    let start = blocks[0];
    let queue = [0];

    let addEdge = (b: BasicBlock, i: number) => {
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

    return start;
}

async function main() {
    let prog: bril.Program = JSON.parse(await readStdin());
    let start = cfg(basicBlocks(prog.functions[0]));
    console.log(start.successors[1]);
}

process.on('unhandledRejection', e => { throw e });

main();
