import java.util.LinkedList
import java.util.TreeMap
import java.util.TreeSet

typealias Block = LinkedList<BrilInstr>
fun blocks(brilFunction: BrilFunction): List<Block> {
    val result = LinkedList<Block>()
    var curr = LinkedList<BrilInstr>()
    if (brilFunction.instrs.firstOrNull() !is BrilLabel) {
        //System.err.println("first is")
        //System.err.println(brilFunction.instrs.firstOrNull())
        curr.add(BrilLabel(label = "syntactic_entry", pos = null))
    }
    brilFunction.instrs.forEach { instr -> 
        when (instr) {
            is BrilLabel -> {
                if (curr.isNotEmpty()) {
                    result.add(curr)
                    curr = LinkedList<BrilInstr>()
                }
                curr.add(instr)
            }
            is BrilOp -> {
                curr.add(instr)
                if (instr.op == "jmp" || instr.op == "br") {
                    result.add(curr)
                    curr = LinkedList<BrilInstr>()
                }
            }
        }
    }

    if (curr.isNotEmpty()) {
        result.add(curr)
    }

    if (result.last().last() !is BrilRetOp) {
        result.last().add(BrilRetOp(op = "ret", arg = null))
    }
    return result
}

fun blocksToIds(
    blocks: List<Block>
): TreeMap<Int, Block> {
    val result = TreeMap<Int, Block>()
    blocks.forEachIndexed { i, b -> 
        result.put(i, b)
    }
    return result
}

fun bidToLabel(
    blocks: List<Block>
): TreeMap<Int, String> {
    val result = TreeMap<Int, String>()
    blocks.forEachIndexed { i, b -> 
        val instr = b.firstOrNull()
        if (instr is BrilLabel) {
            result[i] = instr.label
        }
    }
    return result
}

typealias LabelToBlock = TreeMap<String, Int>
fun labelToBlock(
    blocks: List<Block>
): LabelToBlock {
    val result = LabelToBlock()
    blocks.forEachIndexed { i, b -> 
        when (val l = b.firstOrNull()) {
            is BrilLabel -> {
                result.put(l.label, i)
            }
            else -> Unit
        }
    }
    return result
}

typealias Cfg = TreeMap<Int, TreeSet<Int>>
typealias Blocks = List<Block>

fun cfg(
    function: BrilFunction,
): Pair<Blocks, Cfg> {
    val blocks = blocks(function)
    val cfg = cfg(blocks, labelToBlock(blocks))
    var preventBackEdgesToEntry = false
    for (i in 1 ..< blocks.size) {
        if (cfg[i]!!.contains(0)) {
            preventBackEdgesToEntry = true
            break
        }
    }

    return if (preventBackEdgesToEntry) {
        val startBlock = Block()
        startBlock.add(
            BrilLabel(
                label = "dg6413",
                pos = null,
            )
        )
        startBlock.add(
            BrilJmpOp(
                op = "jmp",
                label = (blocks[0][0] as BrilLabel).label,
            )
        )
        (blocks as LinkedList<Block>).add(0, startBlock)
        for (i in cfg.size - 1 downTo 0) {
            cfg.put(i + 1, TreeSet(cfg[i]!!))
        }
        cfg.put(0, TreeSet<Int>().let { it.add(1); it })
        blocks to cfg(blocks, labelToBlock(blocks))
    } else {
        blocks to cfg(blocks, labelToBlock(blocks))
    }
}

fun cfg(
    blocks: List<Block>,
    labelToBlock: TreeMap<String, Int>,
): Cfg {
    val result = Cfg()
    blocks.forEachIndexed { i, b -> 
        val edgesOut = TreeSet<Int>()
        val last = b.lastOrNull() as? BrilOp
        if (last is BrilRetOp) {
            // Do not add an edge out
        } else if (last is BrilJmpOp) {
            edgesOut.add(labelToBlock.get(last.label)!!)
        } else if (last is BrilBrOp) {
            edgesOut.add(labelToBlock.get(last.labelL)!!)
            edgesOut.add(labelToBlock.get(last.labelR)!!)
        } else if (i < blocks.size - 1) {
            edgesOut.add(i + 1)
        }
        result.put(i, edgesOut)
    }
    return result
}

typealias Predecessors = TreeMap<Int, TreeSet<Int>>
fun predecessors(
    cfg: Cfg,
): Predecessors {
    val res = Predecessors()
    for (i in 0 ..< cfg.size) {
        res[i] = TreeSet()
    }

    for (i in 0 ..< cfg.size) {
        val next = cfg[i]!!
        for (b in next) {
            res[b]!!.add(i)
        }
    }
    return res
}
