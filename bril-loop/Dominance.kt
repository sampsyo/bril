
import com.squareup.moshi.*
import okio.*
import java.util.TreeSet
import java.util.TreeMap
import java.io.*
import com.squareup.moshi.kotlin.reflect.KotlinJsonAdapterFactory

typealias Doms = TreeMap<Int, TreeSet<Int>>

// The dominates algorithm that we saw in class, notice that this algo 
// returns the "is dominated by" not the "dominates" relation. To obtain
// the domiantes relation, you can use the flip_doms function to convert
// the representation.
fun doms(
    blocks: List<Block>,
    cfg: Cfg,
): Doms {
    val predecessors = predecessors(cfg)
    fun bigcap(c: Collection<TreeSet<Int>>): TreeSet<Int> {
        return c.reduceOrNull { acc, s ->
            acc.filter { s.contains(it) }.let { TreeSet(it) }
        }
        ?: TreeSet<Int>()
    }
    var res = TreeMap<Int, TreeSet<Int>>()
    // Starting point, all blocks are dominated by all others.
    blocks.forEachIndexed { i, b -> 
        val all = TreeSet<Int>()
        for(i in 0..<blocks.size) {
            all.add(i)
        }
        res.put(i, all)
    }
    res.put(0, TreeSet<Int>().also { it.add(0)})

    var prev: Doms? = null
    while (prev != res) {
        prev = res
        res = TreeMap(res)
        // starting at 1 to ignore entry
        for(i in 1..<blocks.size) {
            val preds = predecessors[i]!!
            val preddoms = preds.map{ TreeSet(res.get(it)!!) }
            val bigcap = bigcap(preddoms)
            bigcap.add(i)
            res.put(i, bigcap)
        }
    }
    return res
}

// Convert the "is-dominated" by relation to the "dominates" relation.
// By default also converts to strict domination, use the argument 
// strict = false to preserve reflexivity.
fun flip_doms(
    doms: Doms,
    strict: Boolean = true,
): Doms {
    val res: Doms = Doms()
    for (i in 0 ..< doms.size) {
        res[i] = TreeSet<Int>()
    }

    for (i in 0 ..< doms.size) {
        val vi = i
        val ni = doms[i]!!
        for (vj in ni) {
            if (strict && vi != vj) {
                res[vj]!!.add(vi)
            } else if (!strict) {
                res[vj]!!.add(vi)
            }
        }
    }

    return res
}

data class DTree(
    val bid: Int,
    val children: List<DTree>
)

fun DTree.formatted(level: Int = 0): String {
    val indent = "  ".repeat(level)
    return children.fold("$indent$bid") { acc, tree -> 
        "$acc\n${tree.formatted(level + 1)}"
    }
}

typealias Idoms = TreeMap<Int, TreeSet<Int>>

// A naive implementation of the immediate dominators relations of a 
// dominators graph. Notice that this function returns the "dominates by"
// relation, and not the "is dominated by" relation. 
// Inefficient, has O(n^3) complexity, but can be useful for testing
// and as a nice direct translation to the mathematical definition.
fun idomsNaive(
    doms: Doms,
): Idoms {
    val res = Idoms()
    // Initialization
    for (i in 0..< doms.size) {
        res.put(i, TreeSet<Int>())
    }

    for (i in 0 ..< doms.size) {
        val vi = i
        val ni = doms[i]!!
        for (j in 0 ..< doms.size) {
            val vj = j
            val nj = doms[j]!!
            // if vi dominates vk.
            if (vj == vi || nj.contains(vi).not()) {
                continue
            }
            var isUnique = true
            for (k in 0 ..< doms.size) {
                val vk = k
                val nk = doms[k]!!
                if (vk == vi || vk == vj) {
                    continue
                }
                if (nj.contains(vk).not()) {
                    continue
                }
                // if some note vk s.t. vi dominates vk and vk dominates vj
                if (nk.contains(vi) && nj.contains(vk)) {
                    isUnique = false
                    break
                }
            }
            if (isUnique) {
                res[vi]!!.add(vj)
            }
        }
    }

    return res
}

// An alternative definition, relies on BFS/DFS to build the graph.
// Requires doms to be the "dominates" irreflexive relation. See flip_doms for more info.
// More efficient than idomsNaive, but relies on an auxiliary TreeSet so 
// complexity is O(n * log n)
fun idomsSearch(
    doms: Doms
): Idoms {
    val res = Idoms()
    // Initialization
    for (i in 0..< doms.size) {
        res.put(i, TreeSet<Int>())
    }
    
    // Although using a tree map, this works as a priority queue.
    // The idea is to process nodes starting with the ones that dominate the least, and finish
    // up with the ones that dominate the most (the entry of the cfg)
    // The key is the number of nodes that it dominates, the second is the id of the node.
    val queue = TreeMap<Int, TreeSet<Int>>()
    for (i in 0 ..< doms.size) {
        val vi = i
        val ni = doms[i]!!
        val si = ni.size
        if (queue.contains(si).not()) {
            queue.put(si, TreeSet<Int>())
        }
        queue[si]!!.add(vi)
    }

    // The nodes that have already been visited.
    val seen = TreeSet<Int>()
    
    queue.forEach { (_, nodes) -> 
        nodes.forEach { v -> 
            for (n in doms[v]!!) {
                if (seen.contains(n).not()) {
                    seen.add(n)
                    res[v]!!.add(n)
                }
            }
        }
    }

//    while (queue.isNotEmpty()) {
//        val (l, v) = queue.pollFirst()
//        for(n in doms[v]!!) {
//            if (seen.contains(n).not()) {
//                seen.add(n)
//                res[v]!!.add(n)
//            }
//        }
//    }

    return res
}

fun dtree(
    idoms: Idoms,
    // By default we bulid the dtree of the first basic block in the function
    start: Int = 0,
): DTree {
    return DTree(
        start,
        children = idoms[start]!!.mapNotNull { 
            dtree(idoms, it)
        }
    )
}

typealias DomFrontier = TreeMap<Int, TreeSet<Int>>

// Return the dominance frontier, expects to receive the non-strict "dominates" relation.
fun domfrontier(
    doms: Doms,
    cfg: Cfg,
): DomFrontier {
    val res = DomFrontier()
    val predecessors = predecessors(cfg)
    for (vi in 0 ..< cfg.size) {
        res.put(vi, TreeSet())
        for (vj in 0 ..< cfg.size) {
            if (doms[vi]!!.contains(vj)) {
                continue
            }
            for (vk in predecessors[vj]!!) {
                if (doms[vi]!!.contains(vk) || vi == vk) {
                    res[vi]!!.add(vj)
                    break
                }
            }
        }
    }
    return res
}
