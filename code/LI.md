### Finding Loop Invariants

The loop invariant is instruction that does not change execution result during the loop execution. We need to go to individual loops, individual blocks in the loop and finally individual instructions inside the loop to check if the instruction is loop invariant. A instruction is loop invariant when any of its argument is either constant or variable `var` satisfying:

1. all reaching definitions of `var` are outside the loop.
2. There is exactly one reaching definition of `var` and the definition is loop-invariant.

These two condition is realized by the following code.

```python
rd = reach_def[block][var] #var is defined at rd block
c1 = all([x not in loops[loop] for x in rd ]) #all rd blocks outside the loop
li = loop_invariants[loop].get(rd[0]) #None or LIs in rd block
li = [] if li is None else li
c2 = len(rd)==1 and any([var == i['dest'] for i in li]) #one reaching definition and var is defined as LI (matches one of dest in LIs in rd block).
```

### Create Pre-Headers of Loop Headers

Before we actually move code, we need to create pre-headers for loop headers. These pre-headers are empty blocks that should be placed before loop header blocks. Notice this assumes bril code should not have two edges that does not the loop pointing to the loop header. Using these empty blocks, we can easily move loop invariants out of the loop when the requirements are satisfied.

In implementation, for each block, we first copy old block content and then check if the next block is a loop header. If so, we create an empty block. 

```python
for edge in loops:#we use back edge as key to denote loop
    if b_names[i+1] in edge[1]: #edge[1] is the pre-header block name
    	name = fresh('b', new_blocks) # generate block name that is never used before
        new_blocks[name] = []
        pre_header = {x:name for x in loops[edge]}
        break
```

<img src="move_LI.png" alt="reaching definition lecture" style="zoom:60%;" />

### Move Loop Invariant to Pre-Headers

Not all pre-headers are allowed to be moved to the pre-headers. If the destination of an LI is `d`, it needs to satisfy the following condition:

1. There is only one definition of  `d` in the loop
2.  `d`dominates all its uses, or equivalently, `d` is not live-out of it's pre-header.
3.  `d`s block dominates all loop exits where $d$ is live-out

To learn the first condition, we need to know all definitions inside the loop and check if  `d` is unique in the list `defs`

```python
defs = [ins.get('dest') for b in loops[back_edge] for ins in blocks[b] if ins.get('dest')]
defs.count(instr['dest']) ==  # if true, first check passed
```

For the second condition, we can check the predecessor block of pre-header we just added, by simply read the index of pre-header and subtract 1. Then check if `d` is live-out of block.

```python
ind = b_names.index(pre_header[b_name]) - 1 #b_name is the name of block where d is LI.
instr['dest'] not in live_var[1][b_names[ind]] #if true, second check passed
```

For the third condition, we need information of exits of blocks. The exits are blocks that have successors not in the loop. 

```python
exits = {}
for k in loops:
    exits[k] = []
    for l in loops[k]: 
        if any( s not in loops[k] for s in succ[l]):
            exits[k].append(l)
```

After that, we just need to find all exit blocks where `d` is live-out and check if `d`'s' block dominates all loop exits.

```python
edest = [e for e in exits[back_edge] if instr['dest'] in live_var[1][e]]
all([b_name in dom[e] for e in edest]) # if true, third check passed.
```

### Block to Dictionaries

` json.load(sys.stdin)['functions']` gives us dictionaries and for each dictionary we obtain list of instructions when the key is `instrs`. Then we change this list of instructions into a directory of blocks. We would like to reverse this process to regenerate list of instructions with modified blocks. However, original block does not have so many labels introduced when generating block names and creating pre-headers. Luckily, the modified blocks are still ordered dictionary and we can omit blocks we introduced. Therefore in this step, we only create `label` instruction when the original labels in blocks. The rest is just copy every instruction other than labels to the new list of instructions.

## Hardest Part
1. There are more properties we need than we originally expected. At first, we only generated loops, reaching variables. Then we for loop invariants code motion, we needed exits to the loop, dominance relations, live variables. 
2. The representation of different variables are randomly decided at first and need implementation after we finalized the representation. For example,the loop invariant at first was stored as the list of instructions. Later, we found it necessary to change the storage form to the dictionary whose key is the block name. Otherwise, we would need to search and match the instruction to block, e.g, in the `move_LI` function.
