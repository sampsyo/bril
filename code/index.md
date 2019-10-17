## Implementation

### Finding loops

All loops are built around back edges. We started by finding all pairs of `[tail, head]` such that there exists an  edge `tail->head` where `head` dominates `tail`. This is demonstrated by the pseudo code below:

```python
for node in cfg:
  for dominator in dom[node]: # dom[node] is an array of dominators of the node
    if dominator in successor[node]: # successor[node] is an array of successors of the node
      list.append([node,dominator]) #storing the backedge pair of [tail,head] into the list
```



Assuming this is a reversible CFG, all back edges would be associated with a natural loop. This assumption makes it easier to find loops associated with each back edge. Hence, we'd like to associate a list of nodes (basic blocks) which form a loop corresponding to a back edge pair from the previous step.

For a back edge `A->B` we know that `B` dominates `A`, hence all paths to `A` are through `B`. So we can find predecessors of `A` recursively until we hit `B` and include them in the loop. This is the smallest set of points which include `A` and `B` such that for all nodes, `n` in the list `L`:  `preds(n)` $\in$ `L` or `n = B` . 

```python
all_loops = []
for A,B in back_edges:
  natural_loop=[B]
  explored=[B] # won't find predecessors of explored nodes
  find_pred(A,natural_loop,explored) # add predecessors and current node to natural_lop list
  all_loops.append(natural_loop)
return all_loops
```



### Reaching Definition

For finding loop invariant code, we look into reaching definitions of the argument and check wether they are outside the loop. The reaching definition problem was discussed in the lecture of CS 6120. For each block we can generate a list of reaching inputs and outputs. The data structure that we chose to represent this is a dictionary of dictionaries. We decided to store the block number associated with the variable in each block. Hence reaching definitions of each block has a dictionary of variables (as keys) and block number they were defined in (as values). This helps in keeping track of reaching definitions outside the loop to identify loop invariant code.  The reaching definitions look something like this:

```
reaching_defs = {blocks:{variable:[list of block numbers]}}
```

Instead of the usual:

```
reaching_defs = {blocks:[list of variables]}
```

The reaching definition problem can be solved using the worklist algorithm after defining merge and the transfer function:

<img src="reaching_defs.png" alt="reaching definition lecture" style="zoom:80%;" />

The union function for our data structure is more nuanced than a simple union of sets of variables. In case of multiple definitions of a variable from predecessor blocks we union the list of block numbers associated with the variable. So if block 1 and block 2 are the inputs for the current block we merge (take union) the lists corresponding to each variable in these blocks. This way we keep track of the block numbers of definitions of each variable.

```python
out = {}
    for s in dicts:
        for k,v in s.items():
            if k not in out:
                out.update({k:v}) # add a reaching definition of a variable 'k' if not already present
            else:
                out[k] = list(set(out[k])|set(v)) # take union of lists of block numbers for a  variable 'k'
    return out
```



