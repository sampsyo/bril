import sys
import json
from collections import namedtuple

from form_blocks import form_blocks
import cfg
from util import var_args

# A single dataflow analysis consists of these part:
# - forward: True for forward, False for backward.
# - init: An initial value (bottom or top of the latice).
# - merge: Take a list of values and produce a single value.
# - transfer: The transfer function.
Analysis = namedtuple('Analysis', ['forward', 'init', 'merge', 'transfer'])


def union(sets):
    out = set()
    for s in sets:
        out.update(s)
    return out


def df_worklist(blocks, analysis):
    """The worklist algorithm for iterating a data flow analysis to a
    fixed point.
    """
    preds, succs = cfg.edges(blocks)

    # Switch between directions.
    if analysis.forward:
        first_block = list(blocks.keys())[0]  # Entry.
        in_edges = preds
        out_edges = succs
    else:
        first_block = list(blocks.keys())[-1]  # Exit.
        in_edges = succs
        out_edges = preds

    # Initialize.
    in_ = {first_block: analysis.init}
    out = {node: analysis.init for node in blocks}

    # Iterate.
    worklist = list(blocks.keys())
    while worklist:
        node = worklist.pop(0)

        inval = analysis.merge(out[n] for n in in_edges[node])
        in_[node] = inval

        outval = analysis.transfer(blocks, node, inval)

        if outval != out[node]:
            out[node] = outval
            worklist += out_edges[node]

    if analysis.forward:
        return in_, out
    else:
        return out, in_


def fmt(val):
    """Guess a good way to format a data flow value. (Works for sets and
    dicts, at least.)
    """
    if isinstance(val, set):
        if val:
            return ', '.join(v for v in sorted(val))
        else:
            return 'empty'
    elif isinstance(val, dict):
        if val:
            return ', '.join('{}: {}'.format(k, v)
                             for k, v in sorted(val.items()))
        else:
            return 'empty'
    else:
        return str(val)

def run_df(func_instrs, analysis):
    # Form the CFG.
    blocks = cfg.block_map(form_blocks(func_instrs))
    cfg.add_terminators(blocks)

    in_, out = df_worklist(blocks, analysis)

    return in_, out

def gen(block):
    """Variables that are written in the block.
    """
    return {i['dest'] for i in block if 'dest' in i}


def use(block):
    """Variables that are read before they are written in the block.
    """
    defined = set()  # Locally defined.
    used = set()
    for i in block:
        used.update(v for v in var_args(i) if v not in defined)
        if 'dest' in i:
            defined.add(i['dest'])
    return used


def defined_transfer(blocks, name, in_):
  return in_.union(gen(blocks[name]))


def live_transfer(blocks, name, out):
  block = blocks[name]
  return use(block).union(out - gen(block))


def cprop_transfer(blocks, name, in_vals):
  block = blocks[name]
  out_vals = dict(in_vals)
  for instr in block:
    if 'dest' in instr:
      if instr['op'] == 'const':
        out_vals[instr['dest']] = instr['value']
      else:
        out_vals[instr['dest']] = '?'

  return out_vals


def cprop_merge(vals_list):
    out_vals = {}
    for vals in vals_list:
        for name, val in vals.items():
            if val == '?':
                out_vals[name] = '?'
            else:
                if name in out_vals:
                    if out_vals[name] != val:
                        out_vals[name] = '?'
                else:
                    out_vals[name] = val
    return out_vals

def list_dedup(lst):
  seen = set()
  new_lst = []
  for d in lst:
    t = tuple(d.items())
    if t not in seen:
      seen.add(t)
      new_lst.append(d)

  return new_lst


def reaching_def_transfer(blocks, name, in_set):
  block = blocks[name]
  var = set()
  defs = set()

  for i,instr in enumerate(block):
    # look for value ops
    if "op" in instr and "dest" in instr:
      var.add(instr["dest"])
      defs.add((name,i))

  # filter out other defs of variables in in_set
  out_set = filter(lambda (name,i): blocks[name][i]["dest"] not in var, in_set)
  out_set += defs

  return out_set


def list_union(lists):
  lst = []
  for l in lists:
    lst += l

  return list_dedup(lst)


ANALYSES = {
    # A really really basic analysis that just accumulates all the
    # currently-defined variables.
    'defined': Analysis(
        True,
        init=set(),
        merge=union,
        transfer=defined_transfer
    ),

    # Live variable analysis: the variables that are both defined at a
    # given point and might be read along some path in the future.
    'live': Analysis(
        False,
        init=set(),
        merge=union,
        transfer=live_transfer
    ),

    # A simple constant propagation pass.
    'cprop': Analysis(
        True,
        init={},
        merge=cprop_merge,
        transfer=cprop_transfer,
    ),

    # reaching definitions
    'rdef': Analysis(
        True,
        init=set(),
        merge=union,
        transfer=reaching_def_transfer
    )
}

if __name__ == '__main__':
    bril = json.load(sys.stdin)
    in_, out = run_df(bril["functions"][0]["instrs"], ANALYSES[sys.argv[1]])
    print(in_)
    print(out)

