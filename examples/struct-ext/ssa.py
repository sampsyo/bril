#!/usr/bin/python

import sys
import json
from dom import Dominators
from brilpy import *
from functools import reduce

TERM = 'jmp', 'br', 'ret'

def to_ssa(prog):
    for func in prog['functions']:

        # Add dummy id operations for each argument.
        # This is a bit of a hack because of the fact that you can reassign to
        # the args anywhere in the function.
        # We don't want the first block to be a place we can jump to, because
        # we can't have a phi as the first instruction to disambiguate args
        if 'args' in func:
            if func['args']:
                for a in func['args']:
                    func['instrs'] = [{'op':'id', 'args':[a['name']], 'type':a['type'], 'dest':a['name']}] + \
                                     func['instrs']
                func['instrs'] = [{'label':'pre_entry'}] + func['instrs']

        # Next we need to canonicalize labels, in case any labels appear
        # directly in a row, this would break things later
        label_last = False
        last = None
        i = 0
        while i < len(func['instrs']):
            inst = func['instrs'][i]
            if 'label' in inst:
                if label_last:
                    for j in func['instrs']:
                        if 'labels' in j and inst['label'] in j['labels']:
                            labels = []
                            for lbl in j['labels']:
                                if lbl == inst['label']:
                                    labels.append(last['label'])
                                else:
                                    labels.append(lbl)
                            j['labels'] = labels
                    func['instrs'].pop(i)

                else:
                    i += 1
                    last = inst
                    label_last = True
            else:
                i += 1
                label_last = False

        # This last bit is just because even after the above, a valid bril
        # program could end with a label, but we don't want that (i.e., an empty
        # block
        if label_last:
            func['instrs'].append({'op':'ret'});

        g = CFG(func)

        domins = Dominators(func)

        defs = {}
        for i,b in enumerate(g.blocks):
            if i == 0 and 'args' in func:
                for arg in func['args']:
                    defs[arg['name']] = [i]

            for instr in b:
                if 'dest' in instr:
                    if instr['dest'] in defs:
                        defs[instr['dest']].append(i)
                    else:
                        defs[instr['dest']] = [i]

        # for each block, these are the phis we'll add at the end. Each has a
        # map from orig.var -> to a map that will become the instruction itself,
        phis = []
        for i in range(g.n):
            phis.append({})

        # Following pseudocode from Lesson 5 notes
        # ``Step one''
        for v,vdefs in defs.items():
            for d in vdefs:
                for b in domins.frontier[d]:
                    if v not in phis[b]:
                        phis[b][v] = {'op':'phi', 'args':[], 'labels':[]} # will handle dest/args later

                    if b not in defs[v]:
                        defs[v].append(b)

        # ``Step two''
        stack = {}
        next_name = {}
        for v in defs.keys():
            stack[v] = []
            next_name[v] = 0

        # args' bottom-stack names are their original names
        if 'args' in func:
            for arg in func['args']:
                stack[arg['name']] = [arg['name']]


        def new_name(ogvar):
            n = ogvar + '_' + str(next_name[ogvar])
            next_name[ogvar] += 1
            stack[ogvar].append(n)
            return n

        # b: index of block
        def rename(b):


            # map from vars to count of names pushed (so we can pop them)
            push_count = {}

            for v,p in phis[b].items():
                p['dest'] = new_name(v)

            for instr in g.blocks[b]:

                # replace old names with stack names
                if 'args' in instr:
                    newargs = []
                    if 'op' in instr and instr['op'] == 'getmbr':
                        newargs.append(stack[instr['args'][0]][-1])
                        newargs.append(instr['args'][1])
                    else:
                        for arg in instr['args']:
                            newargs.append(stack[arg][-1])
                    instr['args'] = newargs

                # replace destination with new name (and push onto stack)
                if 'dest' in instr:
                    name = new_name(instr['dest'])
                    if instr['dest'] in push_count:
                        push_count[instr['dest']] += 1
                    else:
                        push_count[instr['dest']] = 1

                    instr['dest'] = name

            for s in g.edges[b]:

                for v in set(phis[s].keys()): # (copy keyset so we can remove)

                    # we found a path to this block where it is unassigned: this phi should go away
                    if not stack[v]: 
                        phis[s].pop(v)

                    # otherwise update the var-use to use the current name
                    else:
                        phis[s][v]['args'].append(stack[v][-1])
                        phis[s][v]['labels'].append(g.names[b])

            if b in domins.dom_tree:
                for b_dom in domins.dom_tree[b]:
                    rename(b_dom)

            # pop all the names
            for var,count in push_count.items():
                for j in range(count):
                    stack[var].pop()

        rename(0)


        # Add labels to blocks missing labels, and add jumps to blocks that fall
        # through
        for i,b in enumerate(g.blocks):
            # Add a label if missing
            if 'label' not in b[0]:
                b.insert(0, {'label': g.names[i]})

            for v,p in phis[i].items():
                # don't need a phi if only one label or arg
                if len(set(p['labels'])) > 1 and len(set(p['args'])) > 1: 
                    b.insert(1, p)

            # Add a jmp if missing
            if i > 0 and ('op' not in g.blocks[i-1][-1] or g.blocks[i-1][-1]['op'] not in TERM):
                g.blocks[i-1].append({'op':'jmp', 'labels':[b[0]['label']]})


        # Write all the blocks' instructions to a new "linear" function
        newinstrs = []
        for i,b in enumerate(g.blocks):
            newinstrs += b

        if 'op' not in newinstrs[-1] or newinstrs[-1]['op'] not in TERM:
            newinstrs.append({'op':'ret'});

        func['instrs'] = newinstrs

    return prog

def from_ssa(prog):

    for func in prog['functions']:

        g = CFG(func)

        # First compute a map from label -> block idx
        # Note: we assume every block in SSA form has a label (is this true?)
        block_by_label = {}
        term = []
        for i,b in enumerate(g.blocks):
            block_by_label[b[0]['label']] = i

            # also temporarily save the TERM instruction (so when we add id's we
            # can just tack them on the end)... this is a bit awkward
            if 'op' in b[-1] and b[-1]['op'] in TERM:
                term.append(b.pop())
            else:
                term.append(None)
        
        # print(term)

        for i,b in enumerate(g.blocks):
            if len(b) == 1:
                continue

            j = 1
            while j < len(b) and 'op' in b[j] and b[j]['op'] == 'phi':
                for k in range(len(b[j]['args'])):
                    inst = {'op': 'id', 'dest': b[j]['dest'],
                            'args':[b[j]['args'][k]]}
                    g.blocks[block_by_label[b[j]['labels'][k]]].append(inst)

                j += 1
        
        # write changes, omitting phis
        newinstr = []
        for i,b in enumerate(g.blocks):
            for inst in b:
                if not ('op' in inst and inst['op'] == 'phi'):
                    newinstr.append(inst)
            if term[i]:
                newinstr.append(term[i])

        func['instrs'] = newinstr

    return prog
