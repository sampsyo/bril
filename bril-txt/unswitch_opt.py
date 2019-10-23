"""
Sameer Lal, sjl328.
CS 6120 Graduate Compilers.

Performs loop unswitching optimization on natural loops.

"""
from collections import ChainMap
from form_blocks import form_blocks, TERMINATORS
from cfg_dot import *
import cfg
import json
import sys
import dom
import util
import copy
from collections import OrderedDict




def create_jump(dest):
    instr = { "args": [ dest ], "op": "jmp" }
    return instr

def cfg_edges(bril):
    """
        Generate list of edges in CFG.
    """
    edge_list = []
    for func in bril['functions']:
        blocks = block_map(form_blocks(func['instrs']))
        # Insert terminators into blocks that don't have them.
        for i, block in enumerate(blocks.values()):
            if block[-1]['op'] not in TERMINATORS:
                if i == len(blocks) - 1:
                    # In the last block, return.
                    block.append({'op': 'ret', 'args': []})
                else:
                    # Otherwise, jump to the next block.
                    dest = list(blocks.keys())[i + 1]
                    block.append({'op': 'jmp', 'args': [dest]})
        # Add the control-flow edges.
        for i, (name, block) in enumerate(blocks.items()):
            succ = successors(block[-1])
            for label in succ:
                edge_list.append((name, label))
        return edge_list


def find_backedges(cfg_edges, dom_dict):
    """
    Baby:  Returns a list of backedges in a CFG
    Adv: TODO: return only natural loops (i.e nonnested loops)
    """
    # Iterate through all edges (u,v) and see if v is dominated by u
    back_edges = []
    for e in cfg_edges:
        head, tail = e
        if tail in dom_dic[head]:
            back_edges.append(e)
    return back_edges

def loop_finder(bril, cfg_edges, dom_dic):
    """
    Looks at CFG and outputs blocks that form a loop
    
    """
    loop_blocks = []
    back_edges = find_backedges(cfg_edges, dom_dic)

    #   Essentially we look for a backedge whose tail is dominated by the head
    #   and then we backtrack by populating a stack and predecessors
    #   to extract the loop body.
    for e in back_edges:
        n, h = e
        body = [h]
        stack = []
        stack.append(n)
        while( len(stack) ):
            d = stack.pop()
            if d not in body:
                body.append(d)
                # Search for predecessors
                # TODO optimize this using hash table in case of big graphs
                for ind, ele in enumerate(cfg_edges):
                    a, b = ele                    
                    if b == d:
                        stack.append(a)
        loop_blocks.append(body)
    return h, loop_blocks


def extract_variables_loop(bril, potential_loop, block_map):
    """
        Extract variables that are defined/modified in a loop excluding
        terminators.
    """
    all_variables = []
    # Obtain all variables
    
    for b in potential_loop:
        vb = [] #variables in one block
        block = block_map[b]
        for inst in block:
            vb = vb + util.var_loop(inst)
        all_variables = all_variables +  vb
    all_variables = set(all_variables)
    return set(all_variables)

def is_switchable(bril, potential_loop, block_map):
    """
        Returns array of blocks that are switchable within a loop.
        Returns [] if none exist
    """

    switchable_blocks = []
    loop_variables = extract_variables_loop(bril, potential_loop, block_map)
    for b in potential_loop:
        block = block_map[b]
        term_instr = block[-1]
        if term_instr['op'] == 'br':
            vt = term_instr['args'][0]
            if vt not in loop_variables:
                switchable_blocks.append(b)
                break
    return switchable_blocks, term_instr
    

def get_blocks_map(bril):
    for func in bril['functions']:
        blocks = block_map(form_blocks(func['instrs']))
    return blocks


def reorder_cfg_pre_if(header_block, block_list, edges):
    ordered = []
    for e in edges:
        h, t = e
        if h == header_block and t in block_list:
            ordered.append(t)
            header_block = t
    # TODO: implement
    return ordered

def reorder(bril, header, potential_loop, block_map, term_instr, dom_dic, edges):
    """
        Perform the loop unswitching.  Return dictionary
        of overwritten blocks in CFG
    """
    
    
    # The idea is that we will create a new dictionary that should
    # overright block names in the original block mapping dictionary
    # We will return this dictionary and merge the two, giving this
    # dictionary higher priority, which will alter the CFG and thus
    # the program itself

    new_block_map = {}

    # Identify statement "t" that performs the unconditional transfer
    # We need to modify the branches so it points to "if_loop_logic"
    # and "else_loop_logic"
    t = term_instr
    
    ## ---------- Create new block names ----------------
    
    t_args = t.get('args')
    t_args_var = t_args[0]
    
    old_if = t_args[1]
    old_else = t_args[2]

    if_hash = str(hash(old_if) % 100)
    else_hash = str(hash(old_else) %100)
    
    # Create if/else loop logic block names
    if_ll = 'if_loop_logic' + if_hash
    else_ll = 'else_loop_logic' + else_hash

    # Create if/else body block names
    if_bb = 'if_body_block' + if_hash
    else_bb = 'else_body_block' + else_hash

    if_jmp = 'if_jmp' + if_hash
    else_jmp = 'else_jmp' + else_hash

    if_by = 'if_bypass' + if_hash
    else_by = 'else_bypass' + else_hash    
    

    ## ---------- Create header block ----------------
    # Create first block with same name but different instructions
    header_block = header # First block is always head 

    # New Header block should just consist of the case statement
    t['args'] = [t.get('args')[0], if_ll, else_ll]
    new_block_map[header_block] = [t]

    
    ## ---------- Create loop logic blocks -----------------
    # Create for loop logic block. This will all be contained in 
    # the header block which encoded whether we should continue
    # looping or not.
    loop_logic = block_map[header_block]
    loop_exit_label = loop_logic[-1]['args'][2]
    
    # We need one copy for if and one copy for else
    if_loop_logic = copy.deepcopy(loop_logic)
    else_loop_logic = copy.deepcopy(loop_logic)

    # Modify exit nodes for each
    # if -> if_bb, if_by
    # else -> else_bb, else_by
    if_cond = if_loop_logic[-1]['args'][0]
    else_cond = else_loop_logic[-1]['args'][0]

    if_loop_logic[-1]['args'] = [if_cond, if_bb, if_by]
    else_loop_logic[-1]['args'] = [else_cond, else_bb, else_by]

    new_block_map[if_ll] = if_loop_logic
    new_block_map[else_ll] = else_loop_logic

    ## ---------- Create the body blocks -----------------

    # Contents should include stuff that dominates the block
    # and is dominated by the big loop block

    before_body = []
    # Body before = loop body before the if/else statement
    # To figure this out, we essentially look for a block
    # that is dominated by the start of the for loop
    # and that dominates BOTH the "if" and "else" branch
    
    # Blocks that domiante both If and Else (set intersection)
    # as a check, this should always equal the union
    dominated_if_else = set(dom_dic[ old_if ]).intersection(
        set(dom_dic[ old_else])
        )
    # TODO: I think there may be a bug here if there are multiple and it
    # adds it out of order ... look into this when evaluating
    for hblock in dominated_if_else:
        if hblock != header_block and header_block in dom_dic[hblock]:
            before_body.append(hblock) 
    
    
    # We need to sort before_body according to CFG.
    # Proof: everything is dominated by header, so we start with that
    before_body = reorder_cfg_pre_if(header_block, before_body, edges)
    
    

    # IF::
    flag = False
    if_bod = []
    for i, b in enumerate(before_body):    
        if flag:
            if_bod += [ { "label" : str(b) + if_hash } ]
        flag = True
        t = copy.deepcopy(block_map[b])
        if_bod += t
        # Change last branching if jump
        # TODO: All terminator cases
        if if_bod[-1]['op'] == 'jmp':
            # Change branch to hashed version
            if_bod[-1]['args'] = [ if_bod[-1]['args'][0] + if_hash ]
        


    # Delete last branching instruction before going to if/else
    if_bod = copy.deepcopy(if_bod[:-1])
    
    

    # ELSE::
    flag = False
    el_bod = []
    for i, b in enumerate(before_body):
        if flag:
            el_bod += [ { "label" : str(b) + else_hash } ]
        flag = True
        t = copy.deepcopy(block_map[b])
        el_bod += t
        # TODO: for all terminators
        if el_bod[-1]['op'] == 'jmp':
            # Change branch to hashed version
            el_bod[-1]['args'] = [ el_bod[-1]['args'][0] + else_hash ]

    # Delete last branching instruction before going to if/else
    el_bod = copy.deepcopy(el_bod[:-1])


    ## Old block contents should be added to respective block
    # This is the portion of the loop that is selectively executed 
    old_if_contents = block_map[old_if][:-1]
    old_else_contents = block_map[old_else][:-1]


    ## Blocks after if/else should be added as well.
    # We add blocks to "if_bb" if they are dominated by if_bb
    if_dom = []
    for lp in potential_loop:
        if old_if in dom_dic[lp] and lp != old_if:
            if_dom.append(lp[:-1])
    # print(if_dom)
    
    # We add blocks to "else_bb" if they are dominated by if_bb 
    else_dom = []
    for lp in potential_loop:
        if old_else in dom_dic[lp] and lp != old_else:
            else_dom.append(lp[:-1])
    # print(else_dom)

    ## Finally, we create the last block that is shared
    last_block = potential_loop[1] # this is the backedge to the loop
    # in particular, it will always be the second element of potential_loop
    # breaking this invariant means that there is not a loop

    last_contents = block_map[last_block]
    jump_instr = last_contents[-1]

    # Surgery on last instruction
    last_contents = last_contents[:-1]

    # Modify last instruction jump
    if_last_instr = copy.deepcopy(jump_instr)
    else_last_instr = copy.deepcopy(jump_instr)

    if_last_instr['args'] = [if_jmp]
    else_last_instr['args'] = [else_jmp]
   
    
    ## Now combine
    if_bb_body = if_bod + old_if_contents  \
        + if_dom + last_contents + [if_last_instr]
    
    else_bb_body = el_bod + old_else_contents \
         + else_dom + last_contents + [else_last_instr]


    new_block_map[if_bb] = if_bb_body
    new_block_map[else_bb] = else_bb_body    

    # -------- Creating jump blocks -----------

    # This is easy.  We just add a jump instruction
    new_block_map[if_jmp] = [create_jump(if_ll)]
    new_block_map[else_jmp] = [create_jump(else_ll)]

    # --------- Creating bypass blocks ----------
    new_block_map[if_by] = [create_jump(loop_exit_label)]
    new_block_map[else_by] = [create_jump(loop_exit_label)]


    # # Debug:
    # for k in new_block_map:
        # print('--> ', k, ' ', new_block_map[k], '\n \n')
    
    return new_block_map
def merge_blocks(old_map, loop_blocks, new_map):
    """
        Take in old map and new optimized map and 
        returns new bril program mapping.

        CAUTION: We need to be very careful about
        keeping the ordering invariant.   This is for two reasons
        (1) we can mess up the flow of the assembly code and cause the program
        to not end, and
        (2) there are other optimizations that take into account block locality.
    """
    
    flag = True
    opt = {}
    for k in old_map:
        if k in new_map:
            if flag:
                flag = False
                for n in new_map:
                    opt[n] = new_map[n]
        else: 
            opt[k] = old_map[k]
    return opt


    return dict(ChainMap(new_map, old_map))

def generate_json(oldjson, opt_map):
    '''
        TODO:
    '''
    out = {}
    # Default beginning
    instructions = opt_map['b1']
    for k in opt_map:
        if k != 'b1':
            instructions += [ { "label" : str(k)  } ]
            instructions += opt_map[k]
    # TODO: make this less hacky
    for fun in oldjson:
        junk = oldjson[fun]
        for j in junk:
            j['instrs'] = instructions
    
    # Hacky fix .. figure out why this is weird.
    oldjson = str(oldjson)
    oldjson = oldjson.replace("'", "\"")
    oldjson = oldjson.replace("True", "true") 
    oldjson = oldjson.replace("False","false")
    return oldjson

if __name__ == '__main__':
    f = sys.stdin
    bril = json.load(f)
    blocks_map = get_blocks_map(bril)
    edges = cfg_edges(bril)
    dom_dic = dom.get_dom_dict(bril)
    header, potential_loop = loop_finder(bril, edges, dom_dic)
    
    print(potential_loop)

    bl, term_instr = is_switchable(bril, potential_loop[0], blocks_map)

    if not bl:
        print('no optimizations available')
    else:        
        opt_map = reorder(bril, header, potential_loop[0], blocks_map, term_instr, dom_dic, edges)
        opt_map =  merge_blocks(blocks_map, potential_loop[0], opt_map)
        js = generate_json(bril, opt_map)

        
        with open('../test/opt_tests/optimizedversion.json', 'w') as outfile:
            outfile.write(js)

        # Debugging:
        # for k in blocks_map:
        #     print(' ~~> ', k, ' : ', blocks_map[k])

        # for k in opt_map:
        #     print(' ==> ', k, ' : ', opt_map[k])
    
    


    
    
