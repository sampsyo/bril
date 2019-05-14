import json
import sys
from collections import defaultdict


def print_instr(instr):
    if instr['op'] == 'const':
        print('  {} = const {}'.format(
            instr['dest'],
            instr['value'],
        ))
    else:
        print('  {} = {} {}'.format(
            instr['dest'],
            instr['op'],
            ' '.join(instr['args']),
        ))


def print_func(func):
    print('digraph {} {{'.format(func['name']))

    # Find the basic blocks and the successor edges.
    blocks = defaultdict(list)
    succs = defaultdict(list)
    cur_block = 'entry'
    for instr_or_label in func['instrs']:
        if 'label' in instr_or_label:
            cur_block = instr_or_label['label']
        else:
            blocks[cur_block].append(instr_or_label)
            if instr_or_label['op'] == 'jmp':
                succs[cur_block].append(instr_or_label['args'][0])
            elif instr_or_label['op'] == 'br':
                succs[cur_block] += instr_or_label['args'][1:]

    # Print out the edges.
    for src, dests in succs.items():
        for dest in dests:
            print('  {} -> {};'.format(src, dest))

    print('}')


def print_prog(prog):
    for func in prog['functions']:
        print_func(func)


if __name__ == '__main__':
    print_prog(json.load(sys.stdin))
