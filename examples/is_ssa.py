import json
import sys


def is_ssa(bril):
    for func in bril['functions']:
        assigned = set()
        for instr in func['instrs']:
            if 'dest' in instr:
                if instr['dest'] in assigned:
                    return 'no'
                else:
                    assigned.add(instr['dest'])
    return 'yes'


if __name__ == '__main__':
    print(is_ssa(json.load(sys.stdin)))
