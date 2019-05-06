import json
import sys


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
    print('{} {{'.format(func['name']))
    for instr in func['instrs']:
        print_instr(instr)
    print('}')


def print_prog(prog):
    for func in prog['functions']:
        print_func(func)


if __name__ == '__main__':
    print_prog(json.load(sys.stdin))
