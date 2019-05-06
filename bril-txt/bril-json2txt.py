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


def print_label(label):
    print('{}:'.format(label['label']))


def print_func(func):
    print('{} {{'.format(func['name']))
    for instr_or_label in func['instrs']:
        if 'label' in instr_or_label:
            print_label(instr_or_label)
        else:
            print_instr(instr_or_label)
    print('}')


def print_prog(prog):
    for func in prog['functions']:
        print_func(func)


if __name__ == '__main__':
    print_prog(json.load(sys.stdin))
