"""A text format for Bril.

This module defines both a parser and a pretty-printer for a
human-editable representation of Bril programs. There are two commands:
`bril2txt`, which takes a Bril program in its (canonical) JSON format and
pretty-prints it in the text format, and `bril2json`, which parses the
format and emits the ordinary JSON representation.
"""

import lark
import sys
import json

__version__ = '0.0.1'


# Text format parser.

GRAMMAR = """
start: (legacy_func | func)*

legacy_func: CNAME "{" instr_list "}"
func: type CNAME "(" arg_list ")" "{" instr_list "}"

arg_list: | arg ("," arg)*

arg: IDENT ":" type

param_list: | IDENT ("," IDENT)*

instr_list: instr*

?instr: call | const | vop | eop | label 

const.5: IDENT ":" type "=" "const" lit ";"
vop.4: IDENT ":" type "=" CNAME IDENT* ";"
eop.3: CNAME IDENT* ";"
call.2: [IDENT ":" type "="] "call" IDENT "(" param_list ")" ";"
label.1: IDENT ":"

lit: SIGNED_INT  -> int
  | BOOL     -> bool

type: CNAME
BOOL: "true" | "false"
IDENT: ("_"|"%"|LETTER) ("_"|"%"|"."|LETTER|DIGIT)*
COMMENT: /#.*/

%import common.SIGNED_INT
%import common.WS
%import common.CNAME
%import common.LETTER
%import common.DIGIT
%ignore WS
%ignore COMMENT
""".strip()


class JSONTransformer(lark.Transformer):
    def start(self, items):
        return {'functions': items}

    def legacy_func(self, items):
        name = items.pop(0)
        instr_list = items.pop(0)
        return {
            'name': str(name),
            'args': [],
            'instrs': instr_list,
        }

    def func(self, items):
        typ = items.pop(0)
        name = items.pop(0)
        arg_list = items.pop(0)
        instr_list = items.pop(0)
        if typ == 'void':
            return {
                'name': str(name),
                'args': arg_list,
                'instrs': instr_list,
            }
        else:
            return {
                'name': str(name),
                'type': typ,
                'args': arg_list,
                'instrs': instr_list,
            }

    def arg_list(self, items):
        return items

    def arg(self, items):
        name = items.pop(0)
        typ = items.pop(0)
        return {
            'name': name,
            'type': typ,
        }

    def instr_list(self, items):
        return items

    def call(self, items):
        obj = {}
        if len(items) == 4:
            dest = items.pop(0)
            typ = items.pop(0)
            obj.update({
                'dest': str(dest),
                'type': typ,
            })

        name = items.pop(0)
        args = items.pop(0)
        obj.update({
            'op': 'call',
            'name': name,
            'args': args,
        })
        return obj

    def param_list(self, items):
        return items

    def const(self, items):
        dest = items.pop(0)
        type = items.pop(0)
        val = items.pop(0)
        return {
            'op': 'const',
            'dest': str(dest),
            'type': type,
            'value': val,
        }

    def vop(self, items):
        dest = items.pop(0)
        type = items.pop(0)
        op = items.pop(0)
        return {
            'op': str(op),
            'dest': str(dest),
            'type': type,
            'args': [str(t) for t in items],
         }

    def eop(self, items):
        op = items.pop(0)
        return {
            'op': str(op),
            'args': [str(t) for t in items],
         }

    def label(self, items):
        name = items.pop(0)
        return {
            'label': name,
        }

    def int(self, items):
        return int(str(items[0]))

    def bool(self, items):
        if str(items[0]) == 'true':
            return True
        else:
            return False

    def type(self, items):
        return str(items[0])


def parse_bril(txt):
    # Strip comments (must be entire line) from Bril before parsing
    lines = [l for l in txt.split("\n") if not l.strip().startswith("//")]
    txt = "\n".join(lines)
    parser = lark.Lark(GRAMMAR)
    tree = parser.parse(txt)
    data = JSONTransformer().transform(tree)
    return json.dumps(data, indent=2, sort_keys=True)


# Text format pretty-printer.

def instr_to_string(instr):
    if instr['op'] == 'const':
        return '{}: {} = const {}'.format(
            instr['dest'],
            instr['type'],
            str(instr['value']).lower(),
        )
    elif instr['op'] == 'call':
        if 'dest' in instr:
            return '{}: {} = {} {}({})'.format(
                instr['dest'],
                instr['type'],
                instr['op'],
                instr['name'],
                ', '.join(instr['args']),
            )
        else: 
            return '{} {}({})'.format(
                instr['op'],
                instr['name'],
                ', '.join(instr['args']),
            )
    elif 'dest' in instr:
        return '{}: {} = {} {}'.format(
            instr['dest'],
            instr['type'],
            instr['op'],
            ' '.join(instr['args']),
        )
    else:
        return '{} {}'.format(
            instr['op'],
            ' '.join(instr['args']),
        )


def print_instr(instr):
    print('  {};'.format(instr_to_string(instr)))


def print_label(label):
    print('{}:'.format(label['label']))

def print_args(args):
    return '{}'.format(', '.join(['{} : {}'.format(arg['name'], arg['type']) for arg in args]))


def print_func(func):
    typ = func['type'] if 'type' in func else 'void'
    print('{} {}({}) {{'.format(typ, func['name'], print_args(func['args'])))
    for instr_or_label in func['instrs']:
        if 'label' in instr_or_label:
            print_label(instr_or_label)
        else:
            print_instr(instr_or_label)
    print('}')


def print_prog(prog):
    for func in prog['functions']:
        print_func(func)


# Command-line entry points.

def bril2json():
    print(parse_bril(sys.stdin.read()))


def bril2txt():
    print_prog(json.load(sys.stdin))
