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
start: func*

func: CNAME "{" instr* "}"

?instr: micro | label | group

?micro: const | vop | eop

group: "[" IDENT* ":" micro+ ":" label "]"

const.4: IDENT ":" type "=" "const" lit ";"
vop.3: IDENT ":" type "=" CNAME IDENT* ";"
eop.2: CNAME IDENT* ";"
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

    def func(self, items):
        name = items.pop(0)
        return {'name': str(name), 'instrs': items}

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

    def group(self, items):
        conds = items.pop(0)
        instrs = items.pop(0)
        label = items.pop(0)
        return {
            "conds": conds,
            "instrs": instrs,
            failLabel: label,
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
    parser = lark.Lark(GRAMMAR)
    tree = parser.parse(txt)
    data = JSONTransformer().transform(tree)
    return json.dumps(data, indent=2, sort_keys=True)


# Text format pretty-printer.

def micro_to_string(micro):
    if micro['op'] == 'const':
        return '{}: {} = const {}'.format(
            micro['dest'],
            micro['type'],
            str(micro['value']).lower(),
        )
    elif 'dest' in micro:
        return '{}: {} = {} {}'.format(
            micro['dest'],
            micro['type'],
            micro['op'],
            ' '.join(micro['args']),
        )
    else:
        return '{} {}'.format(
            micro['op'],
            ' '.join(micro['args']),
        )


def instr_to_string(instr):
    if 'instrs' in instr:
        # print(instr)
        cond_strs = " ".join(instr['conds'])
        instr_strs = " ".join(map(lambda s: f"{micro_to_string(s)};", instr['instrs']))
        failLabel = instr['failLabel']
        return f"[{cond_strs} : {instr_strs} : {failLabel}]"
    else:
        return f"{micro_to_string(instr)};"


def print_instr(instr):
    print('  {}'.format(instr_to_string(instr)))


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


# Command-line entry points.

def bril2json():
    print(parse_bril(sys.stdin.read()))


def bril2txt():
    print_prog(json.load(sys.stdin))
