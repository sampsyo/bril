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

func: FUNC ["(" arg_list? ")"] [tyann] "{" instr* "}"
arg_list: | arg ("," arg)*
arg: IDENT ":" type
?instr: const | vop | eop | label

const.4: IDENT [tyann] "=" "const" lit ";"
vop.3: IDENT [tyann] "=" op ";"
eop.2: op ";"
label.1: LABEL ":"

op: IDENT (FUNC | LABEL | IDENT)*

?tyann: ":" type

lit: SIGNED_INT  -> int
  | BOOL         -> bool
  | SIGNED_FLOAT -> float

type: IDENT "<" type ">" -> paramtype
    | IDENT              -> primtype

BOOL: "true" | "false"
IDENT: ("_"|"%"|LETTER) ("_"|"%"|"."|LETTER|DIGIT)*
FUNC: "@" IDENT
LABEL: "." IDENT
COMMENT: /#.*/

%import common.SIGNED_INT
%import common.SIGNED_FLOAT
%import common.WS
%import common.LETTER
%import common.DIGIT
%ignore WS
%ignore COMMENT
""".strip()


class JSONTransformer(lark.Transformer):
    def start(self, items):
        return {'functions': items}

    def func(self, items):
        name, args, typ = items[:3]
        instrs = items[3:]
        func = {
            'name': str(name)[1:],  # Strip `@`.
            'instrs': instrs,
        }
        if args:
            func['args'] = args
        if typ:
            func['type'] = typ
        return func

    def arg(self, items):
        name = items.pop(0)
        typ = items.pop(0)
        return {
            'name': name,
            'type': typ,
        }

    def arg_list(self, items):
        return items

    def const(self, items):
        dest, type, val = items
        out = {
            'op': 'const',
            'dest': str(dest),
            'value': val,
        }
        if type:
            out['type'] = type
        return out

    def vop(self, items):
        dest, type, op = items
        out = {'dest': str(dest)}
        if type:
            out['type'] = type
        out.update(op)
        return out

    def op(self, items):
        opcode = str(items.pop(0))

        funcs = []
        labels = []
        args = []
        for item in items:
            if item.type == 'FUNC':
                funcs.append(str(item)[1:])
            elif item.type == 'LABEL':
                labels.append(str(item)[1:])
            else:
                args.append(str(item))

        out = {'op': opcode}
        if args:
            out['args'] = args
        if funcs:
            out['funcs'] = funcs
        if labels:
            out['labels'] = labels
        return out

    def eop(self, items):
        op, = items
        return op

    def label(self, items):
        name, = items
        return {
            'label': str(name)[1:]  # Strip `.`.
        }

    def int(self, items):
        return int(str(items[0]))

    def bool(self, items):
        if str(items[0]) == 'true':
            return True
        else:
            return False

    def paramtype(self, items):
        return {items[0]: items[1]}

    def primtype(self, items):
        return str(items[0])

    def float(self, items):
        return float(items[0])


def parse_bril(txt):
    parser = lark.Lark(GRAMMAR, maybe_placeholders=True)
    tree = parser.parse(txt)
    data = JSONTransformer().transform(tree)
    return json.dumps(data, indent=2, sort_keys=True)


# Text format pretty-printer.

def type_to_str(type):
    if isinstance(type, dict):
        assert len(type) == 1
        key, value = next(iter(type.items()))
        return '{}<{}>'.format(key, type_to_str(value))
    else:
        return type


def instr_to_string(instr):
    if instr['op'] == 'const':
        tyann = ': {}'.format(type_to_str(instr['type'])) \
            if 'type' in instr else ''
        return '{}{} = const {}'.format(
            instr['dest'],
            tyann,
            str(instr['value']).lower(),
        )
    else:
        rhs = instr['op']
        if instr.get('funcs'):
            rhs += ' {}'.format(' '.join(
                '@{}'.format(f) for f in instr['funcs']
            ))
        if instr.get('args'):
            rhs += ' {}'.format(' '.join(instr['args']))
        if instr.get('labels'):
            rhs += ' {}'.format(' '.join(
                '.{}'.format(f) for f in instr['labels']
            ))
        if 'dest' in instr:
            tyann = ': {}'.format(type_to_str(instr['type'])) \
                if 'type' in instr else ''
            return '{}{} = {}'.format(
                instr['dest'],
                tyann,
                rhs,
            )
        else:
            return rhs


def print_instr(instr):
    print('  {};'.format(instr_to_string(instr)))


def print_label(label):
    print('.{}:'.format(label['label']))


def args_to_string(args):
    if args:
        return '({})'.format(', '.join(
            '{}: {}'.format(arg['name'], type_to_str(arg['type']))
            for arg in args
        ))
    else:
        return ''


def print_func(func):
    typ = func.get('type', 'void')
    print('@{}{}{} {{'.format(
        func['name'],
        args_to_string(func.get('args', [])),
        ': {}'.format(type_to_str(typ)) if typ != 'void' else '',
    ))
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
