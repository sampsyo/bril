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

GRAMMAR = r"""
start: (struct | func)*

struct: STRUCT IDENT "=" "{" mbr* "}"
mbr: IDENT ":" type ";"

func: FUNC ["(" arg_list? ")"] [tyann] "{" stmt* "}"
stmt: continue_ | instr | loop_ | if_ | block_ | break_
arg_list: | arg ("," arg)*
arg: IDENT ":" type
loop_: LOOP "{" stmt* "}"
if_: IF IDENT then_ [else_]
then_: THEN "{" stmt* "}"
else_: ELSE "{" stmt* "}"
block_: BLOCK "{" stmt* "}"
break_: BREAK lit ";"
continue_: CONTINUE lit ";"
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
  | "nullptr"    -> nullptr
  | CHAR         -> char

type: IDENT "<" type ">"  -> paramtype
    | IDENT               -> primtype

BOOL: "true" | "false"
STRUCT: "struct"
CHAR:  /'.'/ | /'\\[0abtnvfr]'/
IDENT: ("_"|"%"|LETTER) ("_"|"%"|"."|LETTER|DIGIT)*
FUNC: "@" IDENT
LABEL: "." IDENT
COMMENT: /#.*/
IF: "if"
THEN: "then"
ELSE: "else"
LOOP: "loop"
BREAK: "break"
CONTINUE: "continue"
BLOCK: "block"

%import common.SIGNED_INT
%import common.SIGNED_FLOAT
%import common.WS
%import common.LETTER
%import common.DIGIT
%ignore WS
%ignore COMMENT
""".strip()

control_chars = {
    '\\0': 0,
    '\\a': 7,
    '\\b': 8,
    '\\t': 9,
    '\\n': 10,
    '\\v': 11,
    '\\f': 12,
    '\\r': 13,
}


def _pos(token):
    """Generate a position dict from a Lark token."""
    return {'row': token.line, 'col': token.column}


class JSONTransformer(lark.Transformer):
    def __init__(self, include_pos=False):
        super().__init__()
        self.include_pos = include_pos

    def start(self, items):
        structs = [i for i in items if 'mbrs' in i]
        funcs = [i for i in items if 'mbrs' not in i]
        if structs:
            return {
                'structs': structs,
                'functions': funcs,
            }
        else:
            return {
                'functions': funcs,
            }

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
        if self.include_pos:
            func['pos'] = _pos(name)
        return func

    def stmt(self, items):
        return items.pop(0)

    def loop_(self, items):
        op = str(items[0])
        name = str(items[1])
        body = items[2:]
        return {
            'op': op,
            'args': [name],
            'children': [body],
        }

    def block_(self, items):
        op = str(items[0])
        body = items[1:]
        return {
            'op': op,
            'children': [body],
        }

    def break_(self, items):
        op = str(items[0])
        val = int(str(items[1]))
        return {
            'op': op,
            'value': val,
        }

    def continue_(self, items):
        op = str(items[0])
        val = int(str(items[1]))
        return {
            'op': op,
            'value': val,
        }

    def if_(self, items):
        op = str(items[0])
        arg = str(items[1])
        tru = items[2]
        fls = items[3]
        return {
            'op': op,
            'args': [arg],
            'children': [tru, fls],
        }

    def then_(self, items):
        return items[1:]
        return {
            'stmts': items[1:]
        }

    def else_(self, items):
        return items[1:]
        return {
            'stmts': items[1:],
        }

    def arg(self, items):
        name = items.pop(0)
        typ = items.pop(0)
        return {
            'name': name,
            'type': typ,
        }

    def struct(self, items):
        name = items[1]
        mbrs = items[2:]
        return {
            'name': name,
            'mbrs': mbrs,
        }

    def mbr(self, items):
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
        if self.include_pos:
            out['pos'] = _pos(dest)
        return out

    def vop(self, items):
        dest, type, op = items
        out = {'dest': str(dest)}
        if type:
            out['type'] = type
        out.update(op)
        if self.include_pos:
            out['pos'] = _pos(dest)
        return out

    def op(self, items):
        op_token = items.pop(0)
        opcode = str(op_token)

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
        if self.include_pos:
            out['pos'] = _pos(op_token)
        return out

    def eop(self, items):
        op, = items
        return op

    def label(self, items):
        name, = items
        out = {
            'label': str(name)[1:]  # Strip `.`.
        }
        if self.include_pos:
            out['pos'] = _pos(name)
        return out

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

    def nullptr(self, items):
        return 0

    def char(self, items):
        value = str(items[0])[1:-1]  # Strip `'`.
        if value in control_chars:
            return chr(control_chars[value])
        return value


def parse_bril(txt, include_pos=False):
    """Parse a Bril program and return a JSON string.

    Optionally include source position information.
    """
    parser = lark.Lark(GRAMMAR, maybe_placeholders=True)
    tree = parser.parse(txt)
    data = JSONTransformer(include_pos=include_pos).transform(tree)
    return json.dumps(data, indent=2, sort_keys=True)


# Text format pretty-printer.

def type_to_str(type):
    if isinstance(type, dict):
        assert len(type) == 1
        key, value = next(iter(type.items()))
        return '{}<{}>'.format(key, type_to_str(value))
    else:
        return type


def value_to_str(type, value):
    if not isinstance(type, dict) and type.lower() == "char":
        control_chars_reverse = {y: x for x, y in control_chars.items()}
        if ord(value) in control_chars_reverse:
            value = control_chars_reverse[ord(value)]
        return "'{}'".format(value)
    else:
        return str(value).lower()


def formatted_children(instrs, indentLevel):
    formatted = map(lambda x: instr_to_string(x, indentLevel), instrs)
    joined = '\n'.join(formatted)
    return joined


def loop_to_string(instr, indentLevel):
    indent = '  ' * indentLevel
    return '{}loop {{\n{}\n{}}}'.format(
        indent,
        formatted_children(instr['children'][0], indentLevel + 1),
        indent
    )


def if_to_string(instr, indentLevel):
    indent = '  ' * indentLevel
    if (len(instr['children']) == 1):
        return '{}if {}\n{}then{{\n {}\n {}}}'.format(
            indent,
            instr['args'][0],
            indent,
            formatted_children(instr['children'][0], indentLevel + 1),
            indent,
        )

    return '{}if {}\n{}then {{\n{}\n{}}}\n{}else {{\n{}\n{}}}'.format(
        indent,
        instr['args'][0],
        indent,
        formatted_children(instr['children'][0], indentLevel + 1),
        indent,
        indent,
        formatted_children(instr['children'][1], indentLevel + 1),
        indent,
    )


def block_to_string(instr, indentLevel):
    indent = '  ' * indentLevel
    return '{}block {{\n{}\n{}}}'.format(
        indent,
        formatted_children(instr['children'][0], indentLevel + 1),
        indent
    )


def instr_to_string(instr, indentLevel=1):
    indent = '  ' * indentLevel
    if instr['op'] == 'loop':
        return loop_to_string(instr, indentLevel)
    if instr['op'] == 'if':
        return if_to_string(instr, indentLevel)
    if instr['op'] == 'block':
        return block_to_string(instr, indentLevel)
    if instr['op'] == 'break':
        return '{}break {};'.format(
            indent,
            instr['value'] if instr['value'] else 0,
        )
    if instr['op'] == 'continue':
        return '{}continue {};'.format(
            indent,
            instr['value'] if instr['value'] else 0,
        )
    if instr['op'] == 'const':
        tyann = ': {}'.format(type_to_str(instr['type'])) \
            if 'type' in instr else ''
        return '{}{}{} = const {};'.format(
            indent,
            instr['dest'],
            tyann,
            value_to_str(instr['type'], instr['value']),
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
            return '{}{}{} = {};'.format(
                indent,
                instr['dest'],
                tyann,
                rhs,
            )
        else:
            return '{}{};'.format(indent, rhs)


def print_instr(instr):
    print('{}'.format(instr_to_string(instr)))


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
    print(parse_bril(sys.stdin.read(), '-p' in sys.argv[1:]))


def bril2txt():
    print_prog(json.load(sys.stdin))
