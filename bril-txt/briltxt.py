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
import ntpath
import argparse

__version__ = '0.0.1'


# Text format parser.

GRAMMAR = """
start: imp* func*

imp: "import" CNAME ";"
func: CNAME arg* "{" instr* "}" | CNAME arg* ":" type "{" instr* "}"

?instr: const | vop | eop | label

const.4: IDENT ":" type "=" "const" lit ";"
vop.3: IDENT ":" type "=" CNAME IDENT* ";"
eop.2: CNAME IDENT* ";"
label.1: IDENT ":"

lit: SIGNED_INT  -> int
  | BOOL     -> bool

type: CNAME
arg: IDENT | "(" IDENT ":" type ")"
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
        data = {'functions': []}
        imports = []
        for item in items:
            if 'import' in item:
                imports.append(item)
            elif 'name' in item:
                data['functions'].append(item)
            else:
                raise Exception('Unknown statement')
        if len(imports) > 0:
            data['imports'] = imports
        return data

    def imp(self, items):
        name = items.pop(0)
        data = {'import': str(name)}
        return data

    def func(self, items):
        name = items.pop(0)
        args = []
        while len(items) and type(items[0]) == lark.tree.Tree and \
            items[0].data == "arg":
            arg = items.pop(0).children
            args.append(
                dict(name=arg[0], type=arg[1] if len(arg) > 1 else None))
        
        function_type = items.pop(0) if type(items[0]) == str else None
        data = {'name': str(name), 'instrs': items}
        if len(args):
            data['args'] = args
        if function_type is not None:
            data['type'] = function_type
        return data

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

def parse_helper(txt):
    parser = lark.Lark(GRAMMAR)
    tree = parser.parse(txt)
    data = JSONTransformer().transform(tree)
    return data

def unroll_imports(txt, modules, imported):
    stage_one = parse_helper(txt)

    for imp in stage_one.get('imports', list()):
        if imp['import'] not in imported:
            try:
                with open(modules['{}.bril'.format(imp['import'])], 'r') as mod:
                    mod_code = mod.read()
                    imported.append(imp['import'])
            except:
                print("Couldn't find module: {}".format(imp['import']), file=sys.stderr)
                sys.stderr.flush()
                sys.exit(1)
            mod_data = unroll_imports(mod_code, modules, imported)
            stage_one['functions'] = stage_one['functions'] + mod_data['functions']

    if 'imports' in stage_one:
        stage_one.pop('imports')
    return stage_one

def parse_bril(txt, module_paths):
    modules = {}
    for path in module_paths:
        head, tail = ntpath.split(path)
        modules[tail or ntpath.basename(head)] = path
    stage_two = unroll_imports(txt, modules, [])
    if stage_two:
        return json.dumps(stage_two, indent=2, sort_keys=True)
    else:
        return None

# Text format pretty-printer.

def instr_to_string(instr):
    if instr['op'] == 'const':
        return '{}: {} = const {}'.format(
            instr['dest'],
            instr['type'],
            str(instr['value']).lower(),
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


def print_func(func):
    print('{} {{'.format(func['name'], func.get('type', 'void')))
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

def linkbril():
    main_module = sys.argv[1]
    module_paths = sys.argv[2:]
    try:
        with open(main_module, 'r') as main:
            program_txt = main.read()
    except EnvironmentError:
        print("Couldn't find main module", file=sys.stderr)
        return

    json = parse_bril(program_txt, module_paths)
    if json:
        print(json)

def bril2json():
    module_paths = sys.argv[1:]
    json = parse_bril(sys.stdin.read(), module_paths)
    if json:
        print(json)

def bril2txt():
    print_prog(json.load(sys.stdin))
