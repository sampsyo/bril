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

__version__ = '0.0.2'


# Text format parser.

GRAMMAR = """
start: (func | imp)*

imp: "import" CNAME modfunc* ";"
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
modfunc: CNAME | "(" CNAME ")" | "(" CNAME "as" CNAME ")"
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

    def imp(self, items):
        name = items.pop(0)
        functions = []

        while len(items) > 0 and type(items[0]) == lark.tree.Tree and \
            items[0].data == "modfunc":
            func = items.pop(0).children
            if len(func) == 2:
                functions.append({'name': func[0], 'alias': func[1]})
            elif len(func) == 1:
                functions.append({'name': func[0], 'alias': func[0]})

        data = {'import': str(name)}
        if len(functions):
            data['functionids'] = functions
        else:
            data['functionids'] = []
        return data

    def func(self, items):
        name = items.pop(0)
        args = []
        while len(items) > 0 and type(items[0]) == lark.tree.Tree and \
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

# Import dependencies of imported functions as well

def func_walk(imports, statements):
    dependencies = imports
    imported = []
    data = []
    while(dependencies):
        dependency = dependencies.pop(0)
        statement_pointer = 0

        while (statement_pointer != len(statements)):
            statement = statements[statement_pointer]
            if 'instrs' in statement and \
             dependency['name'] == statement['name'] and \
             dependency['alias'] not in imported:
                statement['name'] = dependency['alias']
                imported.append(statement['name'])
                data.append(statement)
                for instr in statement['instrs']:
                    if instr['op'] == 'call':
                        dependencies.append({'name': instr['args'][0], 'alias': instr['args'][0]})
                statement_pointer = len(statements)
            else:
                statement_pointer += 1
    return data

def unroll_imports(txt, modules):
    stage_one = parse_helper(txt)

    imports = {}
    for statement in stage_one['functions']:
        if "import" in statement.keys():
            imports[statement['import']] = statement['functionids']

    for imp in imports.keys():
        stage_one['functions'].pop(0)
        try:
            with open(modules['{}.bril'.format(imp)], 'r') as mod:
                mod_code = mod.read()
        except KeyError:
            print("Couldn't open module: {}".format(imp))
            return
        mod_data = unroll_imports(mod_code, modules)

        if imports[imp] != []:
            dependencies = func_walk(imports[imp], mod_data['functions'])
            stage_one['functions'] = stage_one['functions'] + dependencies
        else:
            stage_one['functions'] = stage_one['functions'] + mod_data['functions']
    
    return stage_one

def parse_bril(txt):
    data = parse_helper(txt)
    return json.dumps(data, indent=2, sort_keys=True)

def link_bril(main_module, module_paths):
    modules = {}
    try:
        with open(main_module, 'r') as main:
            program_txt = main.read()
    except EnvironmentError:
        print("Couldn't open main module")
        return

    for path in module_paths:
        head, tail = ntpath.split(path)
        modules[tail or ntpath.basename(head)] = path
    stage_two = unroll_imports(program_txt, modules)
    return json.dumps(stage_two, indent=2, sort_keys=True)

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
    json = link_bril(main_module, module_paths)
    if json and json != 'null':
        print(json)

def bril2json():
    json = parse_bril(sys.stdin.read())
    if json and json != 'null':
        print(json)

def bril2txt():
    print_prog(json.load(sys.stdin))
