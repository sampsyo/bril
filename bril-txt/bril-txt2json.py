import lark
import sys
import json

GRAMMAR = """
start: func*

func: CNAME "{" instr* "}"

instr: IDENT "=" "const" lit        -> const
  | IDENT "=" CNAME IDENT*          -> op
  | IDENT ":"                       -> label

lit: NUMBER | "true" | "false"

IDENT: ("_"|"%"|LETTER) ("_"|"%"|LETTER|DIGIT)*

%import common.NUMBER
%import common.WS
%import common.CNAME
%import common.LETTER
%import common.DIGIT
%ignore WS
""".strip()


class JSONTransformer(lark.Transformer):
    def start(self, items):
        return {'functions': items}

    def func(self, items):
        name = items.pop(0)
        return {'name': str(name), 'instrs': items}

    def const(self, items):
        dest = items.pop(0)
        val = items.pop(0)
        return {
            'op': 'const',
            'dest': str(dest),
            'value': str(val),
        }

    def op(self, items):
        dest = items.pop(0)
        op = items.pop(0)
        return {
            'op': str(op),
            'dest': str(dest),
            'args': [str(t) for t in items],
         }

    def label(self, items):
        name = items.pop(0)
        return {
            'label': name,
        }

    def lit(self, items):
        return str(items[0])


def parse_bril(txt):
    parser = lark.Lark(GRAMMAR)
    tree = parser.parse(txt)
    data = JSONTransformer().transform(tree)
    return json.dumps(data, indent=2)


if __name__ == '__main__':
    print(parse_bril(sys.stdin.read()))
