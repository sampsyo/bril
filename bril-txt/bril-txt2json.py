import lark
import sys
import json

GRAMMAR = """
start: func*

func: CNAME "{" instr* "}"

instr: VAR "=" "const" NUMBER
  | VAR "=" CNAME VAR*

VAR: ("_"|"%"|LETTER) ("_"|"%"|LETTER|DIGIT)*

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

    def instr(self, items):
        dest = items.pop(0)
        op = items.pop(0)
        return {
            'op': str(op),
            'dest': str(dest),
            'args': [str(t) for t in items],
         }


def parse_bril(txt):
    parser = lark.Lark(GRAMMAR)
    tree = parser.parse(txt)
    data = JSONTransformer().transform(tree)
    return json.dumps(data, indent=2)


if __name__ == '__main__':
    print(parse_bril(sys.stdin.read()))
