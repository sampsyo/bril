from hypothesis import *
from hypothesis.strategies import *

from briltxt import *

import io
from contextlib import redirect_stdout

names = text(alphabet=characters(min_codepoint=97, max_codepoint=122), 
             min_size=3,
             max_size=3)

types = sampled_from(["int", "bool"])

@composite 
def function_type(draw):
    return draw(sampled_from([draw(types), "void"]))

@composite
def bril_effect_instr(draw):
    opcode = draw(names)
    args = draw(lists(names, max_size=3))
    return {
        "op": opcode,
        "args": args}

@composite
def bril_value_instr(draw):
    opcode = draw(names)
    args = draw(lists(names, max_size=3))
    dest = draw(names)
    typ = draw(types)
    return {
        "op": opcode,
        "args": args,
        "dest": dest,
        "type": typ}

@composite
def bril_constant_instr(draw):
    typ = draw(types)
    dest = draw(names)
    if (typ == "int"):
        value = draw(sampled_from(range(100)))
    elif (typ == "bool"):
        value = draw(sampled_from(["true", "false"]))
    return {
        "op": "const",
        "value": value,
        "dest": dest,
        "type": typ}

@composite
def bril_instr(draw):
    return draw(sampled_from([
        draw(bril_effect_instr()),
        draw(bril_constant_instr()),
        draw(bril_constant_instr()),
        ]))

@composite
def bril_function(draw):
    name = draw(names)
    args = [] # TODO
    instrs = draw(lists(bril_instr(), max_size=5))
    typ = draw(function_type())
    if (typ != "void"):
        return  {
          "name": name,
          "instrs": instrs,
          "args": args,
          "type": typ,
        }
    else:
        return  {
          "name": name,
          "instrs": instrs,
          "args": args,
        }

@composite
def bril_program(draw):
    main = draw(bril_function())
    main["name"] = "main"

    all_funs = [main] + draw(lists(bril_function(), max_size=2))

    return {
      "functions": all_funs,
    }

add_prog = {
    "functions": [
        {
          "name": "main",
          "instrs": [
            { "op": "const", "type": "int", "dest": "v0", "value": 1 },
            { "op": "const", "type": "int", "dest": "v1", "value": 2 },
            { "op": "add", "type": "int", "dest": "v2",
              "args": ["v0", "v1"] },
            { "op": "print", "args": ["v2"] }
          ],
          "args": []
        }
      ]
    }

@example(prog=add_prog)
@given(bril_program())
def test_json_to_text_compile(prog):
    f = io.StringIO()
    with redirect_stdout(f):
        print_prog(prog)
    print_prog_output = f.getvalue()
    json_output = json.loads(parse_bril(print_prog_output))
    assert json_output == prog
