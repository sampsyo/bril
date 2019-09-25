from hypothesis import *
from hypothesis.strategies import *

from briltxt import *

import io
import subprocess
from contextlib import redirect_stdout

# names = text(alphabet=characters(min_codepoint=97, max_codepoint=122), 
#              min_size=1,
#              max_size=1)

names = text(alphabet=characters(min_codepoint=97, max_codepoint=100), 
             min_size=1,
             max_size=1)

types = sampled_from(["int", "bool"])

@composite
def opcodes(draw):
    return draw(sampled_from(["print", draw(names)]))

@composite 
def function_type(draw):
    return draw(sampled_from([draw(types), "void"]))

@composite
def bril_effect_instr(draw):
    opcode = draw(opcodes())
    if (opcode == "print"):
        args = draw(names)
    else:
        args = draw(lists(names, min_size=1, max_size=3))
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
        value = draw(sampled_from([True, False]))
    return {
        "op": "const",
        "value": value,
        "dest": dest,
        "type": typ}

@composite
def bril_call_instr(draw):
    name = draw(names)
    args = draw(lists(names, max_size=3))
    typ = draw(function_type())
    if (typ != "void"):
        dest = draw(names)
        return  {
          "op": "call",
          "name": name,
          "args": args,
          "dest": dest,
          "type": typ,
        }
    else:
        return  {
          "op": "call",
          "name": name,
          "args": args,
        }

@composite
def bril_instr(draw):
    return draw(sampled_from([
        draw(bril_effect_instr()),
        draw(bril_constant_instr()),
        draw(bril_constant_instr()),
        draw(bril_call_instr()),
        ]))

@composite
def bril_arg(draw):
    name = draw(names)    
    typ = draw(types)
    return {
        "name" : name,
        "type" : typ,
    }

@composite
def bril_function(draw):
    name = draw(names)
    args = draw(lists(bril_arg(), max_size=3))
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

test_prog_no_dest = {
    "functions": [
        {
            "args": [],
            "instrs": [
                { "args": [], "dest": "aaa", "name": "aaa", "type": "int"}
            ],
            "name": "main",
            "type": "int"
        }
      ]
    }

test_prog = {'functions': [{'args': [],
   'instrs': [{'dest': 'v0', 'op': 'const', 'type': 'int', 'value': 1},
    {'dest': 'v1', 'op': 'const', 'type': 'int', 'value': 2},
    {'args': ['v0', 'v1'], 'dest': 'v2', 'op': 'add', 'type': 'int'},
    {'args': ['v2'], 'op': 'print'}],
   'name': 'main'}]}

# # @example(prog=add_prog)
# @given(bril_program())
# @settings(verbosity=Verbosity.debug, max_examples=200)
# def test_json_to_bril_txt(prog):
# # def test_json_to_bril_txt():
# #     prog = test_prog

#     with open("progs", "a") as f:
#         f.write(json.dumps(prog, indent=4))
#         f.write("\n")

#     f = io.StringIO()
#     with redirect_stdout(f):
#         print_prog(prog)
#     print_prog_output = f.getvalue()

#     with open("bril_progs", "a") as f:
#         f.write(print_prog_output)
#         f.write("\n")

#     json_output = json.loads(parse_bril(print_prog_output))
#     assert json_output == prog

fail_prog = {'functions': [{'args': [],
   'instrs': [{'args': 'a', 'op': 'print'}],
   'name': 'main',
   'type': 'int'}]}

# @given(bril_program())
# def test_interpreter(prog):
def test_interpreter():
    prog = fail_prog

    # TODO: feed main args?

    with open("interp_progs", "a") as f:
        f.write(json.dumps(prog, indent=4))
        f.write("\n")

    try:
        output = subprocess.check_output("brili",
                                         input=json.dumps(prog).encode(),
                                         shell=True)
        with open("outputs", "a") as f:
            f.write(str(output))
            f.write("\n")
    except subprocess.CalledProcessError as e:
        assert(e.returncode == 2)

test_interpreter()