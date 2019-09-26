"""An automated property-based tester for Bril programs. 

This module uses the Hypothesis framework to define strategies for generating 
Bril JSON programs. It runs two flavors of tests: (1) checking that Bril text to
JSON is invertible, and (2) checking that Bril programs cause the interpreter to
only fail with  known errors, rather than exceptions in the implementation 
details.
"""

from briltxt import *
from contextlib import redirect_stdout
from hypothesis import *
from hypothesis.strategies import *
import io
import subprocess


"""
Strategies for compositionally generating test Bril programs in JSON.
"""

names = text(alphabet=characters(min_codepoint=97, max_codepoint=122), 
             min_size=1,
             max_size=1)

types = sampled_from(["int", "bool"])

@composite
def effect_opcodes(draw):
    return draw(sampled_from(["br", "jmp", "print", "ret"]))

@composite
def value_opcodes(draw):
    return draw(sampled_from(["add", "mul", "sub", "div", "id", "nop", "eq",
                              "lt", "gt", "ge", "le", "not", "and", "or"]))

@composite 
def function_type(draw):
    return draw(sampled_from([draw(types), "void"]))

@composite
def bril_effect_instr(draw):
    opcode = draw(effect_opcodes())
    if (opcode == "print"):
        args = [draw(names)]
    else:
        args = draw(lists(names, min_size=1, max_size=3))
    return {
        "op": opcode,
        "args": args}

@composite
def bril_value_instr(draw):
    opcode = draw(value_opcodes())
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
def bril_instr_or_label(draw):
    return draw(sampled_from([
        draw(names),
        draw(bril_instr()),
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
    instrs = draw(lists(bril_instr_or_label(), max_size=10))
    typ = draw(function_type())
    if (typ != "void"):
        return {
          "name": name,
          "instrs": instrs,
          "args": args,
          "type": typ,
        }
    else:
        return {
          "name": name,
          "instrs": instrs,
          "args": args,
        }

@composite
def bril_program(draw):
    main = draw(bril_function())
    main["name"] = "main"

    all_funs = [main] + draw(lists(bril_function(), max_size=10))

    return {
      "functions": all_funs,
    }

@composite
def bril_structued_function(draw):
    constants = draw(lists(bril_constant_instr(), min_size=2, max_size=2))
    values = draw(lists(bril_value_instr(), min_size=1, max_size=2))
    effects = draw(lists(bril_effect_instr(), min_size=1, max_size=2))
    instrs = constants + values + effects

    name = draw(names)
    args = draw(lists(bril_arg(), max_size=3))
    typ = draw(function_type())
    if (typ != "void"):
        return {
          "name": name,
          "instrs": instrs,
          "args": args,
          "type": typ,
        }
    else:
        return {
          "name": name,
          "instrs": instrs,
          "args": args,
        }

@composite
def bril_structued_program(draw):
    main = draw(bril_structued_function())
    main["name"] = "main"

    all_funs = [main] + draw(lists(bril_structued_function(), max_size=10))

    return {
      "functions": all_funs,
    }

"""
Example test program
"""

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

"""
Test for invertibility of JSON-to-Bril text.
"""
@example(prog=add_prog)
@given(bril_program())
@settings(verbosity=Verbosity.debug, max_examples=300)
def test_json_to_bril_txt(prog):
    with open("hypothesis_json_programs", "a") as f:
        f.write(json.dumps(prog, indent=4))
        f.write("\n")

    f = io.StringIO()
    with redirect_stdout(f):
        print_prog(prog)
    print_prog_output = f.getvalue()

    with open("hypothesis_bril_programs", "a") as f:
        f.write(print_prog_output)
        f.write("\n")

    json_output = json.loads(parse_bril(print_prog_output))
    assert json_output == prog

"""
Test for Brili interpreter failing with only known errors.
"""
@example(prog=add_prog)
@given(bril_structued_program())
@settings(verbosity=Verbosity.debug, max_examples=300)
def test_interpreter(prog):
    with open("hypothesis_interp_programs", "a") as f:
        f.write(json.dumps(prog, indent=4))
        f.write("\n")

    try:
        output = subprocess.check_output("brili",
                                         input=json.dumps(prog).encode(),
                                         shell=True)
        with open("hypothesis_interp_outputs", "a") as f:
            f.write(str(output))
            f.write("\n")
    except subprocess.CalledProcessError as e:
        assert(e.returncode == 2)
