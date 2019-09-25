"""Type inference for Bril
"""
import json
import sys

ARITHMETIC_OPS = ["add", "mul", "sub", "div"]
COMPARISON_OPS = ["eq", "lt", "gt", "le", "ge"]
LOGIC_OPS = ["not", "and", "or"]

def type_var(gamma, var, expected_type, i):
    if var in gamma and gamma[var] != expected_type:
        raise Exception(
            '(stmt {}) Expected "{}" to have type "{}" but found "{}"'.format(
                i,
                var,
                expected_type,
                gamma[var]
        ))
    gamma[var] = expected_type


def infer_types_func(func):
    gamma = {}
    typed_func = func.copy()
    # Keep track of whether or not any type was inferred.
    # If so, then we need to try type checking again, because for `id` ops,
    # e.g. "x = id y", we need to know the type of `y` first, which may not
    # happen until later in the program because we're going sequentially
    # through the instructions
    old_gamma_size = -1
    while len(gamma) != old_gamma_size:
        old_gamma_size = len(gamma)
        for i, instr in enumerate(typed_func["instrs"]):
            # Continue if we have a label
            if "op" not in instr:
                continue

            # Handle constants
            if instr["op"] == "const":
                if instr["value"] == True or instr["value"] == False:
                    type_var(gamma, instr["dest"], "bool", i)
                else:
                    type_var(gamma, instr["dest"], "int", i)

            # Handle effect operations
            if instr["op"] == "ret" or instr["op"] == "jmp":
                continue

            # Handle value operations
            elif instr["op"] in ARITHMETIC_OPS:
                for arg in instr["args"]:
                    type_var(gamma, arg, "int", i)
                type_var(gamma, instr["dest"], "int", i)

            elif instr["op"] in COMPARISON_OPS or instr["op"] in LOGIC_OPS:
                for arg in instr["args"]:
                    type_var(gamma, arg, "bool", i)
                type_var(gamma, instr["dest"], "bool", i)

            elif instr["op"] == "br":
                type_var(gamma, instr["args"][0], "bool", i)

            # Handle misc. operations
            elif instr["op"] == "print" or instr["op"] == "nop":
                continue
            elif instr["op"] == "id" and instr["args"][0] in gamma:
                type_var(gamma, instr["dest"], gamma[instr["args"][0]], i)

            # Set the type for this instruction to be whatever we've inferred
            if "dest" in instr and instr["dest"] in gamma:
                typed_func["instrs"][i]["type"] = gamma[instr["dest"]]
        # end for over instructions
    # end while
    return typed_func

def infer_types(bril):
    typed_bril = {"functions": []}
    for f in bril["functions"]:
        typed_function = infer_types_func(f)
        typed_bril["functions"].append(typed_function)
    return typed_bril

if __name__ == '__main__':
    bril = json.load(sys.stdin)
    typed_bril = infer_types(bril)
    json.dump(typed_bril, sys.stdout, indent=2, sort_keys=True)
