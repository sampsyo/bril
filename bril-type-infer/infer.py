"""Type inference for Bril
"""
import json
import sys
import copy

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


# Runtime is O(n^2) where n is the number of instructions, because of
# main {
#   jmp l2;
# l1:
#   a = id b;
#   b = id c;
#   c = id d;
#   ...
#   y = id z;
#   ret;
# l2:
#   z = const 0;
# }
def infer_types_func(func):
    gamma = {}
    typed_func = copy.deepcopy(func)
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
                if instr["value"] is True or instr["value"] is False:
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

            elif instr["op"] in COMPARISON_OPS:
                for arg in instr["args"]:
                    type_var(gamma, arg, "int", i)
                type_var(gamma, instr["dest"], "bool", i)

            elif instr["op"] in LOGIC_OPS:
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

def analyze_vars(typed_func):
    labels = set()
    gamma = dict()
    for instr in typed_func["instrs"]:
        if "label" in instr:
            labels.add(instr["label"])
        else:
            if "dest" in instr and "type" in instr:
                gamma[instr["dest"]] = instr["type"]
    return gamma, labels

def typecheck_label(label, gamma):
    if label in gamma:
        raise Exception(
            'Expected "{}" to be a label, but it was a'
            ' variable of type "{}"'
            .format(label, gamma[label])
        )

def typecheck_func(original_func, typed_func):
    gamma, labels = analyze_vars(typed_func)
    for instr in original_func["instrs"]:
        if "label" in instr and instr["label"] in gamma:
            raise Exception(
                'Expected "{}" to be a label, but it was a'
                ' variable of type "{}"'
                .format(instr["label"], gamma[instr["label"]])
            )

        if "op" in instr:
            if "dest" in instr and "type" in instr and instr["type"] != gamma[instr["dest"]]:
                raise Exception(
                    'Expected "{}" to have type, but it was explicitly'
                    ' typed as "{}"'
                    .format(gamma[instr["dest"]], instr["type"])
                )
            elif instr["op"] == "jmp":
                typecheck_label(instr["labels"][0], gamma)
            elif instr["op"] == "br":
                typecheck_label(instr["labels"][0], gamma)
                typecheck_label(instr["labels"][1], gamma)

def typecheck(original_bril, typed_bril):
    for i in range(len(original_bril["functions"])):
        typecheck_func(original_bril["functions"][i], typed_bril["functions"][i])

if __name__ == '__main__':
    bril = json.load(sys.stdin)
    typed_bril = infer_types(bril)
    if '-t' in sys.argv:
        typecheck(bril, typed_bril)
    json.dump(typed_bril, sys.stdout, indent=2, sort_keys=True)
