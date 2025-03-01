import json
import sys


def get_types(func):
    """Collect the type for every `get` instruction."""
    types = {}
    for instr in func["instrs"]:
        if instr.get("op") == "get":
            types[instr["dest"]] = instr["type"]
    return types


def shadow(name):
    return f'shadow.{name}'


def func_from_ssa(func):
    types = get_types(func)

    out_instrs = []
    for instr in func["instrs"]:
        if instr.get("op") == "set":
            # Sets become copies into shadow variables.
            copy = {
                "op": "id",
                "dest": shadow(instr["args"][0]),
                "type": types[instr["args"][0]],
                "args": [instr["args"][1]],
            }
            out_instrs.append(copy)
        elif instr.get("op") == "get":
            # Gets become copies out of shadow variables.
            copy = {
                "op": "id",
                "dest": instr["dest"],
                "type": instr["type"],
                "args": [shadow(instr["dest"])],
            }
            out_instrs.append(copy)
        else:
            # Preserve other instructions.
            out_instrs.append(instr)

    func["instrs"] = out_instrs


def from_ssa(bril):
    for func in bril["functions"]:
        func_from_ssa(func)
    return bril


if __name__ == "__main__":
    print(json.dumps(from_ssa(json.load(sys.stdin)), indent=2, sort_keys=True))
