import json
import sys

def tool():
    # Memory Access Instructions
    prog = json.load(sys.stdin)
    print(prog)

    loads = 0
    stores = 0
    allocs = 0

    for function in prog['functions']:
        for instr in function['instrs']:
            if instr.get('label') is None:
              print(instr["op"])
              if (instr["op"] == 'load'):
                  loads += 1
              elif (instr["op"] == 'store'):
                  stores += 1
              elif (instr["op"] == 'alloc'):
                  allocs += 1
    print(f"Load count: {loads}")
    print(f"Store count: {stores}")
    print(f"Alloc count: {allocs}")

    





if __name__ == '__main__':
    tool()