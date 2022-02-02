import json
import sys

def print_vars(dests):
    for func in dests.keys():
        print(func + ':')
        print(dests[func])

def get_vars():
    """ Prints the destination registers introduced in each function
    """
    dests = {}
    prog = json.load(sys.stdin)
    for func in prog['functions']:
        func_name = func['name']
        func_dests = []
        for instr in func['instrs']:
            if 'dest' in instr.keys():
                func_dests.append(instr['dest'])
        dests[func_name] = func_dests 
    print_vars(dests)

if __name__ == '__main__':
    get_vars()
