import json
import sys

def count_instructions(bril):
    return sum([len(f['instrs']) for f in bril['functions']])

if __name__ == '__main__':
    print(count_instructions(json.load(sys.stdin)))