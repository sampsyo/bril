import ujson
import sys

entry_name = '__entry__'

class BasicBlock():
    def __init__(self, label=None):
        self.label = label
        self.instructions = []

    def __str__(self):
        s = '\n'
        s += self.label if self.label else '<unlabeled>'
        s += ':\n\t'
        s += '\n\t'.join([str(i) for i in self.instructions])
        return s

    def __repr__(self):
        return str(self)

def parse2blocks(json_program):
    functions = []

    for func in json_program['functions']:
        basic_blocks = []
        current_block = BasicBlock(label=entry_name)

        for instr in func['instrs']:
            # Labels indicate the end of the _previous_ block
            if 'label' in instr:
                label = instr['label']
                if label == entry_name:
                    print("Oh no! Label already named", entry_name)

                if current_block.instructions:
                    basic_blocks.append(current_block)
                    current_block = BasicBlock(label=label)
                else:
                    current_block.label = label
                continue

            # Otherwise, it's an instruction!
            opcode = instr['op']
            current_block.instructions.append(instr)
            # Branches, jumps, and returns indicate the end of current block
            if opcode in ['br', 'jmp', 'ret']:
                basic_blocks.append(current_block)
                current_block = BasicBlock()

        basic_blocks.append(current_block)
        functions.append(basic_blocks)

    return functions

if __name__ == '__main__':
    funs = parse2blocks(ujson.loads(sys.stdin.read()))
    print(funs)
