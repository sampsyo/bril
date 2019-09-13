import ujson
import sys

from operator import itemgetter

entry_name = '__entry__'

class BasicBlock():
    def __init__(self, label=None):
        self.label = label
        self.instructions = []
        self.next = None

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


def local_value_numbering(block):
    # array: (number) expression, cannonical
    # map: variable -> int

    expressions = []
    vars_to_exprs = {}

    for instr in block.instructions:
        # value statement!
        if 'dest' in instr:
            if 'args' in instr:
                args = [expressions[vars_to_exprs[a]] if a in vars_to_exprs else a for a in instr['args']]
            else:
                args = None

            computation = (instr['op'], args)
            matches = [(i, e) for i, e in enumerate(expressions) if e[0] == computation]
            if len(matches) > 1:
                raise ValueError("bad bad not good!!!")

            if matches:
                pass
                # handle!

            else:
                # Add this to our expression list
                expressions.append((computation, instr['dest']))






        # effectful: no destination
        else:
            print("ah")





# def block_with_label(label, block_list):
#     return next(x for x in block_list if x.label == label)

# def blocks2cfg(functions):
#     all_cfgs = []

#     for block_list in functions:
#         root = None
#         entry = block_with_label(entry_name, block_list)

#         for block in block_list:
#             last = block.instructions[-1]
#             # if last['op'] == 'jmp':
#             #     dest = 

if __name__ == '__main__':
    funs = parse2blocks(ujson.loads(sys.stdin.read()))
    # print(funs)
    # blocks2cfg(funs)
    blocks = funs[0]
    for b in blocks: 
        local_value_numbering(b)




