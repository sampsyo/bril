import json
import sys

TERMINATORS = ['jmp', 'br', 'ret']

def form_blocks(body):
    blocks = [[]]
    for i in body:
        if 'label' in i: #A label
            blocks.append([i])
        else: #An actual instruction
            blocks[-1].append(i)
            if i['op'] in TERMINATORS: #A terminator instruction
                blocks.append([])
    return blocks

def label_blocks(blocks):
    lbl2block = {}
    i = 0
    for block in blocks:
        if 'label' in block[0]: # block already has a label
            lbl2block[block[0]['label']] = block
            block = block[1:]
        else: # block has no label
            label = 'block' + str(i) #create a unique label for it
            i += 1
            lbl2block[label] = block
    return lbl2block

def cfg():
    prog = json.load(sys.stdin)
    for func in prog['functions']:
        blocks = form_blocks(func['instrs'])
        lbl2block = label_blocks(blocks)
        for block in blocks:
            print(block)
        # build cfg
        cfg = {} #label -> list of labels of successive blocks
        for i in range(len(blocks)):
            last = blocks[i][-1]
            label = blocks[i][0].get('label', 'block' + str(i))
            if last['op'] in ['jmp', 'br']:
                cfg[label] = last['labels']
            else:
                if i < len(blocks) - 1:
                    cfg[label] = [blocks[i+1][0].get('label', 'block' + str((i + 1)))]
                else:
                    cfg[label] = []
        print(cfg)

if __name__ == '__main__':
    cfg()
