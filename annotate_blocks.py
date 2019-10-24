import collections
import json
import sys
sys.path.append('examples/')
sys.path.append('bril-txt/')
import briltxt
import cfg
import form_blocks


def annotate(program):
    line_num = 1
    for function in program['functions']:
        function['line'] = line_num
        line_num += 1
        blocks = cfg.block_map(form_blocks.form_blocks(function['instrs']))
        for instr in function['instrs']:
            instr['line'] = line_num
            line_num += 1
            if 'label' in instr:
                instr['block'] = instr['label']
            else:
                bs = [b for b in blocks if instr in blocks[b]]
                instr['block'] = [b for b in blocks if instr in blocks[b]][0]
    return program


if __name__ == '__main__':
    bril_file = sys.argv[1]
    with open(bril_file) as f:
        program = json.loads(briltxt.parse_bril(f.read()))
    print(json.dumps(annotate(program), indent=2, sort_keys=True))