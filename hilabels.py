# adds ".hi" to every label names.

import json
import sys

if __name__ == '__main__':
	prog = json.load(sys.stdin)
	for func in prog['functions']:
		for instr in func['instrs']:
			if 'label' in instr:
				instr['label'] += ".hi"
	json.dump(prog, sys.stdout)
