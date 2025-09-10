# Counts the total number of labels and functions
#  in a bril program.

import json
import sys

if __name__ == '__main__':
	prog = json.load(sys.stdin)
	num_funcs = 0
	num_labels = 0
	for func in prog['functions']:
		num_funcs = num_funcs + 1
		for instr in func['instrs']:
			if 'label' in instr:
				num_labels = num_labels + 1
	print(num_funcs + num_labels)
