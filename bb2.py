import json
import sys


TERMINATORS = {'jmp', 'br', 'ret'}


def form_blocks(instrs):
	cur_block = []
	for instr in instrs:
		if 'label' in instr and cur_block:
			yield cur_block
			cur_block = []
		cur_block.append(instr)
		if 'label' not in instr and instr['op'] in TERMINATORS:
			yield cur_block
			cur_block = []
	if cur_block:
		yield cur_block


def drive():
	prog = json.load(sys.stdin)
	for func in prog['functions']:
		blocks = list(form_blocks(func['instrs']))
		print(f"@{func['name']} has {len(blocks)} blocks")
		for i, block in enumerate(blocks):
			print(f" - block {i}:")
			for instr in block:
				print(instr)
			print("")
		print("")
		print("")


if __name__ == '__main__':
	drive()
