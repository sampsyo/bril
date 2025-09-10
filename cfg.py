# constructs CFG, represented by a set of edges.
# each edge is a tuple (i, j) which represents
# a flow from block i to block j.


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
		if ('label' not in instr and 
			instr['op'] in TERMINATORS):
			yield cur_block
			cur_block = []
	if cur_block:
		yield cur_block


def form_cfg(blocks):
	cfg = []
	label_to_blockids = {}
	for i, block in enumerate(blocks):
		if 'label' in block[0]:
			label_to_blockids[block[0]['label']] = i
	for i, block in enumerate(blocks):
		last = block[-1]
		if ('label' not in last and
			 last['op'] == 'jmp'):
			bid_target = label_to_blockids[last['labels'][0]]
			cfg.append((i, bid_target))
		elif ('label' not in last and
			 last['op'] == 'br'):
			bid_target1 = label_to_blockids[last['labels'][0]]
			bid_target2 = label_to_blockids[last['labels'][1]]
			cfg.append((i, bid_target1))
			cfg.append((i, bid_target2))
		elif ('label' not in last and
			 last['op'] == 'ret'):
			continue
		else:
			if block != blocks[-1]:
				cfg.append((i, i+1))
	return cfg


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

		# print cfg of this function
		print(f" * cfg of @{func['name']}:")
		cfg = list(form_cfg(blocks))
		for (i, j) in cfg:
			print(f"    - bb{i} -> bb{j}")

		print("")
		print("")


if __name__ == '__main__':
	drive()
