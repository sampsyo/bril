import json
import sys

allocated = set()
def check_fn(body):
	for instr in body:
		if 'op' in instr:
			if instr['op'] == 'alloc':
				allocated.add(instr['dest'])
			if instr['op'] == 'free':
				allocated.remove(instr['args'][0])

def memleak():
	prog = json.load(sys.stdin)
	for func in prog['functions']:
		check_fn(func['instrs'])
	if len(allocated) > 0:
		for e in allocated:
			print(e, " is not freed")

if __name__ == '__main__':
	memleak()
