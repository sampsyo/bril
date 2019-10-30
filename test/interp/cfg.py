import json
import sys
def cfg(jscode):
	for inst in jscode['functions'][0]['instrs']:
		if 'op' in inst.keys():
			print(inst['op'])

if __name__ == '__main__':
    cfg(json.load(sys.stdin))
