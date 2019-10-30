import json
import sys
import os
import copy
from collections import OrderedDict

sys.path.insert(0, "examples/")
from cfg import block_map, successors, add_terminators
from form_blocks import form_blocks
from dom import get_dom, get_pred
from df import df_worklist, ANALYSES
from util import fresh
from funcs import *

BASIC_OP = ['sub', 'add']
ADV_OP = ['mul', 'div']
BRANCH_OP = ['jmp', 'br']


def inductionVar(blocks, loops, licd, reach_def):
	"""Scan the loop body to find all basic induction variables
	"""
	basic_var = {}	# basic induction variable: X=X+c/X-c, c is const/LI
	induction_var = {}
	loop_invariant = {}	# candidate induction variable
	constants = {}
	trace = {}
	# print()
	# print(loops)
	# print()
	# print(licd)
	# print()
	# print(reach_def)
	# print()

	for loop in loops.keys():
		basic_var[loop], induction_var[loop] = [], []
		loop_invariant[loop], constants[loop] = [], []
		trace[loop] = {}

		cond, body = loop[1], loop[0]
		# print(cond, '|', body)

		for item in reach_def[cond]:
			if cond not in reach_def[cond][item] and body not in reach_def[cond][item]:
				constants[loop].append(item)
		# print('const:', constants)

		for item in reach_def[body]:
			if body in reach_def[body][item] and len(reach_def[body][item]) == 2:
				induction_var[loop].append(item)
		# print('induction var:', induction_var)

		for item in licd[loop]:
			loop_invariant[loop].append(item['dest'])
		# print('loop invariant:', loop_invariant)
		# print()

		temp_trace = []
		for item in induction_var[loop]:
			for instr in blocks[body]:	# traverse definition
				if instr['dest'] == item:
					trace[loop][item] = [instr['op']] + instr['args']
					temp_trace = temp_trace + instr['args']
					break
		# print('trace:', trace)
		# print('temp:', temp_trace)

		while temp_trace:
			# print(temp_trace)
			temp = temp_trace.pop()
			# print(temp)
			for instr in blocks[body]:	# further traverse definition
				if instr['op'] in BRANCH_OP:
					continue
				if instr['dest'] == temp:
					trace[loop][temp] = [instr['op']] + instr['args']
					# print(trace)					
					for arg in instr['args']:
						if arg in constants[loop]:
							continue
						elif arg in trace[loop].keys():
							continue
						elif arg in temp_trace:
							continue
						else:
							temp_trace.append(arg)
					break

		# print('trace', trace)
		for item in induction_var[loop]:
			temp_ind = [] + trace[loop][item][1:]
			# print()
			# print(item)
			while temp_ind:
				temp = temp_ind.pop()
				# print(temp)
				if temp == item:
					temp_ind = []
					basic_var[loop].append(temp)
					break
				elif temp in constants[loop] or temp in loop_invariant[loop]:
					continue
				elif temp in induction_var[loop]:
					temp_ind = []
					continue
				elif trace[loop][temp][0] in ADV_OP:
					break
				else:
					temp_ind = temp_ind + trace[loop][temp][1:]
		# print('basic:', basic_var)
	return constants, loop_invariant, basic_var, induction_var, trace


def strength_red(blocks, pre_header, constants, loop_invariant, basic_var, induction_var, trace, loops, reach_def):
	'''
	This function returns:
	blocks: a modification to the blocks - input the function. 
	It reduce mul/div of induction variables to add/sub
	'''
	b_names = list(blocks.keys())
	names = reach_def[b_names[-1]].keys()
	# print('name:', names)
	# print('block names:', b_names)
	# print('loop:', loops)
	# print('constants:', constants)
	# print('loop invariant', loop_invariant)
	# print('basic var:', basic_var)
	# print('induction var:', induction_var)
	# print('trace:', trace)

	for loop in loops:
		# print(loop[1])
		ind = b_names.index(pre_header[loop[1]])
		# print(ind)
		for var in induction_var[loop]:
			if var in basic_var[loop]:
				continue

			temp = [var]
			result = []
			while temp:
				# print('\t', temp)
				current_var = temp.pop()
				if trace[loop][current_var][0] not in ADV_OP:
					for i in range(len(trace[loop][current_var])-1):
						if trace[loop][current_var][i+1] == var or trace[loop][current_var][i+1] in basic_var[loop]:
							continue
						temp.append(trace[loop][current_var][i+1])
				else:
					result.append(trace[loop][current_var])
			# print('result', result)

			for i in range(len(result)):
				temp = result[i][1:]
				while temp:
					current_var = temp.pop()
					if current_var in basic_var[loop]:
						current_basic = current_var
					elif current_var in constants[loop] or current_var in loop_invariant[loop]:
						continue
					else:
						temp = temp + trace[loop][current_var][1:]
				# print('current_basic',current_basic)

				temp = [current_basic]
				while temp:
					current_var = temp.pop()
					if trace[loop][current_var][0] ==  'id':
						temp.append(trace[loop][current_var][1])
					elif trace[loop][current_var][0] == 'sub':
						flag = 'sub'
						delta = trace[loop][current_var][-1]
						break
					elif trace[loop][current_var][0] == 'add':
						flag = 'add'
						delta = trace[loop][current_var][-1]
						break

				tmp_name = fresh('s', names)
				new_name = fresh('s', names.append(tmp_name))
				if flag == 'sub':
					var_def = {"op": "add", "args": [current_basic, delta], "dest": tmp_name, "type": "int"}
				else:
					var_def = {"op": "sub", "args": [current_basic, delta], "dest": tmp_name, "type": "int"}
				var_def["comment"] = "strength reduction"
				blocks[pre_header[loop]].append(var_def)

				if result[i][1] in constants[loop]:
					new_def = {"op": result[i][0], "args": [result[i][1], var_def], "dest": new_name, "type": "int"}
				else:
					new_def = {"op": result[i][1], "args": [var_def, result[i][2]], "dest": new_name, "type": "int"}
				new_def["comment"] = "strength reduction"
				blocks[pre_header[loop]].append(new_def)

				for instr in blocks[pre_header[loop]+2]:
					if instr['args'] == result[i][1:]:
						ind = blocks[pre_header[loop]+2].index(instr)
						dest = blocks[pre_header[loop]+2][ind]['dest']
						blocks[pre_header[loop]+2][ind]['dest'] = new_name
						blocks[pre_header[loop]+2][ind]['op'] = flag
						blocks[pre_header[loop]+2][ind]['args'] = [new_name, delta]
						blocks[pre_header[loop]+2][ind]['comment'] = 'strength reduction'
					elif dest in instr['args']:
						ind = blocks[pre_header[loop]+2].index(instr)
						args_ind = blocks[pre_header[loop]+2][ind]['args'].index(dest)
						blocks[pre_header[loop]+2][ind]['args'][args_ind] = new_name
						blocks[pre_header[loop]+2][ind]['comment'] = 'strength reduction'
	return blocks



def strengthRed():
	bril = json.load(sys.stdin)

	for i, func in enumerate(bril['functions']):
		exits, live_var, dom, oblocks, loops, reach_def, _ = loop_king(func)
		loop_invariants = find_LI(oblocks, loops, reach_def)
		pre_header, new_blocks = create_preheaders(oblocks, loops)
		code_motion, licd = move_LI(new_blocks, pre_header, loop_invariants, loops, dom, live_var, exits)
		bril['functions'][i] = blocks_to_func(code_motion, func)
		#print('licd', licd)

		# print(reach_def)
		# print()
		constants, loop_invariant, basic_var, induction_var, trace = inductionVar(oblocks, loops, licd, reach_def)
		pre_header, new_blocks = create_preheaders(oblocks, loops)
		# print('blocks', new_blocks)
		st = strength_red(new_blocks, pre_header, constants, loop_invariant, basic_var, induction_var, trace, loops, reach_def)
		bril['functions'][i] = blocks_to_func(st, func)
	print(json.dumps(bril)) 
	return

if __name__ == '__main__':
	strengthRed()