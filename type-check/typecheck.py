'''Static type checking for a Bril program'''

import json
import sys
import re

sys.path.insert(1, './bril-txt')
from briltxt import parse_bril

OP_ARITHMETIC = ['add', 'mul', 'sub', 'div']
OP_COMP = ['eq', 'lt', 'gt', 'le', 'ge']
OP_LOGIC = ['not', 'and', 'or']

def other_type(t):
	if t == 'int':
		return 'bool'
	else:
		return 'int'

#all operations
def check_length(instr, l, line):
	if 'args' in instr:
		if l == -1: 
			return True
		if len(instr['args']) != l:
			print('line %d: incorrect number of arguments' % line)
			return False
	else:
		if l == 0:
			return True
		else:
			print('line %d: incorrect number of arguments' % line)

	return True

#all operations other than print
def check_lhs(instr, line, op):
	lhs = instr['type']
	if lhs != 'int' and lhs != 'bool':
		print('line %d: lhs type %s does not exist' % (line, lhs))
		return False
	if op == 'const': 
		return True
	if lhs == 'bool' and op == 'a':
		print('line %d: assigning arithmetic operation result to boolean' % line)
		print('arithmetic op error!')
		return False
	if lhs == 'int' and op == 'b':
		print('line %d: assigning logic operation result to integer' % line)
		print('logic op error!')
		return False
	if lhs == 'int' and op == 'c':
		print('line %d: assigning comparison operation result to integer' % line)
		print('comparison op error!')
		return False

	return True

#const
def check_const(instr, line):
	try:
		rhs = type(instr['value']).__name__
	except:
		print('line %d: rhs type %s does not exist' % (line, rhs))

	rhs = type(instr['value']).__name__

	if rhs != 'int' and rhs!= 'bool':
		print('line %d: rhs type %s does not exist' % (line, rhs))
		return False

	rhs = type(instr['value']).__name__
	lhs = instr['type']

	if lhs != rhs:
		print('line %d: assigning %s value to %s variable' % (line, rhs, lhs) )
		return False

	return True	

#all operations that takes in variable
def check_if_arg_exist(instr, line, context, branch=False):
	for v in instr['args']:
		if v not in context:
			print('line %d: args %s undefined' % (line, v) )
			return False
		if branch:
			break

	return True

def check_arg_type(instr, line, context, t, branch=False):
	for v in instr['args']:
		if v not in context:
			print('line %d: args %s is not expected type %s' % (line, v, t) )
			return False
		if branch:
			break

	return True

#id
def check_id_equal(instr, line, context):
	rhs = instr['args'][0]
	rhs = 'int' if rhs in context['int'] else 'bool'
	lhs = instr['type']
	if lhs != rhs:
		print('line %d: assigning %s value to %s variable' % (line, rhs, lhs) )
		return False

	return True	

#all operations that takes in variable
def check_redefined(instr, line, context):
	if instr['dest'] in context:
		print('line %d: re-assigning %s value to existing %s variable' % (line,
			other_type(instr['type']), instr['type'] ))
		return False

	return True

#check if branch get;s wrong label
def check_branch(instr, line, index, labels):
	if index == 0:	# jmp
		if instr['args'][0] in labels:
			return True
	elif index == 1:	# branch
		if instr['args'][1] not in labels or instr['args'][2] not in labels:
			print('\tbranch labels undefined')
			return False
		return True

	return False

def instr_check(instrs, labels, ignored):
	'''Check instrs one after another
	'''

	i = 1 # line number
	context = {'int':[], 'bool':[],'all':[]}	# 'int': integer var; 'boolean': boolean var

	for instr in instrs:
		while i in ignored:
			i += 1
		if 'op' in instr:
			# Const arithmetic
			if instr['op'] == 'const':
				if not check_length(instr, 0, i): 
					return False
				if not check_lhs(instr, i, 'const'):
					return False
				if not check_const(instr, i):
					print('const error!')
					return False
				if not check_redefined(instr, i, context[other_type(instr['type'])]):
					print('const error!')
					return False
				context[instr['type']].append(instr['dest'])
				context['all'].append(instr['dest'])

			# arithmetic op
			elif instr['op'] in OP_ARITHMETIC:
				if not check_length(instr, 2, i): 
					return False
				if not check_lhs(instr, i, 'a'):
					return False
				if not check_if_arg_exist(instr, i, context['all']):
					return False
				if not check_arg_type(instr, i, context['int'], 'int'):
					print('arithmetic op error!')
					return False

				if not check_redefined(instr, i, context[other_type(instr['type'])]):
					print('arithmetic op error!')
					return False
				context[instr['type']].append(instr['dest'])
				context['all'].append(instr['dest'])

			# comparison op
			elif instr['op'] in OP_COMP:
				if not check_length(instr, 2, i): 
					return False
				if not check_lhs(instr, i, 'c'):
					return False
				if not check_if_arg_exist(instr, i, context['all']):
					return False
				if not check_arg_type(instr, i, context['int'],'int'):
					print('comparison op error!')
					return False
				if not check_redefined(instr, i, context[other_type(instr['type'])]):
					print('comparison op error!')
					return False
				context[instr['type']].append(instr['dest'])
				context['all'].append(instr['dest'])

			# logic op
			elif instr['op'] in OP_LOGIC:
				length = 1 if instr['op'] == 'not' else 2
				if not check_length(instr, length, i): 
					return False
				if not check_lhs(instr, i, 'b'):
					return False
				if not check_if_arg_exist(instr, i, context['all']):
					return False
				if not check_arg_type(instr, i, context['bool'],'bool'):
					print('logic op error!')
					return False
				if not check_redefined(instr, i, context[other_type(instr['type'])]):
					print('logic op error!')
					return False
				context[instr['type']].append(instr['dest'])
				context['all'].append(instr['dest'])

			# id
			elif instr['op'] == 'id':
				if not check_length(instr, 1, i): 
					return False
				if not check_if_arg_exist(instr, i, context['all']):
					return False
				if not check_id_equal(instr, i, context):
					print('id op error!')
					return False
				if not check_redefined(instr, i, context[other_type(instr['type'])]):
					print('id op error!')
					return False
				context[instr['type']].append(instr['dest'])
				context['all'].append(instr['dest'])

			# print values
			elif instr['op'] == 'print':
				if not check_length(instr, -1, i): 
					return False
				if not check_if_arg_exist(instr, i, context['all']):
					print('print error!')
					return False

			# jmp
			elif instr['op'] == 'jmp':
				if not check_length(instr, 1, i):
					return False
				if not check_branch(instr, i, 0, labels):
					print('line %d: jump to undefined label %s' % (i, instr['args'][0]))
					print('jmp op error!')
					return False

			# branch
			elif instr['op'] == 'br':
				if not check_length(instr, 3,i):
					return False
				if not check_if_arg_exist(instr, i, context['all'], branch=True):
					return False
				if not check_arg_type(instr, i, context['bool'],'bool', branch=True):
					print('br op error!')
					return False
				if not check_branch(instr, i, 1, labels):
					print('line %d: branch args undefined' % i)
					print('br op error!')
					return False

			# return
			elif instr['op'] == 'ret':
				if len(instr['args']) != 0:
					print('line %d: return with arguments' % i)
					print('ret error!')
					return False
				return True

			# nop
			elif instr['op'] == 'nop':
				if len(instr['args']) != 0:
					print('line %d: nop with arguments' % i)
					print('nop error!')
					return False
		i += 1
	return True




def label_pass(instrs, ignored):
	'''A first pass for the Bril program to collect all the labels in 'label'.
	Return error if any label show up more than once.
	'''
	labels = []
	i = 1 # line number
	for instr in instrs:
		while i in ignored:
			i += 1
		if 'label' in instr:
			if instr['label'] in labels:
				print('line %d: label %s show up twice' % (i, instr['label']))
				return False
			labels.append(instr['label'])
		i += 1 

	return labels


def typecheck(bril, ignored):
	'''Given a Bril program, print out the #lines have type check errors
	or type check successfully.
	'''
	func = bril['functions'][0]
	labels = label_pass(func['instrs'], ignored)
	if labels == False:
		print('label error!')
		return
	if not instr_check(func['instrs'], labels, ignored):
		return

	print('Type checking passed')

	return

#if the expression is ignored in parse_bril
def if_ignored(line,i,ignored):
	comment = line.find('#')
	if comment == 0:
		line = ''
	elif comment != -1:
		line = line[comment-1]

	if all(x == '\n' or x == '\t' or x == '\0' or x == ' ' for x in line):
		ignored.append(i)
	if 'main' in line:
		ignored.append(i)
	if '}' in line:
		ignored.append(i)

	return ignored

#if __name__ == '__main__':
def typecheck_main():
	line = sys.stdin.readline()
	inputs = line
	ignored = []
	i = 1
	ignored = if_ignored(line, i, ignored)
	while(line):
		line = sys.stdin.readline()
		inputs += line
		i += 1
		ignored = if_ignored(line, i, ignored)
	instrs = parse_bril(inputs)
	typecheck(json.loads(instrs), ignored)
	#typecheck(json.load(sys.stdin))
