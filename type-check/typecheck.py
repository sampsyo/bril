'''Static type checking for a Bril program'''

import json
import sys

sys.path.insert(1, './bril-txt')
import briltxt

OP_ARITHMETIC = ['add', 'mul', 'sub', 'div']
OP_COMP = ['eq', 'lt', 'gt', 'le', 'ge']
OP_LOGIC = ['not', 'and', 'or']


def check_a(instr, index, a_context=[]):
	if index == 0:	# const
		if type(instr['value']) == int:
			return True
	elif index == 1:	# arithmetic add/mul/sub/div
		if len(instr['args']) != 2:
			print('\tinsufficient #op')
			return False
		elif instr['args'][0] not in a_context or instr['args'][1] not in a_context:
			print('\targs undefined properly')
			return False
		return True
	# elif index == 2:	# print values
	# 	for var in instr['args']:
	# 		if var not in a_context:
	# 			print('\tInt var %s not defined' % var)
	# 			return False
	# 	return True
	elif index == 3:
		if len(instr['args']) == 1 and instr['args'][0] in a_context:
			return True

	return False


def check_b(instr, index, context=[]):
	if index == 0:	# const
		if type(instr['value']) == bool:
			return True
	elif index == 1:	# comparison eq/lt/gt/le/ge
		if len(instr['args']) == 2 and instr['args'][0] in context and instr['args'][1] in context:
			return True
	elif index == 2:	# logic not/and/or
		if instr['op'] == 'not':
			if len(instr['args']) == 1 and instr['args'][0] in context:
				return True
		else:
			if len(instr['args']) == 2 and instr['args'][0] in context and instr['args'][1] in context:
				return True
	elif index == 3:
		if len(instr['args']) == 1 and instr['args'][0] in context:
			return True

	return False


def check_c(instr, index, labels, b_context=[]):
	if index == 0:	# jmp
		if len(instr['args']) == 1 and instr['args'][0] in labels:
			return True
	elif index == 1:	# branch
		if len(instr['args']) == 3:
			if instr['args'][0] not in b_context:
				print('\tbranch cond undefined properly')
				return False
			elif instr['args'][1] not in labels or instr['args'][2] not in labels:
				print('\tbranch labels undefined')
				return False
			return True

	return False


def instr_check(instrs, labels):
	'''Check instrs one after another
	'''

	i = 1 # line number
	context = {'a':[], 'b':[]}	# 'a': arithmetic var; 'b': boolean var
	for instr in instrs:
		i += 1
		# print(i, instr)

		if 'op' in instr:
			# Const arithmetic
			if instr['op'] == 'const':
				if instr['type'] == 'int':
					if not check_a(instr, 0):
						print('line %d: assign non-int value to int var' % i)
						print('const error!')
						return False
					if instr['dest'] in context['b']:
						print('line %d: assign int value to existing bool var' % i)
						print('redefine const error!')
						return False
					if instr['dest'] not in context['a']:
						context['a'].append(instr['dest'])
				elif instr['type'] == 'bool':
					if not check_b(instr, 0):
						print('line %d: assign non-bool value to bool var' % i)
						print('const error!')
						return False
					if instr['dest'] in context['a']:
						print('line %d: assign bool value to existing int var' % i)
						print('redefine const error!')
						return False
					if instr['dest'] not in context['b']:
						context['b'].append(instr['dest'])
				else:
					print('line %d: assign non-int/bool types' % i)
					print('const error!')
					return False


			# arithmetic op
			elif instr['op'] in OP_ARITHMETIC:	
				if instr['type'] == 'int':
					if not check_a(instr, 1, context['a']):
						print('line %d:' % i)
						print('arithmetic op error!')
						return False
					context['a'].append(instr['dest'])
				else:
					print('line %d:' % i)
					print('arithmetic op error!')
					return False

			# comparison op
			elif instr['op'] in OP_COMP:
				if instr['type'] == 'bool':
					if not check_b(instr, 1, context['a']):
						print('line %d: assign non-bool values to bool var' % i)
						print('comparison op error!')
						return False
					context['b'].append(instr['dest'])
				else:
					print('line %d: assign bool values to non-bool var' % i)
					print('comparison op error!')
					return False

			# logic op
			elif instr['op'] in OP_LOGIC:
				if instr['type'] == 'bool':
					if not check_b(instr, 2, context['b']):
						print('line %d: assign non-bool values to bool var' % i)
						print('logic op error!')
						return False
					context['b'].append(instr['dest'])
				else:
					print('line %d: assign bool values to non-bool var' % i)
					print('logic op error!')
					return False

			# id
			elif instr['op'] == 'id':
				if instr['type'] == 'int':
					if not check_a(instr, 3, context['a']):
						print('line %d: id non-int var to int var' % i)
						print('id int error!')
						return False
					if instr['dest'] in context['b']:
						print('line %d: id int value to existing bool var' % i)
						print('redefine var error!')
						return False
					if instr['dest'] not in context['a']:
						context['a'].append(instr['dest'])
				elif instr['type'] == 'bool':
					if not check_b(instr, 3, context['b']):
						print('line %d: id non-bool var to bool var' % i)
						print('id bool error!')
						return False
					if instr['dest'] in context['a']:
						print('line %d: id bool value to existing int var' % i)
						print('redefine var error!')
						return False
					if instr['dest'] not in context['b']:
						context['b'].append(instr['dest'])

			# print values
			elif instr['op'] == 'print':
				for var in instr['args']:
					if var not in context['a'] and var not in context['b']:
						print('\tVar %s not defined' % var)
						print('line %d: print undefined vars' % i)
						print('print error!')
						return False

			# jmp
			elif instr['op'] == 'jmp':
				if not check_c(instr, 0, labels):
					print('line %d: jump to undefined label %s' % (i, instr['args'][0]))
					print('jmp error!')
					return False

			# branch
			elif instr['op'] == 'br':
				if not check_c(instr, 1, labels, context['b']):
					print('line %d: branch args undefined' % i)
					print('br error!')
					return False

			# return
			elif instr['op'] == 'ret':
				if len(instr['args']) != 0:
					print('line %d: return with arguments' % i)
					print('ret error!')
					return False

			# nop
			elif instr['op'] == 'nop':
				if len(instr['args']) != 0:
					print('line %d: nop with arguments' % i)
					print('nop error!')
					return False

	return True




def label_pass(instrs):
	'''A first pass for the Bril program to collect all the labels in 'label'.
	Return error if any label show up more than once.
	'''
	labels = []
	i = 1 # line number
	for instr in instrs:
		i += 1 
		if 'label' in instr:
			if instr['label'] in labels:
				print('line %d: label %s show up twice' % (i, instr['label']))
				return False
			labels.append(instr['label'])

	return labels


def typecheck(bril):
	'''Given a Bril program, print out the #lines have type check errors
	or type check successfully.
	'''
	func = bril['functions'][0]
	labels = label_pass(func['instrs'])
	if labels == False:
		print('label error!')
		return
	if not instr_check(func['instrs'], labels):
		return
	print('Type checking passed')
	return



#if __name__ == '__main__':
def typecheck_main():
	typecheck(json.load(sys.stdin))
