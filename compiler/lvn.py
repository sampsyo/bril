import json
import sys
from collections import defaultdict
from functools import reduce

DEBUG = False
COMMUTATIVE = {'add', 'mul', 'or', 'and', 'eq'}
CAST_VALUE = {
    'int': int,
    'bool': bool
}
EXPRESSIONS = {
    'add': lambda a, b: a + b,
    'mul': lambda a, b: a * b,
    'sub': lambda a, b: a - b,
    'div': lambda a, b: a / b,
    'eq': lambda a, b: a == b,
    'lt': lambda a, b: a < b,
    'gt': lambda a, b: a > b,
    'le': lambda a, b: a <= b,
    'ge': lambda a, b: a >= b,
    'not': lambda a: not a,
    'and': lambda a, b: a and b,
    'or': lambda a, b: a or b
}


class Simplification:
    def __init__(self, trigger, val, result):
        assert trigger in {'either', '2'}
        self.trigger = trigger
        self.val = val
        self.result = result

    def simplify(self, arg1, arg2):
        if self.trigger == 'either':
            if self.val == arg1:
                return self.render_result(arg2)
        if self.val == arg2:
            return self.render_result(arg1)
        # return None if this simplification doesn't do anything
        return None

    def render_result(self, other):
        res = other if self.result == 'identity' else self.result
        return res


SIMPLIFICATIONS = {
    'add': [Simplification('either', 0, 'identity')],
    'mul': [Simplification('either', 1, 'identity')],
    'sub': [Simplification('2', 0, 'identity')],
    'div': [Simplification('2', 1, 'identity')],
    'and': [Simplification('either', True, 'identity'),
            Simplification('either', False, False)],
    'or':  [Simplification('either', False, 'identity'),
            Simplification('either', True, True)]
}


def compute_expression(value, const_vals, do_special_cases=False):
    op = value[0]
    val = None

    if op in SIMPLIFICATIONS and do_special_cases:
        for simplification in SIMPLIFICATIONS[op]:
            ans = simplification.simplify(*const_vals)
            if ans is not None:
                if isinstance(ans, tuple):
                    assert ans[0] == 'variable'
                    return ('id', value[1], ans[1])
                else:
                    return ('const', value[1], ans)

    is_consts = [not isinstance(val, tuple) for val in const_vals]
    if reduce(lambda a, b: a and b, is_consts, True):
        try:
            val = EXPRESSIONS[value[0]](*const_vals)
            return ('const', value[1], val)
        except Exception:
            return value
    else:
        return value


def debug_msg(*args):
    if DEBUG:
        print(*args, file=sys.stderr)


class VarMapping:
    def __init__(self, compute_constant_ops=True, do_special_cases=True):
        self.var_to_idx = dict()
        self.value_to_idx = dict()
        self.table = [] # idx is #, then contains tuples of (value, home)
        self.compute_constant_ops = compute_constant_ops
        self.do_special_cases = do_special_cases

    def insert_var(self, var, value):
        # If it's not already in the table, add it
        if value not in self.value_to_idx:
            idx = len(self.table)
            self.table.append((value, var))
            # This is only set on insert, so it is correct
            self.value_to_idx[value] = idx

        # always map the variable to the index
        # this means we remap on reassignment, which is correct
        self.var_to_idx[var] = self.value_to_idx[value]

    def _idx_to_home_var(self, idx):
        return self.table[idx][1]

    def _idx_to_value(self, idx):
        return self.table[idx][0]

    def unroll_ids(self, value):
        if value[0] == 'id':
            assert len(value) == 3
            return self.unroll_ids(self._idx_to_value(value[2]))
        return value

    def add_unseen_variables(self, args):
        for arg in args:
            if arg not in self.var_to_idx:
                self.insert_var(arg, ('arg', arg))

    def _replace_indices_through_idx(self, indices):
        ans = []
        for idx in indices:
            value = self._idx_to_value(idx)
            value = self.unroll_ids(value)
            ans.append(self.value_to_idx[value])
        return ans

    def _const_value_from_index(self, idx):
        value = self._idx_to_value(idx)
        if value[0] == 'const':
            return CAST_VALUE[value[1]](value[2])
        else:
            # This needs to be an enum or something if we ever handle tuples
            return ('variable', idx)

    def make_value(self, instr):
        op = instr['op']
        type = instr['type'] if 'type' in instr else None
        if 'args' in instr:
            self.add_unseen_variables(instr['args'])
            indices = [self.var_to_idx[arg] for arg in instr['args']]

            indices = self._replace_indices_through_idx(indices)
            if op in COMMUTATIVE:
                indices = sorted(indices)

            value = op, type, *indices
            if op not in {'const', 'id'} and self.compute_constant_ops:
                vals = [self._const_value_from_index(idx) for idx in value[2:]]
                value = compute_expression(
                    value, vals, do_special_cases=self.do_special_cases
                )
            return value
        elif 'value' in instr:
            return (op, instr['type'], instr['value'])
        else:
            assert False, f"idk how to make value for {instr}"

    def indices_to_vars(self, indices):
        return [self._idx_to_home_var(idx) for idx in indices]

    def value_to_home_var(self, value):
        return self._idx_to_home_var(self.value_to_idx[value])


def count_variables(func):
    ans = defaultdict(lambda : 0)
    for instr in func['instrs']:
        if 'dest' in instr:
            ans[instr['dest']] += 1
    return ans


def const_instr(dest, value):
    assert value[0] == 'const'
    assert len(value) == 3
    return {'dest': dest, 'op': 'const', 'value': value[2], 'type': value[1]}


def id_instr(dest, type, home_var):
    return {'dest': dest, 'op': 'id', 'args': [home_var], 'type': type}


def name_dest(var_counts, lvn_count, dest):
    var_counts[dest] -= 1
    if var_counts[dest] > 0:
        dest = f'lvn.{dest}.{lvn_count}'
        assert dest not in var_counts, f"Alas, {dest} is used"
    return dest


def do_lvn():
    prog = json.load(sys.stdin)
    renamed_var_count = 0

    for func in prog['functions']:
        var_table = VarMapping()
        if 'args' in func:
            var_table.add_unseen_variables([
                arg['name'] for arg in func['args']
            ])

        var_counts = count_variables(func)
        new_instrs = []
        for instr in func['instrs']:
            for key in instr.keys():
                assert key in {'op', 'dest', 'args', 'type', 'funcs', 'value',
                               'label', 'labels'}, \
                   f"Unrecognized instruction key {key}"

            if 'op' in instr and instr['op'] not in {'jmp', 'br'}:
                old_instr = instr
                old_value = var_table.make_value(instr)

                # If we simplified the value into a constant, replace the instr
                if old_value[0] == 'const' and instr['op'] != 'const':
                    instr = const_instr(instr['dest'], old_value)
                    debug_msg(f"Replaced {old_instr} w/ constant {instr} when got value")
                    old_instr = old_instr
                debug_msg(f"Converted instruction {instr} to value {old_value}")

                value = var_table.unroll_ids(old_value)
                debug_msg(f"Unrolled id lookups for {old_value} -> {value}")

                if 'dest' in instr:
                    dest = instr['dest']

                    # If the value is already there, replace with id or constant
                    if value in var_table.value_to_idx:
                        home_var = var_table.value_to_home_var(value)
                        if value[0] == 'const':
                            instr = const_instr(dest, value)
                        else:
                            instr = id_instr(dest, instr['type'], home_var)
                            value = var_table.make_value(instr)

                    # If the var gets used again, rename it
                    new_dest = name_dest(var_counts, renamed_var_count, dest)
                    if new_dest != dest:
                        var_table.insert_var(dest, value)
                        instr['dest'] = dest = new_dest
                        renamed_var_count += 1

                    # Put the value in the destination
                    var_table.insert_var(dest, value)

                    debug_msg(f"{old_instr} -> {instr} -> {value}")

                if 'args' in instr:
                    # Replace arg indices with their home variables
                    debug_msg(f"Table: {var_table.table}")
                    instr['args'] = var_table.indices_to_vars(value[2:])

                if value[0] == 'const':
                    debug_msg(f"Constant {instr} should describe {value}")
                    assert instr['op'] == 'const'
                    assert instr['type'] == value[1]
                    assert instr['value'] == value[2]

                new_instrs.append(instr)

            else:
                # If it's a label instruction, then we might have jumped and
                # so we need to drop the cache
                if 'label' in instr:
                    var_table = VarMapping()

                # if its not an op, then just put it back unchanged
                new_instrs.append(instr)

        func['instrs'] = new_instrs
    print(json.dumps(prog))


if __name__ == '__main__':
    do_lvn()