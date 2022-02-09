import json
import sys
from collections import defaultdict

COMMUTATIVE = {'add', 'mul'}


class VarMapping:
    def __init__(self):
        self.var_to_idx = dict()
        self.value_to_idx = dict()
        self.table = [] # idx is #, then contains tuples of (value, home)

    def insert_var(self, var, value):
        # If it's not already in the table, add it
        if value not in self.value_to_idx:
            idx = len(self.table)
            self.table.append((value, var))
            self.value_to_idx[value] = idx

        # always map the variable to the index
        self.var_to_idx[var] = self.value_to_idx[value]

    def var_to_idx(self, var):
        # This gets set for every variable, and is remapped on reassignment
        return self.var_to_idx[var]

    def value_to_idx(self, value):
        # This only gets set the first time we insert, so this code is correct
        return self.value_to_idx[value]

    def idx_to_home_var(self, idx):
        # This needs to be unpacked to go through ids
        value = self.table[idx][0]
        if value[0] == 'id':
            assert len(value) == 2
            return self.idx_to_home_var(value[1])
        return self.table[idx][1]

    def idx_to_value(self, idx):
        # This need to be unpacked to go through ids
        value = self.table[idx][0]
        if value[0] == 'id':
            assert len(value) == 2
            return self.idx_to_value(value[1])
        return value

    def value_to_home_var(self, value):
        return self.idx_to_home_var(self.value_to_idx[value])

    def unroll_ids(self, value):
        if value[0] == 'id':
            assert len(value) == 2
            return self.unroll_ids(self.idx_to_value(value[1]))
        return value

    def add_unseen_variables(self, args):
        for arg in args:
            if arg not in self.var_to_idx:
                self.insert_var(arg, ('arg', arg))

    def vars_to_indices(self, args):
        return [self.var_to_idx[arg] for arg in args]

    def indices_to_vars(self, indices):
        return [self.idx_to_home_var(idx) for idx in indices]

    def make_value(self, instr):
        op = instr['op']
        if 'args' in instr:
            self.add_unseen_variables(instr['args'])
            args = self.vars_to_indices(instr['args'])
        elif 'value' in instr:
            args = [instr['value']]
        else:
            assert False, f"idk how to make value for {instr}"

        if op in COMMUTATIVE:
            args = sorted(args)

        return op, *args


def count_variables(func):
    ans = defaultdict(lambda : 0)
    for instr in func['instrs']:
        if 'dest' in instr:
            ans[instr['dest']] += 1
    return ans


def const_instr(dest, type, value):
    assert value[0] == 'const'
    assert len(value) == 2
    return {'dest': dest, 'op': 'const', 'value': value[1], 'type': type}


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

    # does this happen within a function or across a program?
    for func in prog['functions']:
        var_table = VarMapping()
        if 'args' in func:
            var_table.add_unseen_variables(func['args'])

        var_counts = count_variables(func)
        new_instrs = []
        for instr in func['instrs']:
            for key in instr.keys():
                assert key in {'op', 'dest', 'args', 'type', 'funcs', 'value',
                               'label', 'labels'}, \
                   f"Unrecognized instruction key {key}"

            if 'op' in instr and instr['op'] not in {'jmp', 'br'}:
                value = var_table.make_value(instr)
                value = var_table.unroll_ids(value)

                if 'dest' in instr:
                    dest = instr['dest']

                    # If the value is already there, replace with id
                    if value in var_table.value_to_idx:
                        home_var = var_table.value_to_home_var(value)
                        instr = id_instr(dest, instr['type'], home_var)
                        value = var_table.make_value(instr)

                    # If we're iding a constant, replace it with the constant
                    if instr['op'] == 'id':
                        if value[0] == 'const':
                            instr = const_instr(dest, instr['type'], value)

                    # If the var gets used again, rename it
                    new_dest = name_dest(var_counts, renamed_var_count, dest)
                    if new_dest != dest:
                        var_table.insert_var(dest, value)
                        instr['dest'] = dest = new_dest
                        renamed_var_count += 1

                    # Put the value in the destination
                    var_table.insert_var(dest, value)

                    print(f"{old_instr} -> {instr} -> {value}", file=sys.stderr)


                if 'args' in instr:
                    # Replace arg indices with their home variables
                    instr['args'] = var_table.indices_to_vars(value[1:])
                elif 'value' in instr:
                    assert len(value) == 2
                    instr['value'] = value[1]

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