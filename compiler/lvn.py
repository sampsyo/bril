import json
import sys
from collections import defaultdict

COMMUTATIVE = {'add', 'mul'}


class VarMapping:
    def __init__(self):
        self.var_to_idx = dict()
        self.value_to_idx = dict()
        self.table = [] # idx is #, then contains tuples of (value, home)
        self.lvn_vars = 0

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

    def unroll_ids(self, value):
        print(value, file=sys.stderr)
        if value[0] == 'id':
            assert len(value) == 2
            return self.unroll_ids(self.idx_to_value(value[1]))
        return value

    def maybe_rename_dest_store_old(self, var_counts, dest, value):
        var_counts[dest] -= 1
        if var_counts[dest] > 0:
            # Insert old value so we know what it was when we try to use it
            self.insert_var(dest, value)
            dest = f'lvn.{self.lvn_vars}'
            self.lvn_vars += 1
            assert dest not in var_counts, f"Alas, {dest} is not a unique name"
        return dest

    def new_cache(self):
        new = VarMapping()
        new.lvn_vars = self.lvn_vars
        return new


def count_variables(func):
    ans = defaultdict(lambda : 0)
    for instr in func['instrs']:
        if 'dest' in instr:
            ans[instr['dest']] += 1
    return ans


def make_lookup(var_table, dest, type, value):
    if value[0] == 'const':
        return {
            'dest': dest,
            'op': 'const',
            'value': value[1],
            'type': type
        }
    else:
        if value[0] == 'id':
            idx = value[1]
        else:
            idx = var_table.value_to_idx[value]
        return {
            'dest': dest,
            'op': 'id',
            'args': [var_table.idx_to_home_var(idx)],
            'type': type
        }


def make_value(var_table, instr):
    op = instr['op']

    # Replace every arg with its variable index
    if 'args' in instr:
        args = []
        for arg in instr['args']:
            # We can actually refer to an argument that hasn't been defined
            # (like a function arg, or after a jump), so if it's not there
            # then add it.
            if arg not in var_table.var_to_idx:
                #assert arg is str, f"tried to put invalid {arg} as argument"
                var_table.insert_var(arg, ("arg", arg))
            args.append(var_table.var_to_idx[arg])
    elif 'value' in instr:
        args = [instr['value']]
    else:
        assert False, f"idk what to do with {instr}"

    # Transform args to canonical ordering if op is commutative
    if op in COMMUTATIVE:
        args = sorted(args)

    value = op, *args
    return value


def do_lvn():
    prog = json.load(sys.stdin)
    # does this happen within a function or across a program?
    for func in prog['functions']:
        var_table = VarMapping()
        if 'args' in func:
            for arg in func['args']:
                var_table.insert_var(arg['name'], ('arg', arg['name']))

        var_counts = count_variables(func)
        new_instrs = []
        for instr in func['instrs']:
            for key in instr.keys():
                assert key in {'op', 'dest', 'args', 'type', 'funcs', 'value', 'label', 'labels'}, \
                   f"Unrecognized instruction key {key}, code might not work..."

            if 'op' in instr and instr['op'] not in {'jmp', 'br'}:
                value = make_value(var_table, instr)

                if 'dest' in instr:
                    dest = instr['dest']

                    # If the value is already there, replace instruction with lookup
                    if value in var_table.value_to_idx or instr['op'] == 'id':
                        old_instr = instr
                        value = var_table.unroll_ids(value)
                        instr = make_lookup(var_table, dest, instr['type'], value)
                        value = make_value(var_table, instr)
                        print(f"replacement {old_instr} -> {instr} -> {value}", file=sys.stderr)

                    # If variable is going to be overwritten again...
                    dest = var_table.maybe_rename_dest_store_old(var_counts, dest, value)

                    # Put the value in the destination
                    var_table.insert_var(dest, value)
                    instr['dest'] = dest

                if 'args' in instr:
                    new_args = [var_table.idx_to_home_var(idx) for idx in value[1:]]
                    # Replace arg indices with their home variables
                    instr['args'] = new_args
                elif 'value' in instr:
                    instr['value'] = value[1]
                new_instrs.append(instr)

            else:
                # If it's a label instruction, then we might have jumped and
                # so we need to drop the cache
                if 'label' in instr:
                    var_table = var_table.new_cache()

                # if its not an op, then just put it back unchanged
                new_instrs.append(instr)

        func['instrs'] = new_instrs
    print(json.dumps(prog))


if __name__ == '__main__':
    do_lvn()