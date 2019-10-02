import json
import sys
import random

"""
Select betwen the various versions of python
and numpy that exist on computers I've been writing on.
"""
def choose(arr, w):
    if 'choices' in random.__dict__:
        return random.choices(arr, w)[0]

    # try:
    #     import numpy
    #     return numpy.random.choice(arr, p=w)
    # except:
    z = random.random() * sum(w)
    total = 0
    for a,p in zip(arr,w):
        total += p
        if z <= total:
            return a


def freshL(prefix, names):
    # if prefix not in names:
    #     if type(names) == set:
    #         names.add(prefix)
    #     return prefix

    i = 1
    while True:
        name = prefix + str(i)
        if name not in names:
            if type(names) == set:
                names.add(name);
            return name
        i += 1

if __name__ == '__main__':
    probsndecay = dict(recombine = (5,0.5),
        const = (5, 0.3),
        flip = (4, 0.5),
        label = (10, 0.1),
        br = (2, 0.5),
        jmp = (0.5, 0.5),
        ret = (0.5, 0.5) )

    recombinations = { \
        'and' : (bool,bool,bool),
        'or' : (bool,bool,bool),
        'not' : (bool,bool),
        'gt' : (int,int,bool),
        'lt' : (int,int,bool),
        'ge' : (int,int,bool),
        'le' : (int,int,bool),
        'add' : (int,int,int),
        'mul' : (int,int,int),
        'sub' : (int,int,int),
        'div' : (int,int,int)
    }

    # probs = { l : p for (l,(p, d)) in probsndecay.items() }
    # decays = { l : d for (l,(p, d)) in probsndecay.items() }

    prog_length = int(random.gammavariate(6, 4));

    variables = {} # name => type
    labels = set()
    mainfunc = {'instrs' : [], 'name' : 'main'}
    prevInstr = {}

    def randVarOf( vtype ):
        return random.choice([v for (v,t) in variables.items() if t == vtype]);
    def newv( prefix ):
        return freshL(prefix, variables.keys())

    while len(mainfunc['instrs']) < prog_length:
        ops, probs, decays = zip( *( (l,p,d) for l, (p,d) in probsndecay.items() ) )
        if 'op' in prevInstr and prevInstr['op'] in {'br', 'jmp'} :
            instr_type, (prob, decay) = 'label', probsndecay['label']
            prob = prob / decay
            prog_length -= 1
        else:
            instr_type, prob, decay =  choose(list(zip(ops, probs, decays)), probs)

        if instr_type == 'recombine':
            op, optypes = random.choice(list(recombinations.items()))

            # if !all(otypes[:-1])
            try:
                args = [ randVarOf(t) for t in optypes[:-1] ]
                dest = randVarOf(optypes[-1]) if random.randrange(2) else newv('v')
            except IndexError:
                continue

            instruction = dict(op=op, type = optypes[-1].__name__,
                args=args, dest=dest )
            variables[dest] = optypes[-1]

        elif instr_type == 'const':
            typ = bool if random.randrange(2) == 0 else int
            val = typ(int(random.expovariate(0.1)))
            dest = newv('c')

            instruction = dict(op='const', value=val, type=typ.__name__, dest=dest)
            variables[dest] = typ;

        elif instr_type == 'flip':
            try:
                dest = randVarOf(bool) if random.randrange(2) else newv('b')
                instruction = dict(op = 'flip', args = [], dest=dest, type='bool')
                variables[dest] = bool
            except IndexError:
                continue

        elif instr_type == 'br':
            try:
                l1 = random.choice([*labels])
                l2 = random.choice([*labels])
                switch = randVarOf(bool)

                if l1 == l2:
                    raise IndexError("branch on different labels")

                instruction = dict(op = 'br', args = [switch, l1, l2])
                prog_length += 1
            except IndexError as i:
                # probsndecay[instr_type] = (prob * , decay)
                continue

        elif instr_type == 'jmp': # # NOTE: : only generate upwards jumps.
            try:
                l = random.choice([*labels])
                instruction = dict(op='jmp', args = [l])
                prog_length += 1
            except IndexError:
                continue

        elif instr_type == 'ret':
            instruction = dict(op = 'ret', args=[])

        elif instr_type == 'label':
            if 'label' in prevInstr:
                continue

            instruction = dict(label = freshL('lab', labels))
            prog_length += 1

        # print(labels, variables)

        probsndecay[instr_type] = (prob * decay, decay)
        # print( instr_type )

        mainfunc['instrs'].append( instruction );
        prevInstr = instruction

    for i,instr in enumerate(mainfunc['instrs']) :
        if 'op' in instr:
            if instr['op'] is 'jmp':
                instr['args'] = [  random.choice([*labels]) ]
            if instr['op'] is 'br':
                instr['args'] = [  random.choice([*labels]), mainfunc['instrs'][i+1]['label'] ]
                random.shuffle(instr['args'])



    json.dump({'functions' : [ mainfunc ]}, sys.stdout, indent=2, sort_keys=True)
