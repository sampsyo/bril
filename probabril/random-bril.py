import json
import sys
import random

def freshL(prefix, names):
    if prefix not in names:
        names.add(prefix)
        return prefix
        
    i = 1
    while True:
        name = prefix + str(i)
        if name not in names:
            names.add(name);
            return name
        i += 1
    
if __name__ == '__main__':
    probs = dict(recombine = 0.2,
        const = 0.1,
        flip = 0.25,
        label = 0.2,
        br = 0.2,
        jmp = 0.05)
    
    recombinations = {
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
    
    
    prog_length = int(random.gammavariate(5, 4));
    
    variables = {} # name => type
    labels = set()
    mainfunc = {'instrs' : []}
    
    def randVarOf( vtype ):
        return random.choice([v for v,t in variables if t == vtype]);
    def newv( prefix ):
        return freshL(prefix, variables.keys())
    
    while len(mainfunc['instrs']) < prog_length:
        instr_type =  random.choices(*zip(*probs.items()))
        
        if instr_type == 'recombine':
            op, optypes = random.choice(recombination.items())
            
            # if !all(otypes[:-1])
            try:
                args = [ randVarOf(t) for t in otypes[:-1] ]
                dest = random.randrange(2) ? randVarOf(optypes[-1]) : newv(y)
            except IndexError:
                continue
                
            
            instruction = dict(op=op, type = optypes[-1].__name__, 
                args=args, dest=dest )
            
        elif(instr_type == 'const':
            typ = random.randrange(2) == 0 ? bool : int
            val = typ(int(random.expovariate(0.1)));
            instruction = dict(op='const', 
                value=val, type=typ.__name__, 
                dest=newv('c') )
                
        elif(instr_type == 'flip'):
            pass
        elif(instr_type == 'label'):
            pass
        elif(instr_type == 'br'):
            pass
        elif(instr_type == 'jmp'):
            pass
                
        mainfunc['instrs'].append( instruction );
        
        
    # be super safe. Check for this name beforehand.
    RESTART_LABEL = freshL(RESTART_LABEL, labels);    
    
    for func in bril['functions']:
        # uh-oh. How do cross function jumps work??
        if func['name'] == 'main':
            func['instrs'].insert(0, {"label": RESTART_LABEL});
            
            for i,instr in enumerate(func['instrs']):
                if 'op' in instr and instr['op'] == 'obv':
                    instr['op'] = 'br';
                    nextlabel = freshL("good-observation", labels)
                    instr['args'] = [instr['args'][0], nextlabel, RESTART_LABEL]
                    func['instrs'].insert(i+1, {'label': nextlabel})
    
    #lvn(bril, '-p' in sys.argv, '-c' in sys.argv, '-f' in sys.argv)
    json.dump(bril, sys.stdout, indent=2, sort_keys=True)
