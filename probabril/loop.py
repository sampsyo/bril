import json
import sys


RESTART_LABEL = "RESTART"


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
    bril = json.load(sys.stdin)
    
    labels = set()
    for func in bril['functions']:
        for inst in func['instrs']:
            if 'label' in inst :
                labels.add(inst['label'])
        
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
