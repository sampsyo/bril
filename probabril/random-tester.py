from random_bril import gencode
import subprocess
import sys, re
import json
from scipy.stats import chisquare



spacere = r'\s+'
def nospace_str(stdout) :
    return re.sub(spacere, '', stdout.decode('utf-8'))

def getfreq(counts, key):
    if key.startswith("\'done\'"):
        return counts.get(key[7:],0)
    return 0

N  = 100
if __name__ == '__main__':
    n_errs = 0
    for i in range(10):
        program = gencode()
        progbytes = bytes(json.dumps(program, sort_keys=True), 'utf-8')
        
        xbrili = subprocess.run(['xbrili'], input=progbytes, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        erstr = nospace_str(xbrili.stderr)
        if  len(erstr) > 1:
            n_errs += 1
            # print("ERROR", erstr)
            continue
        
        paths = nospace_str(xbrili.stdout).split('[')
        
        dist = {}
        for p in paths: 
            if len(p) < 2:
                continue
            key, val = p.split(']')
            dist[key] = float(val)
            
        # print(dist)
        counts = {}
        for  j in range(N):
            # sys.stdout.write('.')
            # sys.stdout.flush()

            try:
                brili_j = subprocess.run(['brili', '--envdump', '--noprint'], input =progbytes, stdout=subprocess.PIPE, timeout=2)
            
                env = nospace_str(brili_j.stdout)
                counts[env] = counts.get(env,0) + 1
            except subprocess.TimeoutExpired:
                continue
        # print(counts)
        # print(list(zip(*(round(dist[k] * N), getfreq(counts,k)) for k in dist.keys() )))
        try:
            f_exp, f_obs = zip(* [(round(dist[k] * N), getfreq(counts,k)) for k in dist.keys() if k.startswith("\'done\'")])
        except ValueError:
            f_exp = []
            f_obs = []
            continue
        chisq,p = chisquare(f_obs, f_exp)
        print('chi2', chisq, 'p', p,'\t', f_exp, f_obs)

    print('random programs that could not be executed by xbrili', n_errs)
