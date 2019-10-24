import json
import os
import subprocess
import sys

sys.path.append('bril-txt/')
import briltxt
import annotate_blocks
import block_reordering
import function_reordering


def profile(program, profile_file):
    program = annotate_blocks.annotate(program)
    result = subprocess.run(['brilprofile', profile_file], stdout=subprocess.PIPE, input=json.dumps(program).encode())
    return json.loads(result.stdout)


def optimize(bril_file, profile_file):
    with open(bril_file) as f:
        program = json.loads(briltxt.parse_bril(f.read()))
    
    uo_profile = profile(program, profile_file)
    bril_json = annotate_blocks.annotate(program)

    # Function reordering
    bril_of = function_reordering.reorder(bril_json, uo_profile)
    of_profile = profile(bril_of, profile_file)

    # Block reordering
    bril_ob = block_reordering.reorder(bril_json, uo_profile)
    ob_profile = profile(bril_ob, profile_file)

    # Block reordering after function reordering (order doesn't matter since they're orthogonal optimizations)
    bril_ofb = block_reordering.reorder(bril_of, of_profile)
    ofb_profile = profile(bril_ofb, profile_file)

    f_improvement = of_profile['ip_jumps'] / uo_profile['ip_jumps']
    b_improvement = ob_profile['ip_jumps'] / uo_profile['ip_jumps']
    fb_improvement = ofb_profile['ip_jumps'] / uo_profile['ip_jumps']
    print('Unoptimized:', 1.00)
    print('Functions:  ', f_improvement)
    print('Blocks:     ', b_improvement)
    print('F + B:      ', fb_improvement)
    return (p['ip_jumps'] for p in (uo_profile, of_profile, ob_profile, ofb_profile))


if __name__ == '__main__':
    test_file = sys.argv[1]
    out_file = sys.argv[2]
    if os.path.exists(out_file):
        print('Output file exists! Exiting.')
        sys.exit(1)
    with open(test_file) as tests, open(out_file, 'w') as output:
        for line in tests:
            prefix = line.strip()
            if not prefix:
                continue
            bril_file, profile_file = f'workload/{prefix}.bril', f'workload/{prefix}.in'
            uo, of, ob, ofb = optimize(bril_file, profile_file)
            output.write(f'{uo} {of} {ob} {ofb}\n' )
    print('Completed testing.')
