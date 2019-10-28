import os
import subprocess
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def main():
	sns.set(style="whitegrid")
	graph_data = pd.DataFrame(columns=['test', 'type', 'instruction count'])
	for fn in glob.glob('test/gvn/*.bril'):
		original = subprocess.check_output('cat {} | bril2json | python examples/instruction_count.py'.format(fn), shell=True).strip().decode("utf-8")
		ssa = subprocess.check_output('cat {} | bril2json | python examples/bril2ssa.py | python examples/ssa2bril.py | python examples/tdce.py | python examples/instruction_count.py'.format(fn), shell=True).strip().decode("utf-8")
		lvn = subprocess.check_output('cat {} | bril2json | python examples/bril2ssa.py | python examples/lvn.py | python examples/ssa2bril.py | python examples/tdce.py | python3 examples/instruction_count.py'.format(fn), shell=True).strip().decode("utf-8")
		gvn = subprocess.check_output('cat {} | bril2json | python examples/bril2ssa.py | python examples/global_value_numbering.py | python examples/ssa2bril.py | python examples/tdce.py | python3 examples/instruction_count.py'.format(fn), shell=True).strip().decode("utf-8")
		print('| `{}` | {} | {} | {} | {} |'.format(fn, original, ssa, lvn, gvn))
		graph_data = graph_data.append({'test': fn, 'type': 'ssa', 'instruction count': int(ssa)}, ignore_index=True)
		graph_data = graph_data.append({'test': fn, 'type': 'gvn', 'instruction count': int(gvn)}, ignore_index=True)
	ax = sns.barplot(x="test", y="instruction count", hue="type", data=graph_data)
	plt.savefig('eval.pdf', bbox_inches='tight')
		

if __name__ == main():
	main()