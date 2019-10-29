import os
import subprocess
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

plt.rcParams['patch.linewidth'] = 0
plt.rcParams['patch.edgecolor'] = 'none'
plt.rcParams["patch.force_edgecolor"] = False
#plt.rcParams['figure.figsize']=(12,3)
plt.rcParams['figure.figsize']=(6,2)

do_not_include = ['test/gvn/briggs-et-al-fig-5.bril', 'test/gvn/equivalent-phis.bril', 'test/gvn/cyclic-phi-handling.bril', 'test/gvn/redundant-store-across-block.bril', 'test/gvn/divide-by-zero.bril']

def main():
	sns.set(style="whitegrid")
	graph_data = pd.DataFrame(columns=['Test', 'Type', 'Instruction count'])
	for fn in glob.glob('test/gvn/*.bril'):
		# the briggs & al. example is only for correctness
		if fn in do_not_include or 'ts' in fn.split('/')[-1].split('.'):
			continue
		small_name = fn.split('/')[-1].split('.')[0]
		original = subprocess.check_output('cat {} | bril2json | python examples/instruction_count.py'.format(fn), shell=True).strip().decode("utf-8")
		ssa = subprocess.check_output('cat {} | bril2json | python examples/bril2ssa.py | python examples/SSA2bril.py | python examples/tdce.py | python examples/instruction_count.py'.format(fn), shell=True).strip().decode("utf-8")
		lvn = subprocess.check_output('cat {} | bril2json | python examples/bril2ssa.py | python examples/lvn.py | python examples/SSA2bril.py | python examples/tdce.py | python3 examples/instruction_count.py'.format(fn), shell=True).strip().decode("utf-8")
		gvn= subprocess.check_output('cat {} | bril2json | python examples/bril2ssa.py | python examples/global_value_numbering.py | python examples/SSA2bril.py | python examples/tdce.py | python3 examples/instruction_count.py'.format(fn), shell=True).strip().decode("utf-8")
		print('| `{}` | {} | {} | {} | {} |'.format(fn, original, ssa, lvn, gvn))
		graph_data = graph_data.append({'Test': small_name, 'Type': 'SSA', 'Instruction count': int(ssa)}, ignore_index=True)
		#graph_data = graph_data.append({'Test': small_name, 'Type': 'LVN', 'Instruction count': int(lvn)}, ignore_index=True)
		graph_data = graph_data.append({'Test': small_name, 'Type': 'GVN', 'Instruction count': int(gvn)}, ignore_index=True)
	graph_data = graph_data.sort_values(['Type']).reset_index(drop=True).sort_values(['Instruction count'], ascending=False).reset_index(drop=True)
	ax = sns.barplot(x="Test", y="Instruction count", hue="Type", data=graph_data)
	locs, labels = plt.xticks()
	plt.xticks(locs, labels, rotation='vertical')
	ax.set_yticks(np.arange(0,41,5), minor=True)
	ax.grid(which='minor', alpha=0.6)
	plt.tick_params(axis='x', labelsize=8)
	plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
	#ax.set_aspect(aspect=.05)
	plt.savefig('eval.pdf', bbox_inches='tight')
		

if __name__ == main():
	main()