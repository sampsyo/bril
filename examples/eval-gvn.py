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

do_not_include = ['test/gvn/briggs-et-al-fig-5.bril', 'test/gvn/equivalent-phis.bril', 'test/gvn/cyclic-phi-handling.bril', 'test/gvn/redundant-store-across-block.bril', 'test/gvn/divide-by-zero.bril']

def graph(name, graph_data, y_max, y_space, figsize):
	graph_data = graph_data.sort_values(['Type']).reset_index(drop=True).sort_values(['Instruction count'], ascending=False).reset_index(drop=True)
	plt.rcParams['figure.figsize'] = figsize
	ax = sns.barplot(x="Test", y="Instruction count", hue="Type", data=graph_data)
	locs, labels = plt.xticks()
	plt.xticks(locs, labels, rotation='vertical')
	ax.set_yticks(np.arange(0, y_max, y_space), minor=True)
	ax.grid(which='minor', alpha=0.6)
	plt.tick_params(axis='x', labelsize=8)
	plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
	#ax.set_aspect(aspect=.05)
	plt.savefig(name, bbox_inches='tight')
	plt.close()


def main():
	sns.set(style="whitegrid")
	bril_data = pd.DataFrame(columns=['Test', 'Type', 'Instruction count'])
	ts_data = pd.DataFrame(columns=['Test', 'Type', 'Instruction count'])
	for fn in glob.glob('test/gvn/*.bril'):
		# the briggs & al. example is only for correctness
		if fn in do_not_include:
			continue
		small_name = fn.split('/')[-1].split('.')[0]
		original = subprocess.check_output('cat {} | bril2json | python examples/instruction_count.py'.format(fn), shell=True).strip().decode("utf-8")
		ssa = subprocess.check_output('cat {} | bril2json | python examples/bril2ssa.py | python examples/SSA2bril.py | python examples/tdce.py | python examples/instruction_count.py'.format(fn), shell=True).strip().decode("utf-8")
		lvn = subprocess.check_output('cat {} | bril2json | python examples/bril2ssa.py | python examples/lvn.py | python examples/SSA2bril.py | python examples/tdce.py | python3 examples/instruction_count.py'.format(fn), shell=True).strip().decode("utf-8")
		gvn= subprocess.check_output('cat {} | bril2json | python examples/bril2ssa.py | python examples/global_value_numbering.py | python examples/SSA2bril.py | python examples/tdce.py | python3 examples/instruction_count.py'.format(fn), shell=True).strip().decode("utf-8")
		print('| `{}` | {} | {} | {} | {} |'.format(fn, original, ssa, lvn, gvn))
		if 'ts' in fn.split('/')[-1].split('.'):
			ts_data = ts_data.append({'Test': small_name, 'Type': 'SSA', 'Instruction count': int(ssa)}, ignore_index=True)
			ts_data = ts_data.append({'Test': small_name, 'Type': 'LVN', 'Instruction count': int(lvn)}, ignore_index=True)
			ts_data = ts_data.append({'Test': small_name, 'Type': 'GVN', 'Instruction count': int(gvn)}, ignore_index=True)
		else:
			bril_data = bril_data.append({'Test': small_name, 'Type': 'SSA', 'Instruction count': int(ssa)}, ignore_index=True)
			bril_data = bril_data.append({'Test': small_name, 'Type': 'GVN', 'Instruction count': int(gvn)}, ignore_index=True)
	graph('eval-correctness.pdf', bril_data, y_max=41, y_space=5, figsize=(6,2))
	graph('eval-ts.pdf', ts_data, y_max=81, y_space=10, figsize=(12,3))
		

if __name__ == main():
	main()